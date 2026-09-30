//! The agent loop on Rig's model layer: stream a reply, turn its tool calls
//! into [`ToolCall`] entities, wait for every [`ToolOutput`], send the
//! results back, repeat until the model answers without tools.

use std::collections::VecDeque;

use bevy::prelude::*;
use crossbeam_channel::Receiver;
use futures::StreamExt;
use rig_core::DynModel;
use rig_core::completion::{CompletionRequest, CompletionResponse, ToolDefinition, Usage};
use rig_core::message::{Message, ToolCall as ModelToolCall, ToolResultContent};
use rig_core::operation::Completion;
use rig_core::streaming::{Item, StreamEvent};
use serde::Serialize;

use crate::hot::{CRATE_DIR, HotReload, PatchStatus};
use crate::tools::{ToolCall, ToolOutput, ToolSpec};

/// Model rounds per prompt before the agent gives up.
const MAX_ROUNDS: usize = 40;

/// A prompt from the TUI or from BRP.
#[derive(Message, Clone)]
pub struct Submit(pub String);

/// Stop the current turn.
#[derive(Message, Clone, Default)]
pub struct Cancel;

/// Forget the conversation.
#[derive(Message, Clone, Default)]
pub struct Clear;

/// What the user sees, in order.
#[derive(Resource, Default)]
pub struct Transcript {
    pub entries: Vec<Entry>,
    streaming: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Entry {
    User {
        text: String,
    },
    Assistant {
        text: String,
    },
    Tool {
        name: String,
        arguments: String,
        output: Option<String>,
        is_error: bool,
    },
    Info {
        text: String,
    },
    Error {
        text: String,
    },
}

impl Transcript {
    pub fn push(&mut self, entry: Entry) -> usize {
        self.streaming = false;
        self.entries.push(entry);
        self.entries.len() - 1
    }

    fn stream_text(&mut self, fragment: &str) {
        if self.streaming
            && let Some(Entry::Assistant { text }) = self.entries.last_mut()
        {
            text.push_str(fragment);
            return;
        }
        self.push(Entry::Assistant {
            text: fragment.to_owned(),
        });
        self.streaming = true;
    }
}

pub enum Turn {
    Idle,
    Thinking {
        events: Receiver<ModelEvent>,
        task: tokio::task::JoinHandle<()>,
        round: usize,
        streamed: bool,
    },
    Tools {
        calls: Vec<Pending>,
        round: usize,
    },
}

pub struct Pending {
    entity: Entity,
    call: ModelToolCall,
    entry: usize,
}

pub enum ModelEvent {
    Text(String),
    Done(Box<CompletionResponse>),
    Failed(String),
}

#[derive(Resource)]
pub struct Agent {
    model: DynModel<Completion>,
    pub model_name: String,
    /// Ask for Anthropic's automatic prompt caching; every round resends
    /// the whole conversation.
    cache_prompts: bool,
    /// Tokens used this session.
    pub usage: Usage,
    runtime: tokio::runtime::Runtime,
    history: Vec<Message>,
    pub turn: Turn,
    queue: VecDeque<String>,
}

impl Agent {
    pub fn new(
        model: DynModel<Completion>,
        model_name: String,
        cache_prompts: bool,
        runtime: tokio::runtime::Runtime,
    ) -> Self {
        Self {
            model,
            model_name,
            cache_prompts,
            usage: Usage::default(),
            runtime,
            history: Vec::new(),
            turn: Turn::Idle,
            queue: VecDeque::new(),
        }
    }

    pub fn busy(&self) -> bool {
        !matches!(self.turn, Turn::Idle)
    }

    fn start(&mut self, specs: &Query<&ToolSpec>, round: usize) {
        // Sorted, so the prompt prefix stays byte-identical for caching.
        let mut specs: Vec<&ToolSpec> = specs.iter().collect();
        specs.sort_by(|a, b| a.name.cmp(&b.name));
        let tools = specs
            .into_iter()
            .map(|spec| ToolDefinition {
                name: spec.name.clone(),
                description: spec.description.clone(),
                parameters: serde_json::from_str(&spec.parameters)
                    .unwrap_or_else(|_| serde_json::json!({ "type": "object" })),
            })
            .collect();
        let mut request = CompletionRequest::from(self.history.clone())
            .preamble(preamble())
            .tools(tools)
            .max_tokens(8192);
        if self.cache_prompts {
            request = request
                .additional_params(serde_json::json!({ "cache_control": { "type": "ephemeral" } }));
        }
        let (tx, events) = crossbeam_channel::unbounded();
        let task = self
            .runtime
            .spawn(stream_reply(self.model.clone(), request, tx));
        self.turn = Turn::Thinking {
            events,
            task,
            round,
            streamed: false,
        };
    }
}

fn preamble() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    format!(
        "You are rigpi, a minimal coding agent in the user's terminal. Work in {cwd} with your \
         tools and keep answers short.\n\n\
         You are a Rust program built on Rig and Bevy from the source at {CRATE_DIR}. Your native \
         plugins live in {CRATE_DIR}/src/plugins/: each tool is a function in its own module, \
         listed in `native_tools()` in src/plugins/mod.rs. You can change or extend yourself by \
         editing those files and then calling the `reload` tool, which recompiles you and \
         hot-patches the running process without a restart. When hot-patching, keep the layout \
         of existing types and the signatures of existing Bevy systems unchanged; new functions, \
         new tools, and new behavior are fine.",
        cwd = cwd.display()
    )
}

async fn stream_reply(
    model: DynModel<Completion>,
    request: CompletionRequest,
    tx: crossbeam_channel::Sender<ModelEvent>,
) {
    let mut stream = match model.stream(request) {
        Ok(stream) => stream,
        Err(error) => {
            let _ = tx.send(ModelEvent::Failed(error.to_string()));
            return;
        }
    };
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => {
                let _ = tx.send(ModelEvent::Text(text));
            }
            Ok(_) => {}
            Err(error) => {
                let _ = tx.send(ModelEvent::Failed(error.to_string()));
                return;
            }
        }
    }
    let _ = tx.send(match stream.finish().await {
        Ok(response) => ModelEvent::Done(Box::new(response)),
        Err(error) => ModelEvent::Failed(error.to_string()),
    });
}

pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Transcript>()
            .add_message::<Submit>()
            .add_message::<Cancel>()
            .add_message::<Clear>()
            .add_systems(
                Update,
                (
                    clear,
                    cancel,
                    submit,
                    poll_model,
                    resolve_tools,
                    report_patches,
                )
                    .chain(),
            );
    }
}

fn clear(
    mut clears: MessageReader<Clear>,
    mut agent: ResMut<Agent>,
    mut transcript: ResMut<Transcript>,
) {
    if clears.read().count() > 0 && !agent.busy() {
        agent.history.clear();
        *transcript = Transcript::default();
    }
}

fn cancel(
    mut commands: Commands,
    mut cancels: MessageReader<Cancel>,
    mut agent: ResMut<Agent>,
    mut transcript: ResMut<Transcript>,
    outputs: Query<&ToolOutput>,
) {
    if cancels.read().count() == 0 || !agent.busy() {
        return;
    }
    match std::mem::replace(&mut agent.turn, Turn::Idle) {
        Turn::Thinking { task, .. } => task.abort(),
        Turn::Tools { calls, .. } => {
            // Every call still needs a result for the history to stay valid.
            let results = calls
                .into_iter()
                .map(|pending| {
                    let output = outputs
                        .get(pending.entity)
                        .cloned()
                        .unwrap_or_else(|_| ToolOutput::error("cancelled by the user"));
                    commands.entity(pending.entity).despawn();
                    pending
                        .call
                        .result(vec![ToolResultContent::text(output.content)])
                })
                .collect();
            agent.history.push(Message::tool_results(results));
        }
        Turn::Idle => {}
    }
    agent
        .history
        .push(Message::assistant("[cancelled by the user]"));
    agent.queue.clear();
    transcript.push(Entry::Info {
        text: "cancelled".into(),
    });
}

fn submit(
    mut submits: MessageReader<Submit>,
    mut agent: ResMut<Agent>,
    mut transcript: ResMut<Transcript>,
    specs: Query<&ToolSpec>,
) {
    for Submit(text) in submits.read() {
        agent.queue.push_back(text.clone());
    }
    if agent.busy() {
        return;
    }
    if let Some(text) = agent.queue.pop_front() {
        transcript.push(Entry::User { text: text.clone() });
        agent.history.push(Message::user(text));
        agent.start(&specs, 1);
    }
}

fn poll_model(
    mut commands: Commands,
    mut agent: ResMut<Agent>,
    mut transcript: ResMut<Transcript>,
    specs: Query<&ToolSpec>,
) {
    let Turn::Thinking {
        events,
        round,
        streamed,
        ..
    } = &mut agent.turn
    else {
        return;
    };
    let round = *round;
    let mut finished = None;
    for event in events.try_iter() {
        match event {
            ModelEvent::Text(text) => {
                *streamed = true;
                transcript.stream_text(&text);
            }
            ModelEvent::Done(response) => {
                finished = Some(Ok((response, *streamed)));
                break;
            }
            ModelEvent::Failed(error) => {
                finished = Some(Err(error));
                break;
            }
        }
    }
    let Some(finished) = finished else {
        return;
    };
    let (response, streamed) = match finished {
        Ok(done) => done,
        Err(error) => {
            transcript.push(Entry::Error { text: error });
            // Keep user/assistant alternation for the next prompt.
            agent
                .history
                .push(Message::assistant("[the model call failed]"));
            agent.turn = Turn::Idle;
            return;
        }
    };
    agent.usage += response.usage;
    if !streamed && !response.text().is_empty() {
        transcript.push(Entry::Assistant {
            text: response.text(),
        });
    }
    if let Some(message) = response.message() {
        agent.history.push(message);
    }
    let calls: Vec<ModelToolCall> = response.tool_calls().cloned().collect();
    if calls.is_empty() {
        transcript.streaming = false;
        agent.turn = Turn::Idle;
        return;
    }
    let pending = calls
        .into_iter()
        .map(|call| {
            let name = call.function.name.to_string();
            let arguments = call.function.arguments.to_string();
            let entry = transcript.push(Entry::Tool {
                name: name.clone(),
                arguments: arguments.clone(),
                output: None,
                is_error: false,
            });
            let known = specs.iter().any(|spec| spec.name == name);
            let mut entity = commands.spawn(ToolCall {
                name: name.clone(),
                arguments,
            });
            if !known {
                entity.insert(ToolOutput::error(format!("unknown tool `{name}`")));
            }
            Pending {
                entity: entity.id(),
                call,
                entry,
            }
        })
        .collect();
    agent.turn = Turn::Tools {
        calls: pending,
        round,
    };
}

fn resolve_tools(
    mut commands: Commands,
    mut agent: ResMut<Agent>,
    mut transcript: ResMut<Transcript>,
    calls: Query<Option<&ToolOutput>, With<ToolCall>>,
    specs: Query<&ToolSpec>,
) {
    let Turn::Tools {
        calls: pending,
        round,
    } = &agent.turn
    else {
        return;
    };
    // A call entity that vanished (despawned over BRP, say) counts as done.
    if pending
        .iter()
        .any(|p| matches!(calls.get(p.entity), Ok(None)))
    {
        return;
    }
    let round = *round;
    let Turn::Tools { calls: pending, .. } = std::mem::replace(&mut agent.turn, Turn::Idle) else {
        return;
    };
    let results = pending
        .into_iter()
        .map(|p| {
            let output = match calls.get(p.entity) {
                Ok(Some(output)) => output.clone(),
                _ => ToolOutput::error("the call was removed before it finished"),
            };
            if let Some(Entry::Tool {
                output: shown,
                is_error,
                ..
            }) = transcript.entries.get_mut(p.entry)
            {
                *shown = Some(output.content.clone());
                *is_error = output.is_error;
            }
            if let Ok(mut entity) = commands.get_entity(p.entity) {
                entity.despawn();
            }
            let content = match output.is_error {
                true => format!("Error: {}", output.content),
                false => output.content,
            };
            p.call.result(vec![ToolResultContent::text(content)])
        })
        .collect();
    agent.history.push(Message::tool_results(results));
    if round >= MAX_ROUNDS {
        transcript.push(Entry::Error {
            text: format!("stopped after {MAX_ROUNDS} model rounds"),
        });
        agent
            .history
            .push(Message::assistant("[stopped: too many rounds]"));
        return;
    }
    agent.start(&specs, round + 1);
}

/// Surface hot-patch outcomes in the transcript.
fn report_patches(hot: Res<HotReload>, mut seen: Local<u64>, mut transcript: ResMut<Transcript>) {
    if hot.generation == *seen {
        return;
    }
    *seen = hot.generation;
    let entry = match &hot.status {
        PatchStatus::Applied { count, elapsed } => Entry::Info {
            text: format!("hot-patched (#{count}, {:.2}s)", elapsed.as_secs_f32()),
        },
        PatchStatus::Failed(error) => Entry::Error {
            text: format!("hot patch failed:\n{error}"),
        },
        _ => return,
    };
    transcript.push(entry);
}
