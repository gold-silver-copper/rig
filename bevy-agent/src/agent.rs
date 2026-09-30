//! The conversation as ECS state: a history, one in-flight completion and the
//! tool calls of the current turn as entities. Tools are entities too; native
//! plugins and remote processes both register them and answer calls.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use rig_core::completion::{CompletionRequest, CompletionResponse, Message, ToolDefinition};
use rig_core::driver::DynModel;
use rig_core::error::ProviderError;
use rig_core::message::{ToolCall, ToolResultContent};
use rig_core::operation::Completion;

use crate::{model, plugins};

/// Tool-call rounds allowed per user prompt.
const MAX_TURNS: u32 = 24;

pub struct AgentPlugin {
    pub llm: DynModel<Completion>,
    pub model_name: String,
}

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Llm {
            model: self.llm.clone(),
            name: self.model_name.clone(),
        })
        .init_resource::<Conversation>()
        .init_resource::<Transcript>()
        .add_systems(
            Update,
            (
                start_completion,
                poll_completion,
                reject_unknown_tools,
                poll_tool_tasks,
                collect_tool_results,
            )
                .chain(),
        );
    }
}

/// The model every completion goes to.
#[derive(Resource)]
pub struct Llm {
    pub model: DynModel<Completion>,
    pub name: String,
}

/// One line of the screen transcript.
pub enum Entry {
    User(String),
    Assistant(String),
    Tool {
        name: String,
        args: String,
        result: Option<String>,
    },
    Note(String),
}

#[derive(Resource, Default)]
pub struct Transcript {
    pub entries: Vec<Entry>,
}

impl Transcript {
    pub fn note(&mut self, text: impl Into<String>) {
        self.entries.push(Entry::Note(text.into()));
    }
}

#[derive(Resource, Default)]
pub struct Conversation {
    pub history: Vec<Message>,
    pub wants_completion: bool,
    pub turns: u32,
    /// Tool calls of the current turn, in the order the model made them.
    pub calls: Vec<Entity>,
}

impl Conversation {
    /// Whether a completion or a tool call is outstanding.
    pub fn busy(&self, in_flight: bool) -> bool {
        in_flight || self.wants_completion || !self.calls.is_empty()
    }

    /// Start a new turn from `text`, as the user.
    pub fn prompt(&mut self, text: String, transcript: &mut Transcript) {
        self.history.push(Message::user(&text));
        transcript.entries.push(Entry::User(text));
        self.wants_completion = true;
        self.turns = 0;
    }
}

/// A completion being read from the model.
#[derive(Resource)]
pub struct InFlight {
    task: Task<Result<CompletionResponse, ProviderError>>,
    deltas: async_channel::Receiver<String>,
}

/// A tool the model may call. Native plugins spawn theirs at startup; remote
/// processes spawn theirs over BRP.
#[derive(Component)]
pub struct Tool {
    pub definition: ToolDefinition,
    pub remote: bool,
}

/// A call the model made, waiting for its answer.
#[derive(Component)]
pub struct ToolCallRequest(pub ToolCall);

/// The call's answer, as text for the model.
#[derive(Component)]
pub struct ToolDone(pub String);

/// A native tool still running off the main thread.
#[derive(Component)]
pub struct ToolTask(pub Task<String>);

/// A remote tool call handed to its watcher.
#[derive(Component)]
pub struct Dispatched;

fn start_completion(
    mut commands: Commands,
    mut convo: ResMut<Conversation>,
    mut transcript: ResMut<Transcript>,
    llm: Res<Llm>,
    tools: Query<&Tool>,
    in_flight: Option<Res<InFlight>>,
) {
    if !convo.wants_completion || in_flight.is_some() {
        return;
    }
    convo.wants_completion = false;
    if convo.turns >= MAX_TURNS {
        transcript.note(format!("stopped after {MAX_TURNS} tool rounds"));
        return;
    }
    convo.turns += 1;
    let request = CompletionRequest::from(convo.history.clone())
        .preamble(plugins::tools::preamble())
        .tools(tools.iter().map(|tool| tool.definition.clone()).collect())
        .max_tokens(4096);

    let (tx, deltas) = async_channel::unbounded();
    let task = AsyncComputeTaskPool::get().spawn(model::complete(llm.model.clone(), request, tx));
    commands.insert_resource(InFlight { task, deltas });
    transcript.entries.push(Entry::Assistant(String::new()));
}

fn poll_completion(
    mut commands: Commands,
    in_flight: Option<ResMut<InFlight>>,
    mut convo: ResMut<Conversation>,
    mut transcript: ResMut<Transcript>,
) {
    let Some(mut in_flight) = in_flight else {
        return;
    };
    while let Ok(delta) = in_flight.deltas.try_recv() {
        if let Some(Entry::Assistant(text)) = transcript.entries.last_mut() {
            text.push_str(&delta);
        }
    }
    let Some(result) = check_ready(&mut in_flight.task) else {
        return;
    };
    commands.remove_resource::<InFlight>();
    let response = match result {
        Ok(response) => response,
        Err(error) => {
            transcript.note(format!("model error: {error}"));
            return;
        }
    };
    // The stream delivered the text; a provider that only sends it at the end did not.
    let text = response.text();
    match transcript.entries.last_mut() {
        Some(Entry::Assistant(shown)) if shown.is_empty() && text.is_empty() => {
            transcript.entries.pop();
        }
        Some(Entry::Assistant(shown)) if shown.is_empty() => *shown = text,
        _ => {}
    }
    if let Some(message) = response.message() {
        convo.history.push(message);
    }
    for call in response.tool_calls() {
        let entity = commands.spawn(ToolCallRequest(call.clone())).id();
        convo.calls.push(entity);
        transcript.entries.push(Entry::Tool {
            name: call.function.name.to_string(),
            args: call.function.arguments.to_string(),
            result: None,
        });
    }
}

fn reject_unknown_tools(
    mut commands: Commands,
    calls: Query<
        (Entity, &ToolCallRequest),
        (Without<ToolDone>, Without<ToolTask>, Without<Dispatched>),
    >,
    tools: Query<&Tool>,
) {
    for (entity, call) in &calls {
        let name = call.0.function.name.as_str();
        if !tools.iter().any(|tool| tool.definition.name == name) {
            commands
                .entity(entity)
                .insert(ToolDone(format!("error: no tool named `{name}`")));
        }
    }
}

fn poll_tool_tasks(mut commands: Commands, mut tasks: Query<(Entity, &mut ToolTask)>) {
    for (entity, mut task) in &mut tasks {
        if let Some(output) = check_ready(&mut task.0) {
            commands
                .entity(entity)
                .remove::<ToolTask>()
                .insert(ToolDone(output));
        }
    }
}

fn collect_tool_results(
    mut commands: Commands,
    mut convo: ResMut<Conversation>,
    mut transcript: ResMut<Transcript>,
    done: Query<(&ToolCallRequest, &ToolDone)>,
) {
    if convo.calls.is_empty() {
        return;
    }
    let mut results = Vec::with_capacity(convo.calls.len());
    for &entity in &convo.calls {
        let Ok((call, done)) = done.get(entity) else {
            return;
        };
        results.push((
            call.0.result(vec![ToolResultContent::text(&done.0)]),
            done.0.clone(),
        ));
    }
    for entity in convo.calls.drain(..) {
        commands.entity(entity).despawn();
    }
    let mut shown = results.iter().map(|(_, text)| text);
    for entry in &mut transcript.entries {
        if let Entry::Tool {
            result: result @ None,
            ..
        } = entry
            && let Some(text) = shown.next()
        {
            *result = Some(text.clone());
        }
    }
    convo.history.push(Message::tool_results(
        results.into_iter().map(|(result, _)| result).collect(),
    ));
    convo.wants_completion = true;
}
