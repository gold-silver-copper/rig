//! The glue between Bevy and Rig: the only module that uses both.
//!
//! An agent is an entity with an [`Agent`] marker and components holding its
//! [`Conversation`], its [`Model`], its [`Turn`] state and its
//! [`Instructions`]. Set a turn to [`Turn::Request`] and the glue sends the
//! conversation to the model on a tokio runtime, streams the reply back as
//! [`AgentEvent`] messages, appends it, and spawns one entity per tool call
//! with a [`ToolCall`] component and a [`CallOf`] relationship to the agent.
//! A call ends when some system inserts its [`ToolOutput`]: the glue runs
//! calls to [`DynamicTool`]s itself, and plugins answer the rest from their
//! own systems. Results go back to the model until it answers without tools.
//!
//! Nothing here assumes a single agent: every agent in the world runs its
//! own turn at the same time.

use std::collections::BTreeMap;
use std::error::Error;

use async_channel::{Receiver, TryRecvError};
use bevy::prelude::*;
use futures::StreamExt;
use rig_core::DynModel;
use rig_core::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use rig_core::message::{self, Message, ToolResultContent};
use rig_core::operation::Completion;
use rig_core::providers::anthropic::ThinkingPrefixMismatch;
use rig_core::providers::registry::{ProviderConfig, ProviderRef};
use rig_core::streaming::{Item, StreamEvent};
use rig_core::tool::DynamicTool;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::Instrument;

pub struct GluePlugin;

impl Plugin for GluePlugin {
    fn build(&self, app: &mut App) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap_or_else(|error| panic!("cannot start the tokio runtime: {error}"));
        app.insert_resource(RigRuntime(runtime))
            .init_resource::<Tools>()
            .init_resource::<ModelFactory>()
            .add_message::<AgentEvent>()
            .add_systems(
                Update,
                (
                    send_requests,
                    receive_replies,
                    dispatch_calls,
                    run_rig_tools,
                    finish_rig_tools,
                    collect_results,
                )
                    .chain()
                    .in_set(GlueSystems),
            );
    }
}

/// The glue's systems, for ordering others around them.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct GlueSystems;

/// Marks an agent entity and brings its components.
#[derive(Component, Default)]
#[require(Conversation, Turn, Instructions)]
pub struct Agent;

/// The messages sent to the model, oldest first.
#[derive(Component, Default, Clone, Serialize, Deserialize)]
pub struct Conversation(pub Vec<Message>);

/// The system prompt sent ahead of the conversation.
#[derive(Component, Default, Clone)]
pub struct Instructions(pub String);

/// Where an agent is in its turn. Serializable, so a turn can be saved and
/// resumed: a saved `Request` is sent again.
#[derive(Component, Default, Clone, PartialEq, Serialize, Deserialize)]
pub enum Turn {
    #[default]
    Idle,
    /// Waiting on the model.
    Request,
    /// Running the tool calls of the last reply, one at a time, in order.
    Tools {
        calls: Vec<message::ToolCall>,
        results: Vec<message::ToolResult>,
    },
}

/// The model an agent talks to, as a `vendor[/format]:model` reference.
#[derive(Component)]
pub struct Model {
    spec: String,
    model: Result<DynModel<Completion>, String>,
    params: Option<Value>,
}

impl Model {
    pub fn spec(&self) -> &str {
        &self.spec
    }

    /// Why the model cannot be used, if it cannot.
    pub fn error(&self) -> Option<&str> {
        self.model.as_ref().err().map(String::as_str)
    }
}

/// Builds [`Model`]s. The default reads credentials from the environment;
/// a plugin may replace it, for example to send through a recording proxy.
#[derive(Resource)]
pub struct ModelFactory(
    pub Box<dyn Fn(&ProviderRef) -> Result<DynModel<Completion>, String> + Send + Sync>,
);

impl Default for ModelFactory {
    fn default() -> Self {
        Self(Box::new(|reference| {
            let variable = reference
                .id()
                .map(|id| id.api_key_env())
                .unwrap_or_default();
            let key = std::env::var(variable).unwrap_or_default();
            if key.is_empty() && reference.id().is_none_or(|id| id.requires_credential()) {
                return Err(format!("{variable} is not set"));
            }
            Ok(provider_config(reference, key)
                .completion_model(reference.model(), rig_reqwest::shared()))
        }))
    }
}

/// A reference's configuration as the agent uses it. Anthropic drops
/// replayed thinking blocks the API would otherwise reject once the tool
/// list changes, as it does when a reload or a plugin adds a tool.
pub fn provider_config(reference: &ProviderRef, api_key: String) -> ProviderConfig {
    match reference.config(api_key) {
        ProviderConfig::Anthropic(config) => ProviderConfig::Anthropic(
            config.with_thinking_prefix_mismatch(ThinkingPrefixMismatch::DropBlock),
        ),
        config => config,
    }
}

impl ModelFactory {
    pub fn model(&self, spec: &str) -> Model {
        let reference = ProviderRef::parse(spec).map_err(|error| report(&error));
        // OpenAI keeps Responses for later retrieval unless told not to.
        let params = reference
            .as_ref()
            .ok()
            .filter(|reference| reference.id().is_some_and(|id| id.vendor() == "openai"))
            .map(|_| json!({"store": false}));
        Model {
            spec: spec.to_owned(),
            model: reference.and_then(|reference| (self.0)(&reference)),
            params,
        }
    }
}

/// The runtime Rig's futures run on.
#[derive(Resource)]
pub struct RigRuntime(pub tokio::runtime::Runtime);

/// Every tool the model can call, by name, in a stable order.
#[derive(Resource, Default)]
pub struct Tools(pub BTreeMap<String, Tool>);

pub struct Tool {
    pub definition: ToolDefinition,
    /// Run by the glue; `None` when a plugin's system answers the calls.
    pub run: Option<DynamicTool>,
}

impl Tools {
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.0
            .values()
            .map(|tool| tool.definition.clone())
            .collect()
    }
}

pub trait AddTool {
    /// Registers a tool the glue runs.
    fn add_rig_tool(&mut self, tool: DynamicTool) -> &mut Self;
    /// Registers a tool whose calls a plugin's system answers by inserting
    /// [`ToolOutput`] on call entities with this name.
    fn add_system_tool(&mut self, definition: ToolDefinition) -> &mut Self;
}

impl AddTool for App {
    fn add_rig_tool(&mut self, tool: DynamicTool) -> &mut Self {
        let definition = tool.definition();
        self.world_mut().get_resource_or_init::<Tools>().0.insert(
            definition.name.clone(),
            Tool {
                definition,
                run: Some(tool),
            },
        );
        self
    }

    fn add_system_tool(&mut self, definition: ToolDefinition) -> &mut Self {
        self.world_mut().get_resource_or_init::<Tools>().0.insert(
            definition.name.clone(),
            Tool {
                definition,
                run: None,
            },
        );
        self
    }
}

/// A tool call, one entity each.
#[derive(Component, Clone)]
pub struct ToolCall(pub message::ToolCall);

/// The agent a tool call belongs to.
#[derive(Component)]
#[relationship(relationship_target = ToolCalls)]
pub struct CallOf(pub Entity);

/// An agent's live tool calls; they are despawned with it.
#[derive(Component)]
#[relationship_target(relationship = CallOf, linked_spawn)]
pub struct ToolCalls(Vec<Entity>);

/// A call made by another tool rather than by the model. The caller reads
/// its output and despawns it; it never enters the conversation.
#[derive(Component)]
pub struct Nested;

/// The text a finished call returned.
#[derive(Component, Clone)]
pub struct ToolOutput(pub String);

/// What happened in a turn, for the TUI, telemetry and other observers.
#[derive(Message, Clone)]
pub enum AgentEvent {
    /// A fragment of the reply text, as it streams.
    Text { agent: Entity, text: String },
    /// The model replied; its message is now in the conversation.
    Replied {
        agent: Entity,
        response: Box<CompletionResponse>,
    },
    /// A tool call started.
    CallStarted {
        agent: Entity,
        call: message::ToolCall,
    },
    /// A tool call finished.
    CallFinished {
        agent: Entity,
        call: message::ToolCall,
        output: String,
    },
    /// The turn ended: the model answered without tools, or it failed.
    Ended {
        agent: Entity,
        error: Option<String>,
    },
}

/// A request in flight, on the agent it belongs to.
#[derive(Component)]
pub struct InFlight {
    task: tokio::task::JoinHandle<()>,
    events: Receiver<Reply>,
}

enum Reply {
    Text(String),
    Done(Box<Result<CompletionResponse, String>>),
}

/// The span covering a turn, on its agent.
#[derive(Component)]
pub struct TurnSpan(pub tracing::Span);

/// A glue-run tool call waiting on its future.
#[derive(Component)]
pub struct Running(Receiver<String>);

/// The span covering a tool call, on its entity.
#[derive(Component)]
pub struct CallSpan(pub tracing::Span);

/// An error and its sources on one line.
pub fn report(error: &dyn Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let cause_text = cause.to_string();
        if !text.contains(&cause_text) {
            text = format!("{text}: {cause_text}");
        }
        source = cause.source();
    }
    text
}

fn send_requests(
    mut commands: Commands,
    agents: Query<
        (
            Entity,
            &Turn,
            &Conversation,
            &Instructions,
            &Model,
            Option<&TurnSpan>,
        ),
        (With<Agent>, Without<InFlight>),
    >,
    tools: Res<Tools>,
    runtime: Res<RigRuntime>,
    mut events: MessageWriter<AgentEvent>,
) {
    for (agent, turn, conversation, instructions, model, span) in &agents {
        if *turn != Turn::Request {
            continue;
        }
        let span = match span {
            Some(TurnSpan(span)) => span.clone(),
            None => {
                let span = tracing::info_span!(
                    "invoke_agent",
                    gen_ai.operation.name = "invoke_agent",
                    gen_ai.agent.name = "bevy-agent",
                    gen_ai.request.model = model.spec(),
                );
                commands.entity(agent).insert(TurnSpan(span.clone()));
                span
            }
        };
        let handle = match &model.model {
            Ok(handle) => handle.clone(),
            Err(error) => {
                commands
                    .entity(agent)
                    .insert(Turn::Idle)
                    .remove::<TurnSpan>();
                events.write(AgentEvent::Ended {
                    agent,
                    error: Some(format!("cannot use {}: {error}", model.spec())),
                });
                continue;
            }
        };
        let mut request = CompletionRequest::new(Message::system(instructions.0.clone()));
        request.chat_history.extend(conversation.0.iter().cloned());
        request.tools = tools.definitions();
        request.additional_params = model.params.clone();
        let (sender, receiver) = async_channel::unbounded();
        let task = runtime.0.spawn(
            async move {
                let result = async {
                    let mut stream = handle.stream(request)?;
                    while let Some(item) = stream.next().await {
                        if let Item::Event(StreamEvent::Text { text, .. }) = item? {
                            let _ = sender.send(Reply::Text(text)).await;
                        }
                    }
                    stream.finish().await
                }
                .await;
                let result = result.map_err(|error| report(&error));
                let _ = sender.send(Reply::Done(Box::new(result))).await;
            }
            .instrument(span),
        );
        commands.entity(agent).insert(InFlight {
            task,
            events: receiver,
        });
    }
}

fn receive_replies(
    mut commands: Commands,
    mut agents: Query<(Entity, &InFlight, &mut Turn, &mut Conversation)>,
    mut events: MessageWriter<AgentEvent>,
) {
    for (agent, flight, mut turn, mut conversation) in &mut agents {
        while let Ok(reply) = flight.events.try_recv() {
            match reply {
                Reply::Text(text) => {
                    events.write(AgentEvent::Text { agent, text });
                }
                Reply::Done(done) => {
                    commands.entity(agent).remove::<InFlight>();
                    match *done {
                        Ok(response) => {
                            let calls: Vec<message::ToolCall> =
                                response.tool_calls().cloned().collect();
                            if let Some(message) = response.message() {
                                conversation.0.push(message);
                            }
                            events.write(AgentEvent::Replied {
                                agent,
                                response: Box::new(response),
                            });
                            if calls.is_empty() {
                                *turn = Turn::Idle;
                                commands.entity(agent).remove::<TurnSpan>();
                                events.write(AgentEvent::Ended { agent, error: None });
                            } else {
                                *turn = Turn::Tools {
                                    calls,
                                    results: Vec::new(),
                                };
                            }
                        }
                        Err(error) => {
                            *turn = Turn::Idle;
                            commands.entity(agent).remove::<TurnSpan>();
                            events.write(AgentEvent::Ended {
                                agent,
                                error: Some(error),
                            });
                        }
                    }
                    break;
                }
            }
        }
    }
}

/// Spawns the next call of each agent in its tool phase, once the previous
/// one is collected.
fn dispatch_calls(
    mut commands: Commands,
    agents: Query<(Entity, &Turn, Option<&ToolCalls>, Option<&TurnSpan>), With<Agent>>,
    top_level: Query<(), (With<ToolCall>, Without<Nested>)>,
    tools: Res<Tools>,
    mut events: MessageWriter<AgentEvent>,
) {
    for (agent, turn, live, span) in &agents {
        let Turn::Tools { calls, results } = turn else {
            continue;
        };
        if live.is_some_and(|live| live.iter().any(|call| top_level.contains(call))) {
            continue;
        }
        let Some(call) = calls.get(results.len()) else {
            continue;
        };
        events.write(AgentEvent::CallStarted {
            agent,
            call: call.clone(),
        });
        let name = call.function.name.as_str();
        let span = tracing::info_span!(
            parent: span.and_then(|span| span.0.id()),
            "execute_tool",
            gen_ai.operation.name = "execute_tool",
            gen_ai.tool.name = name,
            gen_ai.tool.call.id = %call.id,
        );
        let mut entity = commands.spawn((ToolCall(call.clone()), CallOf(agent), CallSpan(span)));
        if !tools.0.contains_key(name) {
            entity.insert(ToolOutput(format!(
                "error: there is no tool named `{name}`"
            )));
        }
    }
}

/// Starts calls to tools the glue runs itself, nested or not.
fn run_rig_tools(
    mut commands: Commands,
    calls: Query<(Entity, &ToolCall, &CallSpan), Added<ToolCall>>,
    tools: Res<Tools>,
    runtime: Res<RigRuntime>,
) {
    for (entity, ToolCall(call), CallSpan(span)) in &calls {
        let Some(tool) = tools
            .0
            .get(call.function.name.as_str())
            .and_then(|tool| tool.run.clone())
        else {
            continue;
        };
        let arguments = call.function.arguments.clone();
        let (sender, receiver) = async_channel::bounded(1);
        runtime.0.spawn(
            async move {
                let output = match tool.execute(arguments).await {
                    Ok(output) => output.render(),
                    Err(error) => format!("error: {error}"),
                };
                let _ = sender.send(output).await;
            }
            .instrument(span.clone()),
        );
        commands.entity(entity).insert(Running(receiver));
    }
}

fn finish_rig_tools(mut commands: Commands, calls: Query<(Entity, &Running)>) {
    for (entity, Running(receiver)) in &calls {
        let output = match receiver.try_recv() {
            Ok(output) => output,
            Err(TryRecvError::Empty) => continue,
            Err(TryRecvError::Closed) => "error: the tool stopped without a result".into(),
        };
        commands
            .entity(entity)
            .remove::<Running>()
            .insert(ToolOutput(output));
    }
}

/// Moves finished top-level calls into their agent's turn and hands the
/// results to the model once every call of the reply is answered.
fn collect_results(
    mut commands: Commands,
    calls: Query<(Entity, &ToolCall, &CallOf, &ToolOutput), Without<Nested>>,
    mut agents: Query<(&mut Turn, &mut Conversation)>,
    mut events: MessageWriter<AgentEvent>,
) {
    for (entity, ToolCall(call), CallOf(agent), ToolOutput(output)) in &calls {
        commands.entity(entity).despawn();
        let Ok((mut turn, mut conversation)) = agents.get_mut(*agent) else {
            continue;
        };
        let Turn::Tools { calls, results } = &mut *turn else {
            continue;
        };
        results.push(call.result(vec![ToolResultContent::text(output.clone())]));
        events.write(AgentEvent::CallFinished {
            agent: *agent,
            call: call.clone(),
            output: output.clone(),
        });
        if results.len() == calls.len() {
            conversation
                .0
                .push(Message::tool_results(std::mem::take(results)));
            *turn = Turn::Request;
        }
    }
}

/// Stops an agent's turn: drops its request and live calls, and answers
/// the remaining calls as cancelled so the conversation stays well formed.
pub fn cancel(world: &mut World, agent: Entity) {
    let Some(mut turn) = world.get_mut::<Turn>(agent) else {
        return;
    };
    let turn = std::mem::take(&mut *turn);
    if turn == Turn::Idle {
        return;
    }
    let mut entity = world.entity_mut(agent);
    if let Some(flight) = entity.take::<InFlight>() {
        flight.task.abort();
    }
    entity.remove::<TurnSpan>();
    let live: Vec<Entity> = entity
        .get::<ToolCalls>()
        .map(|calls| calls.iter().collect())
        .unwrap_or_default();
    for call in live {
        world.despawn(call);
    }
    if let Turn::Tools { calls, mut results } = turn {
        for call in calls.iter().skip(results.len()) {
            results.push(call.result(vec![ToolResultContent::text("cancelled by the user")]));
        }
        if let Some(mut conversation) = world.get_mut::<Conversation>(agent) {
            conversation.0.push(Message::tool_results(results));
        }
    }
    world.write_message(AgentEvent::Ended {
        agent,
        error: Some("cancelled".into()),
    });
}
