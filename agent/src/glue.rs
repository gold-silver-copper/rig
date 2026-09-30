//! The glue between Bevy and Rig: agents are entities, and so are their tool
//! calls. Everything here is plain Bevy (components, systems, messages) over
//! plain Rig values (`Message`, `ProviderRef`, `DynamicTool`), and nothing
//! here knows about the TUI, reload, BRP or any plugin.
//!
//! An agent is an entity with [`Agent`], [`Model`], [`Conversation`],
//! [`Inbox`] and [`Turn`]. Push text into its [`Inbox`]; the systems send the
//! conversation to the model, spawn one [`Call`] entity per tool call (related
//! to the agent by [`CallOf`]), run them one at a time in order, answer the
//! model, and repeat until it replies without calling a tool. Progress is
//! reported as [`AgentEvent`] messages.
//!
//! Tools live in the [`Tools`] resource. A [`Tool::Task`] is a Rig
//! `DynamicTool` run here on the async compute pool. A [`Tool::World`] is
//! declared here but run by another system: it waits for its call entity to
//! become [`Ready`], does its work, and inserts [`Done`].
//!
//! With a [`SessionMemory`] resource, an agent with a [`Session`] appends each
//! new message to that Rig `ConversationMemory` and sends the model what the
//! memory loads, so a memory that compacts shapes every request.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, IoTaskPool, Task};
use rig_core::completion::{CompletionRequest, CompletionResponse, Message, ToolDefinition};
use rig_core::id::ConversationId;
use rig_core::memory::ConversationMemory;
use rig_core::message::{AssistantContent, ToolCall, UserContent};
use rig_core::providers::registry::ProviderRef;
use rig_core::tool::{DynamicTool, ToolOutput};
use rig_core::transcript::{tool_result_message, tool_result_output};
use rig_core::ProviderError;
use tracing::Instrument;

/// Adds the [`Tools`] registry, the [`AgentEvent`] and [`Interrupt`]
/// messages, and the systems that drive every agent.
pub struct RigPlugin;

impl Plugin for RigPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Tools>()
            .add_message::<AgentEvent>()
            .add_message::<Interrupt>()
            .configure_sets(Update, (AgentSet::Input, AgentSet::Route, AgentSet::Run).chain())
            .add_systems(
                Update,
                (
                    interrupt,
                    start_turns,
                    send_requests,
                    receive_replies,
                    schedule_calls,
                    run_task_calls,
                    poll_task_calls,
                    report_calls,
                    finish_calls,
                )
                    .chain()
                    .in_set(AgentSet::Run),
            );
    }
}

/// Each frame: input reaches agents, then their requests are routed, then
/// the systems here run them.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AgentSet {
    /// Systems that prompt agents or change their models.
    Input,
    /// Systems that decide where requests go ([`Endpoints`]).
    Route,
    /// The agent loop itself.
    Run,
}

/// An agent. `preamble` is its system prompt.
#[derive(Component, Clone, Debug)]
#[require(Conversation, Inbox, Turn)]
pub struct Agent {
    pub preamble: String,
}

/// The model the agent's next request goes to. Change it at any time.
#[derive(Component, Clone, Debug)]
pub struct Model(pub ProviderRef);

/// Everything said so far, in Rig's provider-neutral form.
#[derive(Component, Clone, Debug, Default)]
pub struct Conversation(pub Vec<Message>);

/// Input waiting for the agent. A follow-up starts a turn once the agent is
/// idle; steering joins the running turn at its next model request.
#[derive(Component, Clone, Debug, Default)]
pub struct Inbox {
    pub follow_ups: VecDeque<String>,
    pub steering: Vec<String>,
}

/// Where the agent's turn is.
#[derive(Component, Debug, Default)]
pub enum Turn {
    #[default]
    Idle,
    /// A model request is due.
    Ready,
    /// Waiting on the model.
    Thinking(Task<Result<CompletionResponse, ProviderError>>),
    /// Running the tool calls of the last reply.
    Acting,
}

impl Turn {
    pub fn is_idle(&self) -> bool {
        matches!(self, Turn::Idle)
    }
}

/// The agent's conversation in [`SessionMemory`].
#[derive(Component, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Session(pub ConversationId);

/// The Rig memory that [`Session`] agents persist to and load requests from.
#[derive(Resource, Clone)]
pub struct SessionMemory(pub Arc<dyn ConversationMemory>);

/// Where to send an agent's requests for a vendor, instead of the vendor's
/// own API with credentials from the environment.
#[derive(Component, Clone, Debug, Default)]
pub struct Endpoints(pub HashMap<String, Endpoint>);

#[derive(Clone, Debug)]
pub struct Endpoint {
    pub base_url: String,
    pub api_key: String,
}

/// The span of the agent's current turn; model and tool spans nest in it.
#[derive(Component, Clone, Debug)]
pub struct TurnSpan(pub tracing::Span);

/// A tool call, as an entity related to its agent.
#[derive(Component, Clone, Debug)]
pub struct Call {
    pub call: ToolCall,
    /// Position in the reply; calls run in this order.
    pub index: usize,
}

#[derive(Component, Debug)]
#[relationship(relationship_target = Calls)]
pub struct CallOf(pub Entity);

#[derive(Component, Debug)]
#[relationship_target(relationship = CallOf, linked_spawn)]
pub struct Calls(Vec<Entity>);

/// The call is next: its handler may run it.
#[derive(Component, Debug)]
pub struct Ready;

/// A [`Tool::Task`] call in flight.
#[derive(Component, Debug)]
pub struct Running(Task<Result<ToolOutput, String>>);

/// The call's outcome: the tool's output, or error text for the model.
#[derive(Component, Clone, Debug)]
pub struct Done(pub Result<ToolOutput, String>);

/// The call's outcome was reported as an [`AgentEvent`].
#[derive(Component, Debug)]
pub struct Reported;

/// Every tool the model is offered, by name.
#[derive(Resource, Default, Clone)]
pub struct Tools(pub BTreeMap<String, Tool>);

#[derive(Clone)]
pub enum Tool {
    /// Run here, on the async compute pool.
    Task(DynamicTool),
    /// Run by some other system, which inserts [`Done`] on [`Ready`] calls.
    World(ToolDefinition),
}

impl Tool {
    pub fn definition(&self) -> ToolDefinition {
        match self {
            Tool::Task(tool) => tool.definition(),
            Tool::World(definition) => definition.clone(),
        }
    }
}

impl Tools {
    pub fn add(&mut self, tool: Tool) {
        self.0.insert(tool.definition().name, tool);
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.0.values().map(Tool::definition).collect()
    }
}

/// What an agent is doing, for anything that shows or records it.
#[derive(Message, Clone, Debug)]
pub struct AgentEvent {
    pub agent: Entity,
    pub kind: EventKind,
}

#[derive(Clone, Debug)]
pub enum EventKind {
    /// A user message entered the conversation.
    Input(String),
    /// The model said this.
    Text(String),
    /// A tool call started.
    CallStarted { name: String, arguments: serde_json::Value },
    /// A tool call finished with this text for the model.
    CallFinished { name: String, output: String, error: bool },
    /// The turn ended: the model replied without calling a tool, failed, or
    /// was interrupted.
    TurnEnded,
    Error(String),
}

/// Stop an agent's turn. Unfinished calls are answered "interrupted".
#[derive(Message, Clone, Copy, Debug)]
pub struct Interrupt(pub Entity);

/// Append `message` to the agent's conversation and its session memory.
pub fn push_message(
    conversation: &mut Conversation,
    session: Option<&Session>,
    memory: Option<&SessionMemory>,
    message: Message,
) {
    if let (Some(session), Some(memory)) = (session, memory)
        && let Err(error) =
            futures::executor::block_on(memory.0.append(&session.0, vec![message.clone()]))
    {
        error!("appending to session {}: {error}", session.0);
    }
    conversation.0.push(message);
}

/// The request for the agent's next turn. With a session memory, the history
/// is what the memory loads (possibly compacted); otherwise the conversation.
pub fn request_for(
    agent: &Agent,
    model: &ProviderRef,
    conversation: &Conversation,
    session: Option<&Session>,
    memory: Option<&SessionMemory>,
    tools: &Tools,
) -> CompletionRequest {
    let history = match (session, memory) {
        (Some(session), Some(memory)) => futures::executor::block_on(memory.0.load(&session.0))
            .unwrap_or_else(|error| {
                error!("loading session {}: {error}", session.0);
                conversation.0.clone()
            }),
        _ => conversation.0.clone(),
    };
    let mut request = CompletionRequest::from(history)
        .preamble(agent.preamble.clone())
        .tools(tools.definitions());
    // The whole conversation is sent every time, so the provider need not
    // keep responses; recordings also refuse stored state.
    if model.id().is_some_and(|id| id.vendor() == "openai") {
        request = request.additional_params(serde_json::json!({"store": false}));
    }
    request
}

/// The model `reference` names, sent to `endpoints` if one matches its
/// vendor, else to the vendor with credentials from the environment.
pub fn resolve_model(
    reference: &ProviderRef,
    endpoints: Option<&Endpoints>,
) -> Result<rig_core::DynModel<rig_core::operation::Completion>, String> {
    let vendor = reference.id().map(|id| id.vendor().to_owned()).unwrap_or_default();
    match endpoints.and_then(|endpoints| endpoints.0.get(&vendor)) {
        Some(endpoint) => {
            let config = reference
                .config(endpoint.api_key.clone())
                .with_base_url(endpoint.base_url.clone());
            let routed = ProviderRef::configured(config, reference.model())
                .map_err(|error| error.to_string())?;
            Ok(routed.completion_model_with(endpoint.api_key.clone(), rig_reqwest::shared()))
        }
        None => reference
            .completion_model()
            .map_err(|error| format!("{reference}: {error}")),
    }
}

fn interrupt(
    mut interrupts: MessageReader<Interrupt>,
    mut agents: Query<(&mut Turn, Option<&Calls>)>,
    calls: Query<(), Without<Done>>,
    mut events: MessageWriter<AgentEvent>,
    mut commands: Commands,
) {
    for Interrupt(agent) in interrupts.read() {
        let Ok((mut turn, owned)) = agents.get_mut(*agent) else {
            continue;
        };
        if matches!(*turn, Turn::Ready | Turn::Thinking(_)) {
            *turn = Turn::Idle;
            events.write(AgentEvent {
                agent: *agent,
                kind: EventKind::Error("interrupted by the user".into()),
            });
            events.write(AgentEvent {
                agent: *agent,
                kind: EventKind::TurnEnded,
            });
            commands.entity(*agent).remove::<TurnSpan>();
        }
        for call in owned.into_iter().flat_map(|calls| calls.iter()) {
            if calls.contains(call) {
                commands
                    .entity(call)
                    .remove::<Running>()
                    .insert(Done(Err("interrupted by the user".into())));
            }
        }
    }
}

fn start_turns(
    mut agents: Query<(
        Entity,
        &Model,
        &mut Turn,
        &mut Inbox,
        &mut Conversation,
        Option<&Session>,
        Option<&Name>,
    )>,
    memory: Option<Res<SessionMemory>>,
    mut events: MessageWriter<AgentEvent>,
    mut commands: Commands,
) {
    for (agent, model, mut turn, mut inbox, mut conversation, session, name) in &mut agents {
        if !turn.is_idle() || (inbox.follow_ups.is_empty() && inbox.steering.is_empty()) {
            continue;
        }
        let text = match inbox.follow_ups.pop_front() {
            Some(text) => text,
            None => std::mem::take(&mut inbox.steering).join("\n\n"),
        };
        push_message(&mut conversation, session, memory.as_deref(), Message::user(text.clone()));
        events.write(AgentEvent {
            agent,
            kind: EventKind::Input(text),
        });
        let span = tracing::info_span!(
            target: "rig_pi::agent",
            "invoke_agent",
            gen_ai.operation.name = "invoke_agent",
            gen_ai.agent.name = name.map(|n| n.as_str()).unwrap_or("agent"),
            gen_ai.provider.name = model.0.id().map(|id| id.vendor()).unwrap_or_default(),
            gen_ai.request.model = model.0.model(),
        );
        commands.entity(agent).insert(TurnSpan(span));
        *turn = Turn::Ready;
    }
}

fn send_requests(
    mut agents: Query<(
        Entity,
        &Agent,
        &Model,
        &Conversation,
        &mut Turn,
        Option<&Session>,
        Option<&Endpoints>,
        Option<&TurnSpan>,
    )>,
    tools: Res<Tools>,
    memory: Option<Res<SessionMemory>>,
    mut events: MessageWriter<AgentEvent>,
) {
    for (entity, agent, model, conversation, mut turn, session, endpoints, span) in &mut agents {
        if !matches!(*turn, Turn::Ready) {
            continue;
        }
        let request = request_for(agent, &model.0, conversation, session, memory.as_deref(), &tools);
        match resolve_model(&model.0, endpoints) {
            Ok(model) => {
                let span = span.map(|span| span.0.clone()).unwrap_or_else(tracing::Span::none);
                let call = model.call(request).instrument(span);
                *turn = Turn::Thinking(IoTaskPool::get().spawn(call));
            }
            Err(error) => {
                *turn = Turn::Idle;
                events.write(AgentEvent {
                    agent: entity,
                    kind: EventKind::Error(error),
                });
                events.write(AgentEvent {
                    agent: entity,
                    kind: EventKind::TurnEnded,
                });
            }
        }
    }
}

fn receive_replies(
    mut agents: Query<(Entity, &Model, &mut Turn, &mut Conversation, Option<&Session>)>,
    memory: Option<Res<SessionMemory>>,
    mut events: MessageWriter<AgentEvent>,
    mut commands: Commands,
) {
    for (agent, model, mut turn, mut conversation, session) in &mut agents {
        let Turn::Thinking(task) = &mut *turn else {
            continue;
        };
        let Some(result) = check_ready(task) else {
            continue;
        };
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                *turn = Turn::Idle;
                events.write(AgentEvent {
                    agent,
                    kind: EventKind::Error(format!("{}: {error}", model.0)),
                });
                events.write(AgentEvent {
                    agent,
                    kind: EventKind::TurnEnded,
                });
                commands.entity(agent).remove::<TurnSpan>();
                continue;
            }
        };
        for part in &response.choice {
            if let AssistantContent::Text(text) = part
                && !text.text.trim().is_empty()
            {
                events.write(AgentEvent {
                    agent,
                    kind: EventKind::Text(text.text.trim().to_owned()),
                });
            }
        }
        let calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
        if let Some(message) = response.message() {
            push_message(&mut conversation, session, memory.as_deref(), message);
        }
        if calls.is_empty() {
            *turn = Turn::Idle;
            events.write(AgentEvent {
                agent,
                kind: EventKind::TurnEnded,
            });
            commands.entity(agent).remove::<TurnSpan>();
            continue;
        }
        for (index, call) in calls.into_iter().enumerate() {
            commands.spawn((Call { call, index }, CallOf(agent)));
        }
        *turn = Turn::Acting;
    }
}

/// Mark the first unfinished call of each acting agent [`Ready`].
fn schedule_calls(
    agents: Query<(Entity, &Turn, &Calls)>,
    calls: Query<(&Call, Has<Ready>, Has<Done>)>,
    mut events: MessageWriter<AgentEvent>,
    mut commands: Commands,
) {
    for (agent, turn, owned) in &agents {
        if !matches!(turn, Turn::Acting) {
            continue;
        }
        let next = owned
            .iter()
            .filter_map(|entity| calls.get(entity).ok().map(|call| (entity, call)))
            .filter(|(_, (_, _, done))| !done)
            .min_by_key(|(_, (call, _, _))| call.index);
        if let Some((entity, (call, false, _))) = next {
            events.write(AgentEvent {
                agent,
                kind: EventKind::CallStarted {
                    name: call.call.function.name.to_string(),
                    arguments: call.call.function.arguments.clone(),
                },
            });
            commands.entity(entity).insert(Ready);
        }
    }
}

fn run_task_calls(
    calls: Query<(Entity, &Call, &CallOf), (With<Ready>, Without<Running>, Without<Done>)>,
    spans: Query<&TurnSpan>,
    tools: Res<Tools>,
    mut commands: Commands,
) {
    for (entity, call, owner) in &calls {
        let name = call.call.function.name.as_str();
        match tools.0.get(name) {
            Some(Tool::Task(tool)) => {
                let tool = tool.clone();
                let arguments = call.call.function.arguments.clone();
                let parent = spans.get(owner.0).map(|s| s.0.clone()).unwrap_or_else(|_| tracing::Span::none());
                let span = tracing::info_span!(
                    target: "rig_pi::agent",
                    parent: &parent,
                    "execute_tool",
                    gen_ai.operation.name = "execute_tool",
                    gen_ai.tool.name = name,
                    gen_ai.tool.call.id = %call.call.id,
                );
                let task = AsyncComputeTaskPool::get().spawn(
                    async move { tool.execute(arguments).await.map_err(|e| error_text(&e)) }
                        .instrument(span),
                );
                commands.entity(entity).insert(Running(task));
            }
            Some(Tool::World(_)) => {}
            None => {
                commands
                    .entity(entity)
                    .insert(Done(Err(format!("no tool named `{name}`"))));
            }
        }
    }
}

/// The text a model should see for a failed tool call.
pub fn error_text(error: &rig_core::tool::ToolExecutionError) -> String {
    match error.model_output().render() {
        text if !text.trim().is_empty() => text,
        _ => error.model_feedback().unwrap_or(error.message()).to_owned(),
    }
}

fn poll_task_calls(mut calls: Query<(Entity, &mut Running)>, mut commands: Commands) {
    for (entity, mut running) in &mut calls {
        if let Some(outcome) = check_ready(&mut running.0) {
            commands.entity(entity).remove::<Running>().insert(Done(outcome));
        }
    }
}

/// Report each call's outcome as it lands.
fn report_calls(
    calls: Query<(Entity, &Call, &CallOf, &Done), Without<Reported>>,
    mut events: MessageWriter<AgentEvent>,
    mut commands: Commands,
) {
    for (entity, call, owner, done) in &calls {
        let (output, error) = match &done.0 {
            Ok(output) => (output.render(), false),
            Err(text) => (text.clone(), true),
        };
        events.write(AgentEvent {
            agent: owner.0,
            kind: EventKind::CallFinished {
                name: call.call.function.name.to_string(),
                output,
                error,
            },
        });
        commands.entity(entity).insert(Reported);
    }
}

/// When every call of a reply is done, answer the model with the results
/// (and any steering input) and request again.
#[allow(clippy::type_complexity)]
fn finish_calls(
    mut agents: Query<(
        Entity,
        &mut Turn,
        &mut Inbox,
        &mut Conversation,
        &Calls,
        Option<&Session>,
    )>,
    calls: Query<(&Call, Option<&Done>)>,
    memory: Option<Res<SessionMemory>>,
    mut events: MessageWriter<AgentEvent>,
    mut commands: Commands,
) {
    for (agent, mut turn, mut inbox, mut conversation, owned, session) in &mut agents {
        if !matches!(*turn, Turn::Acting) {
            continue;
        }
        let mut finished: Vec<(&Call, &Done)> = Vec::new();
        for entity in owned.iter() {
            match calls.get(entity) {
                Ok((call, Some(done))) => finished.push((call, done)),
                _ => break,
            }
        }
        if finished.len() != owned.len() {
            continue;
        }
        finished.sort_by_key(|(call, _)| call.index);
        let interrupted = finished
            .iter()
            .any(|(_, done)| matches!(&done.0, Err(text) if text == "interrupted by the user"));
        let mut content: Vec<UserContent> = Vec::new();
        for (call, done) in &finished {
            let (id, name) = (call.call.id.clone(), call.call.function.name.clone());
            content.push(match &done.0 {
                Ok(output) => tool_result_output(id, name, output.clone()),
                Err(text) => tool_result_message(id, name, text.clone()),
            });
        }
        if !interrupted {
            content.extend(inbox.steering.drain(..).map(|text| {
                events.write(AgentEvent {
                    agent,
                    kind: EventKind::Input(text.clone()),
                });
                UserContent::text(text)
            }));
        }
        push_message(
            &mut conversation,
            session,
            memory.as_deref(),
            Message::User { content },
        );
        for entity in owned.iter() {
            commands.entity(entity).despawn();
        }
        if interrupted {
            *turn = Turn::Idle;
            events.write(AgentEvent {
                agent,
                kind: EventKind::TurnEnded,
            });
            commands.entity(agent).remove::<TurnSpan>();
        } else {
            *turn = Turn::Ready;
        }
    }
}
