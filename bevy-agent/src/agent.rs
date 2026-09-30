//! The turn loop. A queued prompt goes to the model with the conversation
//! and every registered tool; the tool calls it answers with run one at a
//! time; their results go back to the model until it answers without tools.
//! Tools are one-shot Bevy systems that plugins register with [`AddTool`].

use std::collections::BTreeMap;
use std::error::Error;

use async_channel::{Receiver, TryRecvError};
use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use futures::StreamExt;
use rig_core::DynModel;
use rig_core::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use rig_core::message::{Message, ToolCall, ToolResultContent};
use rig_core::operation::Completion;
use rig_core::providers::registry::ProviderRef;
use rig_core::streaming::{Item, StreamEvent};
use serde_json::Value;

use crate::reload::Reload;
use crate::session::{self, Entry, Session, StateDir, Turn};

/// Models offered by `/model`; any `vendor[/format]:model` Rig knows works.
const MODELS: [&str; 4] = [
    "openai:gpt-6.1-sol",
    "anthropic:claude-opus-5-5",
    "gemini:gemini-3.8-flash",
    "deepseek:deepseek-flash",
];

pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap_or_else(|error| panic!("cannot start the tokio runtime: {error}"));
        let mut session = app.world_mut().resource_mut::<Session>();
        let model = connect(&session.model);
        if let Err(error) = &model {
            let message = format!("Cannot use {}: {error}", session.model);
            session.log(Entry::Error(message));
        }
        app.insert_resource(Llm { runtime, model })
            .init_resource::<Tools>()
            .init_resource::<InFlight>()
            .init_resource::<Running>()
            .add_systems(
                Update,
                (start_turn, drive_request, drive_tools, session::autosave).chain(),
            )
            .add_systems(Last, session::save_on_exit);
    }
}

/// The model the next request goes to, and the runtime Rig's HTTP client runs on.
#[derive(Resource)]
pub struct Llm {
    runtime: tokio::runtime::Runtime,
    model: Result<DynModel<Completion>, String>,
}

/// A completion model for `vendor[/format]:model`, credentialed from the environment.
pub fn connect(spec: &str) -> Result<DynModel<Completion>, String> {
    let reference = ProviderRef::parse(spec).map_err(|error| report(&error))?;
    reference.completion_model().map_err(|error| report(&error))
}

/// Switches models without touching the conversation; the next request uses it.
pub fn switch_model(session: &mut Session, llm: &mut Llm, spec: &str) {
    match connect(spec) {
        Ok(model) => {
            llm.model = Ok(model);
            session.model = spec.to_owned();
            session.log(Entry::Notice(format!("Model: {spec}")));
        }
        Err(error) => session.log(Entry::Error(format!("Cannot use {spec}: {error}"))),
    }
}

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

/// What a tool system returns: its output now, or a channel it arrives on.
pub enum ToolReply {
    Done(String),
    Pending(Receiver<String>),
}

impl ToolReply {
    /// Runs `work` on its own thread, so slow tools do not stall the app.
    pub fn spawn(work: impl FnOnce() -> String + Send + 'static) -> Self {
        let (sender, receiver) = async_channel::bounded(1);
        std::thread::spawn(move || {
            let _ = sender.send_blocking(work());
        });
        Self::Pending(receiver)
    }
}

/// Parses tool arguments, or explains to the model why they do not fit.
pub fn arguments<T: serde::de::DeserializeOwned>(value: Value) -> Result<T, ToolReply> {
    serde_json::from_value(value)
        .map_err(|error| ToolReply::Done(format!("error: invalid arguments: {error}")))
}

pub enum Handler {
    Native(SystemId<In<Value>, ToolReply>),
    /// Served by an external process over BRP; see [`crate::brp`].
    External,
}

/// Every tool the model can call, by name.
#[derive(Resource, Default)]
pub struct Tools(BTreeMap<String, (ToolDefinition, Handler)>);

impl Tools {
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.0
            .values()
            .map(|(definition, _)| definition.clone())
            .collect()
    }

    /// Adds or replaces an external tool. Returns whether anything changed.
    pub fn register_external(&mut self, definition: &ToolDefinition) -> bool {
        match self.0.get(&definition.name) {
            Some((existing, Handler::External)) if existing == definition => false,
            Some((_, Handler::Native(_))) => false,
            _ => {
                self.0.insert(
                    definition.name.clone(),
                    (definition.clone(), Handler::External),
                );
                true
            }
        }
    }
}

pub trait AddTool {
    /// Registers a tool the model can call. `handler` is a system that takes
    /// the call's JSON arguments.
    fn add_tool<M>(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        handler: impl IntoSystem<In<Value>, ToolReply, M> + 'static,
    ) -> &mut Self;
}

impl AddTool for App {
    fn add_tool<M>(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        handler: impl IntoSystem<In<Value>, ToolReply, M> + 'static,
    ) -> &mut Self {
        let id = self.world_mut().register_system(handler);
        let definition = ToolDefinition {
            name: name.into(),
            description: description.into(),
            parameters,
        };
        self.world_mut()
            .get_resource_or_init::<Tools>()
            .0
            .insert(name.into(), (definition, Handler::Native(id)));
        self
    }
}

enum LlmEvent {
    Text(String),
    Done(Box<Result<CompletionResponse, String>>),
}

/// The model request in flight, if any.
#[derive(Resource, Default)]
pub struct InFlight(Option<Flight>);

struct Flight {
    task: tokio::task::JoinHandle<()>,
    events: Receiver<LlmEvent>,
    streamed_text: bool,
}

/// The reply of the tool call that is running, if any.
#[derive(Resource, Default)]
pub struct Running(Option<Receiver<String>>);

fn preamble() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let source = env!("CARGO_MANIFEST_DIR");
    format!(
        "You are bevy-agent, a minimal coding agent in a terminal. Be concise.\n\
         Working directory: {cwd}\n\
         Your own source code is the Bevy app at {source}. Each native tool is an \
         ordinary Bevy plugin in {source}/src/plugins/, registered in \
         src/plugins/mod.rs. You use Rig for models; its crates are in {source}/../crates.\n\
         To change yourself, edit that source and call `reload`. It rebuilds you; if \
         the build fails you get the compiler errors and keep running unchanged. If it \
         succeeds you restart into the new binary with this conversation intact and \
         the `reload` call returns, so you can use your changes in the same turn.",
        cwd = cwd.display(),
    )
}

/// Handles a line the user sent: a command now, or a prompt into the queue.
pub fn submit(world: &mut World, text: &str) {
    let (command, argument) = text.split_once(' ').unwrap_or((text, ""));
    match command {
        "" => {}
        "/quit" => {
            world.write_message(AppExit::Success);
        }
        "/model" if argument.is_empty() => {
            let mut session = world.resource_mut::<Session>();
            let current = session.model.clone();
            session.log(Entry::Notice(format!(
                "Model: {current}. Switch with /model <vendor:model>, for example:\n{}",
                MODELS.join("\n")
            )));
        }
        "/model" => world.resource_scope(|world, mut llm: Mut<Llm>| {
            switch_model(
                &mut world.resource_mut::<Session>(),
                &mut llm,
                argument.trim(),
            );
        }),
        _ => world
            .resource_mut::<Session>()
            .queue
            .push_back(text.to_owned()),
    }
}

/// Starts the next queued prompt, or a queued `/reload`, once idle.
fn start_turn(mut session: ResMut<Session>, mut reload: ResMut<Reload>, state: Res<StateDir>) {
    if session.busy() || reload.building() {
        return;
    }
    let Some(front) = session.queue.front() else {
        return;
    };
    if front == "/reload" {
        session.queue.pop_front();
        session.log(Entry::Notice("Building…".into()));
        reload.start(None, &state.0);
        return;
    }
    let Some(prompt) = session.queue.pop_front() else {
        return;
    };
    session.log(Entry::User(prompt.clone()));
    let notes = std::mem::take(&mut session.notes);
    let text = if notes.is_empty() {
        prompt
    } else {
        format!("[agent notes]\n{}\n\n{prompt}", notes.join("\n"))
    };
    session.messages.push(Message::user(text));
    session.turn = Turn::Request;
}

fn drive_request(
    mut session: ResMut<Session>,
    mut flight: ResMut<InFlight>,
    llm: Res<Llm>,
    tools: Res<Tools>,
) {
    if session.turn != Turn::Request {
        return;
    }
    let Some(current) = &mut flight.0 else {
        match &llm.model {
            Ok(model) => flight.0 = Some(send(&llm, model, &session, &tools)),
            Err(error) => {
                session.log(Entry::Error(error.clone()));
                session.turn = Turn::Idle;
            }
        }
        return;
    };
    while let Ok(event) = current.events.try_recv() {
        match event {
            LlmEvent::Text(text) => {
                match session.transcript.last_mut() {
                    Some(Entry::Assistant(streamed)) if current.streamed_text => {
                        streamed.push_str(&text)
                    }
                    _ => session.log(Entry::Assistant(text)),
                }
                current.streamed_text = true;
            }
            LlmEvent::Done(done) => {
                let streamed_text = current.streamed_text;
                flight.0 = None;
                session.turn = match *done {
                    Ok(response) => answered(&mut session, response, streamed_text),
                    Err(error) => {
                        session.log(Entry::Error(error));
                        Turn::Idle
                    }
                };
                return;
            }
        }
    }
}

/// Records the model's answer and returns what the turn does next.
fn answered(session: &mut Session, response: CompletionResponse, streamed_text: bool) -> Turn {
    if !streamed_text && !response.text().is_empty() {
        session.log(Entry::Assistant(response.text()));
    }
    let calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
    match response.message() {
        Some(message) => session.messages.push(message),
        None => session.log(Entry::Notice("(empty response)".into())),
    }
    if calls.is_empty() {
        Turn::Idle
    } else {
        Turn::Tools {
            calls,
            results: Vec::new(),
        }
    }
}

fn send(llm: &Llm, model: &DynModel<Completion>, session: &Session, tools: &Tools) -> Flight {
    let mut request = CompletionRequest::new(Message::system(preamble()));
    request
        .chat_history
        .extend(session.messages.iter().cloned());
    request.tools = tools.definitions();
    let model = model.clone();
    let (sender, events) = async_channel::unbounded();
    let task = llm.runtime.spawn(async move {
        let result = async {
            let mut stream = model.stream(request)?;
            while let Some(item) = stream.next().await {
                if let Item::Event(StreamEvent::Text { text, .. }) = item? {
                    let _ = sender.send(LlmEvent::Text(text)).await;
                }
            }
            stream.finish().await
        }
        .await;
        let _ = sender
            .send(LlmEvent::Done(Box::new(
                result.map_err(|error| report(&error)),
            )))
            .await;
    });
    Flight {
        task,
        events,
        streamed_text: false,
    }
}

/// Runs the pending tool calls in order and hands their results back to the model.
fn drive_tools(world: &mut World) {
    let Turn::Tools { calls, results } = &world.resource::<Session>().turn else {
        return;
    };
    let Some(call) = calls.get(results.len()).cloned() else {
        let mut session = world.resource_mut::<Session>();
        if let Turn::Tools { results, .. } = std::mem::take(&mut session.turn) {
            session.messages.push(Message::tool_results(results));
        }
        session.turn = Turn::Request;
        return;
    };
    let output = match world.resource_mut::<Running>().0.take() {
        Some(reply) => match reply.try_recv() {
            Ok(output) => output,
            Err(TryRecvError::Empty) => {
                world.resource_mut::<Running>().0 = Some(reply);
                return;
            }
            Err(TryRecvError::Closed) => "error: the tool ended without a result".into(),
        },
        None => {
            world.resource_mut::<Session>().log(Entry::Call(format!(
                "{} {}",
                call.function.name, call.function.arguments
            )));
            match dispatch(world, &call) {
                ToolReply::Done(output) => output,
                ToolReply::Pending(reply) => {
                    world.resource_mut::<Running>().0 = Some(reply);
                    return;
                }
            }
        }
    };
    let mut session = world.resource_mut::<Session>();
    session.log(Entry::Output(output.clone()));
    if let Turn::Tools { results, .. } = &mut session.turn {
        results.push(call.result(vec![ToolResultContent::text(output)]));
    }
}

fn dispatch(world: &mut World, call: &ToolCall) -> ToolReply {
    let name = call.function.name.as_str();
    let handler = match world.resource::<Tools>().0.get(name) {
        Some((_, Handler::Native(id))) => Some(*id),
        Some((_, Handler::External)) => None,
        None => return ToolReply::Done(format!("error: there is no tool named `{name}`")),
    };
    match handler {
        Some(id) => world
            .run_system_with(id, call.function.arguments.clone())
            .unwrap_or_else(|error| ToolReply::Done(format!("error: {error}"))),
        None => world.resource_mut::<crate::brp::External>().enqueue(call),
    }
}

/// Stops the turn: drops the model request, or answers the remaining tool
/// calls as cancelled so the conversation stays well formed.
pub fn cancel(world: &mut World) {
    if let Some(flight) = world.resource_mut::<InFlight>().0.take() {
        flight.task.abort();
    }
    world.resource_mut::<Running>().0 = None;
    let mut session = world.resource_mut::<Session>();
    match std::mem::take(&mut session.turn) {
        Turn::Idle => return,
        Turn::Request => {}
        Turn::Tools { calls, mut results } => {
            for call in &calls[results.len()..] {
                results.push(call.result(vec![ToolResultContent::text("cancelled by the user")]));
            }
            session.messages.push(Message::tool_results(results));
        }
    }
    session.log(Entry::Notice("Cancelled.".into()));
}
