//! The agent loop as ECS state: the conversation, the prompt queue, the
//! selected model, and tool calls as entities that tool plugins answer.

use std::collections::{BTreeMap, VecDeque};
use std::future::Future;

use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use rig_core::DynModel;
use rig_core::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use rig_core::error::ProviderError;
use rig_core::message::{Message, ToolCall, ToolResultContent};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ProviderConfig, ProviderRef};
use rig_core::providers::{anthropic, deepseek, gemini, openai};

use crate::session::{CallState, Entry, EntryKind, Paths, Session};
use crate::{Options, ReloadStatus};

/// The models this agent is tested with: alias, vendor, model id.
pub const MODEL_ALIASES: &[(&str, &str, &str)] = &[
    ("sol", "openai", openai::GPT_6_1_SOL),
    ("opus", "anthropic", anthropic::completion::CLAUDE_OPUS_5_5),
    (
        "gemini",
        gemini::PROVIDER_NAME,
        gemini::completion::GEMINI_3_8_FLASH,
    ),
    ("deepseek", "deepseek", deepseek::DEEPSEEK_V4_1_FLASH),
];

pub const DEFAULT_MODEL: &str = "opus";

/// Anthropic binds thinking blocks to the conversation they were made in,
/// tools list included, and rejects them once it differs. Tools change here
/// (reloads, BRP plugins), so ask it to drop such blocks instead.
const THINKING_BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";

/// A resolved model selection.
pub struct Selection {
    pub reference: ProviderRef,
    pub model: DynModel<Completion>,
    /// Provider-specific request fields to send with every request.
    pub params: Option<serde_json::Value>,
}

/// Resolve an alias or provider reference into a model, credentials from the
/// environment.
pub fn resolve_model(name: &str) -> anyhow::Result<Selection> {
    let spelled = MODEL_ALIASES
        .iter()
        .find(|(alias, ..)| *alias == name)
        .map_or_else(
            || name.to_string(),
            |(_, vendor, model)| format!("{vendor}:{model}"),
        );
    let mut reference = ProviderRef::parse(&spelled)?;
    let mut params = None;
    if reference
        .id()
        .is_some_and(|id| id.vendor() == anthropic::ANTHROPIC.name)
    {
        // The credential is dropped by `configured` and read again from the
        // environment by `completion_model`.
        if let ProviderConfig::Anthropic(config) = reference.config("") {
            reference = ProviderRef::configured(
                ProviderConfig::Anthropic(config.with_beta(THINKING_BINDING_BETA)),
                reference.model(),
            )?;
            params = Some(serde_json::json!({
                "thinking": { "block_binding": { "prefix_mismatch_behavior": "drop_block" } }
            }));
        }
    }
    let model = reference.completion_model()?;
    Ok(Selection {
        reference,
        model,
        params,
    })
}

/// The agent loop. Expects the [`Session`] to continue as a resource.
pub struct AgentPlugin(pub Options);

/// Settings every plugin can read.
#[derive(Resource, Clone)]
pub struct Config {
    pub paths: Paths,
    pub brp_port: u16,
    /// The agent's own crate, which `reload` rebuilds.
    pub source_dir: std::path::PathBuf,
}

/// The runtime model calls and slow tools run on.
#[derive(Resource)]
pub struct Tokio(pub tokio::runtime::Runtime);

#[derive(Resource, Default)]
pub struct Conversation(pub Vec<Message>);

#[derive(Resource, Default)]
pub struct Transcript(pub Vec<Entry>);

impl Transcript {
    pub fn push(&mut self, kind: EntryKind, text: impl Into<String>) {
        self.0.push(Entry {
            kind,
            text: text.into(),
        });
    }
}

#[derive(Resource, Default)]
pub struct PromptQueue(pub VecDeque<String>);

/// Runtime events the model has not heard of yet, sent ahead of the next
/// prompt.
#[derive(Resource, Default)]
pub struct Notes(pub Vec<String>);

/// The selected model. `selection` is `None` when the selection has no
/// credential, so the agent can still start and be switched.
#[derive(Resource)]
pub struct ActiveModel {
    pub name: String,
    pub selection: Option<Selection>,
}

impl ActiveModel {
    fn select(name: &str) -> (Self, Option<anyhow::Error>) {
        let (selection, error) = match resolve_model(name) {
            Ok(selection) => (Some(selection), None),
            Err(error) => (None, Some(error)),
        };
        let active = Self {
            name: name.to_string(),
            selection,
        };
        (active, error)
    }

    pub fn label(&self) -> String {
        self.selection
            .as_ref()
            .map_or_else(|| self.name.clone(), |s| s.reference.to_string())
    }
}

/// Where the current turn is.
#[derive(Resource, Default)]
pub enum Turn {
    #[default]
    Idle,
    /// Waiting for the model.
    Thinking(Receiver<Result<CompletionResponse, ProviderError>>),
    /// Waiting for the tool calls of the last response.
    Tools,
}

impl Turn {
    pub fn is_idle(&self) -> bool {
        matches!(self, Turn::Idle)
    }
}

/// Ask for another model, keeping the conversation.
#[derive(Message)]
pub struct SwitchModel(pub String);

/// A tool plugins registered, by name.
pub struct RegisteredTool {
    pub definition: ToolDefinition,
    pub handler: SystemId<In<ToolInvocation>>,
}

#[derive(Resource, Default)]
pub struct ToolRegistry(pub BTreeMap<String, RegisteredTool>);

/// What a tool handler receives: the call and the entity to answer on.
pub struct ToolInvocation {
    pub entity: Entity,
    pub call: ToolCall,
}

/// A call of the current batch; `order` is its place in the response.
#[derive(Component)]
pub struct PendingCall {
    pub call: ToolCall,
    pub order: usize,
}

/// A call's answer. The turn continues once every [`PendingCall`] has one.
#[derive(Component)]
pub struct CallOutput(pub String);

/// An output restored after a reload, already shown before it.
#[derive(Component)]
struct Shown;

/// Answers tool calls from async work: `spawn` runs a future on the Tokio
/// runtime and inserts its output as the call's [`CallOutput`].
#[derive(Resource, Clone)]
pub struct ToolTasks {
    handle: tokio::runtime::Handle,
    sender: Sender<(Entity, String)>,
}

impl ToolTasks {
    pub fn spawn(&self, entity: Entity, work: impl Future<Output = String> + Send + 'static) {
        let sender = self.sender.clone();
        self.handle.spawn(async move {
            let _ = sender.send((entity, work.await));
        });
    }

    pub fn handle(&self) -> &tokio::runtime::Handle {
        &self.handle
    }
}

#[derive(Resource)]
struct ToolOutputs(Receiver<(Entity, String)>);

/// Registering tools from any plugin.
pub trait AgentAppExt {
    /// Offer a tool to the model. `handler` runs once per call and must
    /// eventually insert a [`CallOutput`] on the invocation's entity, directly
    /// or through [`ToolTasks`].
    fn add_tool<M>(
        &mut self,
        name: &str,
        description: &str,
        parameters: serde_json::Value,
        handler: impl IntoSystem<In<ToolInvocation>, (), M> + 'static,
    ) -> &mut Self;
}

impl AgentAppExt for App {
    fn add_tool<M>(
        &mut self,
        name: &str,
        description: &str,
        parameters: serde_json::Value,
        handler: impl IntoSystem<In<ToolInvocation>, (), M> + 'static,
    ) -> &mut Self {
        let handler = self.world_mut().register_system(handler);
        self.world_mut()
            .get_resource_or_init::<ToolRegistry>()
            .0
            .insert(
                name.to_string(),
                RegisteredTool {
                    definition: ToolDefinition {
                        name: name.to_string(),
                        description: description.to_string(),
                        parameters,
                    },
                    handler,
                },
            );
        self
    }
}

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        let options = &self.0;
        let session = app
            .world_mut()
            .remove_resource::<Session>()
            .unwrap_or_default();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|error| panic!("cannot start the Tokio runtime: {error}"));
        let (sender, receiver) = crossbeam_channel::unbounded();
        let tasks = ToolTasks {
            handle: runtime.handle().clone(),
            sender,
        };

        let (active, model_error) = ActiveModel::select(&session.model);
        let mut transcript = Transcript(session.transcript);
        if let Some(error) = model_error {
            transcript.push(
                EntryKind::Error,
                format!("model `{}`: {error:#}", session.model),
            );
        }

        // An interrupted turn resumes where it stopped: its calls come back as
        // entities, the reload call answered with how the reload went.
        let reload_note = match &options.reload {
            Some(ReloadStatus::Succeeded) => "Reload succeeded: the new build compiled, started, \
                 and is now running. The conversation was carried over."
                .to_string(),
            Some(ReloadStatus::Failed(reason)) => format!(
                "Reload failed: the new build compiled, but the new binary {reason}\n\
                 The agent fell back to the last binary that started successfully and is \
                 running that. Your source edits are still on disk."
            ),
            None => "error: this call was interrupted by a restart".to_string(),
        };
        let turn = if session.batch.is_empty() {
            Turn::Idle
        } else {
            Turn::Tools
        };
        for (order, CallState { call, output }) in session.batch.into_iter().enumerate() {
            let pending = PendingCall { call, order };
            match output {
                Some(output) => app.world_mut().spawn((pending, CallOutput(output), Shown)),
                None if session.reload_call == Some(call_key(&pending.call)) => app
                    .world_mut()
                    .spawn((pending, CallOutput(reload_note.clone()))),
                None => app.world_mut().spawn((
                    pending,
                    CallOutput("error: this call was interrupted by a reload".to_string()),
                )),
            };
        }
        // A `/reload` answers no call, so the model hears of it with the next
        // prompt.
        let mut notes = Notes(session.notes);
        if options.reload.is_some() && session.reload_call.is_none() {
            notes.0.push(format!("The user ran /reload. {reload_note}"));
        }
        match &options.reload {
            Some(ReloadStatus::Succeeded) => {
                transcript.push(EntryKind::Info, "reloaded into the new build")
            }
            Some(ReloadStatus::Failed(reason)) => transcript.push(
                EntryKind::Error,
                format!(
                    "reload failed, fell back to the last good binary: the new binary {reason}"
                ),
            ),
            None => {}
        }

        app.insert_resource(Config {
            paths: Paths::new(options.state_dir()),
            brp_port: options.brp_port.unwrap_or(0),
            source_dir: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        })
        .insert_resource(Tokio(runtime))
        .insert_resource(tasks)
        .insert_resource(ToolOutputs(receiver))
        .insert_resource(Conversation(session.messages))
        .insert_resource(transcript)
        .insert_resource(PromptQueue(session.queue))
        .insert_resource(notes)
        .insert_resource(crate::brp::SavedExternalTools(session.external_tools))
        .insert_resource(active)
        .insert_resource(turn)
        .init_resource::<ToolRegistry>()
        .add_message::<SwitchModel>()
        .add_systems(
            Update,
            (switch_model, collect_tool_outputs, show_outputs, drive_turn).chain(),
        );
    }
}

/// The key a call is known by across a restart and to external plugins.
pub fn call_key(call: &ToolCall) -> String {
    call.id.wire().into_owned()
}

fn switch_model(
    mut requests: MessageReader<SwitchModel>,
    mut active: ResMut<ActiveModel>,
    mut transcript: ResMut<Transcript>,
    config: Res<Config>,
    mut commands: Commands,
) {
    for SwitchModel(name) in requests.read() {
        match resolve_model(name) {
            Ok(selection) => {
                *active = ActiveModel {
                    name: name.clone(),
                    selection: Some(selection),
                };
                transcript.push(EntryKind::Info, format!("model: {}", active.label()));
                commands.queue(save_session_command(config.paths.clone()));
            }
            Err(error) => transcript.push(EntryKind::Error, format!("model `{name}`: {error:#}")),
        }
    }
}

fn collect_tool_outputs(outputs: Res<ToolOutputs>, mut commands: Commands) {
    for (entity, output) in outputs.0.try_iter() {
        if let Ok(mut entity) = commands.get_entity(entity) {
            entity.insert(CallOutput(output));
        }
    }
}

fn show_outputs(
    outputs: Query<&CallOutput, (Added<CallOutput>, Without<Shown>)>,
    mut transcript: ResMut<Transcript>,
) {
    for CallOutput(output) in &outputs {
        transcript.push(EntryKind::ToolResult, output.clone());
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_turn(
    mut turn: ResMut<Turn>,
    mut queue: ResMut<PromptQueue>,
    mut notes: ResMut<Notes>,
    mut conversation: ResMut<Conversation>,
    mut transcript: ResMut<Transcript>,
    active: Res<ActiveModel>,
    registry: Res<ToolRegistry>,
    tokio: Res<Tokio>,
    config: Res<Config>,
    calls: Query<(Entity, &PendingCall, Option<&CallOutput>)>,
    restart: Option<Res<crate::reload::RestartPending>>,
    mut commands: Commands,
) {
    match &*turn {
        Turn::Idle => {
            // A reload waiting for the turn to end takes precedence.
            if restart.is_some() {
                return;
            }
            let Some(prompt) = queue.0.pop_front() else {
                return;
            };
            transcript.push(EntryKind::User, prompt.clone());
            let prompt = match std::mem::take(&mut notes.0) {
                notes if notes.is_empty() => prompt,
                notes => format!("[agent runtime: {}]\n\n{prompt}", notes.join(" ")),
            };
            conversation.0.push(Message::user(prompt));
            *turn = request(
                &active,
                &registry,
                &tokio,
                &conversation,
                &config,
                &mut transcript,
            );
        }
        Turn::Thinking(receiver) => {
            let Ok(result) = receiver.try_recv() else {
                return;
            };
            let response = match result {
                Ok(response) => response,
                Err(error) => {
                    transcript.push(EntryKind::Error, format!("{error}"));
                    *turn = Turn::Idle;
                    commands.queue(save_session_command(config.paths.clone()));
                    return;
                }
            };
            let text = response.text();
            if !text.trim().is_empty() {
                transcript.push(EntryKind::Assistant, text);
            }
            let tool_calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
            conversation.0.extend(response.message());
            if tool_calls.is_empty() {
                *turn = Turn::Idle;
                commands.queue(save_session_command(config.paths.clone()));
                return;
            }
            for (order, call) in tool_calls.into_iter().enumerate() {
                transcript.push(
                    EntryKind::ToolCall,
                    format!("{} {}", call.function.name, call.function.arguments),
                );
                let entity = commands.spawn(PendingCall {
                    call: call.clone(),
                    order,
                });
                let entity = entity.id();
                match registry.0.get(call.function.name.as_str()) {
                    Some(tool) => {
                        commands.run_system_with(tool.handler, ToolInvocation { entity, call })
                    }
                    None => {
                        let name = call.function.name.to_string();
                        commands.entity(entity).insert(CallOutput(format!(
                            "error: there is no tool named `{name}`"
                        )));
                    }
                }
            }
            *turn = Turn::Tools;
        }
        Turn::Tools => {
            let mut done: Vec<(Entity, &PendingCall, &CallOutput)> = Vec::new();
            for (entity, pending, output) in &calls {
                let Some(output) = output else { return };
                done.push((entity, pending, output));
            }
            if restart.is_some() {
                return;
            }
            done.sort_by_key(|(_, pending, _)| pending.order);
            let mut results = Vec::new();
            for (entity, pending, CallOutput(output)) in done {
                // Providers reject empty tool results.
                let output = if output.is_empty() {
                    "(no output)"
                } else {
                    output
                };
                results.push(pending.call.result(vec![ToolResultContent::text(output)]));
                commands.entity(entity).despawn();
            }
            conversation.0.push(Message::tool_results(results));
            *turn = request(
                &active,
                &registry,
                &tokio,
                &conversation,
                &config,
                &mut transcript,
            );
        }
    }
}

/// Send the conversation to the model.
fn request(
    active: &ActiveModel,
    registry: &ToolRegistry,
    tokio: &Tokio,
    conversation: &Conversation,
    config: &Config,
    transcript: &mut Transcript,
) -> Turn {
    let Some(selection) = &active.selection else {
        transcript.push(
            EntryKind::Error,
            format!(
                "model `{}` is unavailable; pick another with /model",
                active.name
            ),
        );
        return Turn::Idle;
    };
    let mut history = conversation.0.clone();
    let Some(prompt) = history.pop() else {
        return Turn::Idle;
    };
    let mut request = CompletionRequest::new(prompt)
        .messages(history)
        .preamble(system_prompt(config))
        .tools(
            registry
                .0
                .values()
                .map(|tool| tool.definition.clone())
                .collect(),
        );
    if let Some(params) = &selection.params {
        request = request.additional_params(params.clone());
    }
    let (sender, receiver) = crossbeam_channel::bounded(1);
    let call = selection.model.call(request);
    tokio.0.spawn(async move {
        let _ = sender.send(call.await);
    });
    Turn::Thinking(receiver)
}

fn system_prompt(config: &Config) -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    format!(
        "You are bevy-agent, a minimal coding agent running in a terminal UI.\n\
         Working directory: {cwd}\n\
         Your own source code is the Rust crate at {source} (package `bevy-agent`, a Bevy app). \
         It depends on the Rig crates in {rig} by path, and you may edit those too. \
         Native plugins are ordinary Bevy plugins in {source}/src/plugins/; each tool is \
         registered with `app.add_tool(..)`.\n\
         After changing your own source, call the `reload` tool: it rebuilds you, and if the \
         build succeeds you restart into the new binary with this conversation intact, so you \
         can keep working in the same turn. If the build fails you get the compiler errors \
         and keep running the old binary.\n\
         Be concise. Use tools rather than guessing.",
        cwd = cwd.display(),
        source = config.source_dir.display(),
        rig = config
            .source_dir
            .parent()
            .unwrap_or(&config.source_dir)
            .display(),
    )
}

/// Save everything a reload must carry over.
pub fn save_session(world: &mut World) -> anyhow::Result<()> {
    let session = snapshot(world);
    let paths = world.resource::<Config>().paths.clone();
    session.save(&paths)
}

pub fn save_session_command(paths: Paths) -> impl FnOnce(&mut World) {
    move |world: &mut World| {
        if let Err(error) = snapshot(world).save(&paths) {
            world
                .resource_mut::<Transcript>()
                .push(EntryKind::Error, format!("saving the session: {error:#}"));
        }
    }
}

fn snapshot(world: &mut World) -> Session {
    let mut batch: Vec<(usize, CallState)> = world
        .query::<(&PendingCall, Option<&CallOutput>)>()
        .iter(world)
        .map(|(pending, output)| {
            (
                pending.order,
                CallState {
                    call: pending.call.clone(),
                    output: output.map(|o| o.0.clone()),
                },
            )
        })
        .collect();
    batch.sort_by_key(|(order, _)| *order);
    Session {
        model: world.resource::<ActiveModel>().name.clone(),
        messages: world.resource::<Conversation>().0.clone(),
        transcript: world.resource::<Transcript>().0.clone(),
        queue: world.resource::<PromptQueue>().0.clone(),
        notes: world.resource::<Notes>().0.clone(),
        batch: batch.into_iter().map(|(_, state)| state).collect(),
        reload_call: world
            .get_resource::<crate::reload::RestartPending>()
            .and_then(|pending| pending.call.clone()),
        external_tools: world
            .get_resource::<crate::brp::ExternalTools>()
            .map(|tools| tools.0.values().cloned().collect())
            .unwrap_or_default(),
    }
}
