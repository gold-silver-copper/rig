//! Agent sessions: what kind of agent an entity is, the transcript it shows,
//! and how it survives a restart.
//!
//! The conversation itself is stored by the glue in a Rig
//! `ConversationMemory` (rig-memory's `FileConversationMemory`, one JSON
//! Lines file per session). What only this agent needs (its kind, model,
//! transcript view, queued input and an unfinished tool batch) goes to a
//! small sidecar file per session. On startup the saved agents are spawned
//! again: after a reload always, after a quit only when a plugin asks for it
//! with [`ResumeOnLaunch`].

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use rig_core::completion::Message;
use rig_core::id::ConversationId;
use rig_core::memory::ConversationMemory;
use rig_core::message::{ToolCall, UserContent};
use rig_core::providers::registry::ProviderRef;
use rig_core::tool::ToolOutput;
use rig_core::transcript::{TranscriptError, tool_result_message, validate_canonical};
use rig_memory::FileConversationMemory;
use serde::{Deserialize, Serialize};

use crate::glue::{
    Agent, AgentEvent, AgentSet, Call, CallOf, Calls, Conversation, Done, EventKind, Inbox,
    Model, Reported, Session, SessionMemory, Turn, push_message, turn_span,
};
use crate::{Env, presets};

/// What an agent is for.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    /// The session the TUI was started with.
    Tui,
    /// A session a BRP client created.
    Remote,
    /// An eval run; never saved.
    Eval,
}

/// The agent the TUI shows.
#[derive(Component)]
pub struct Focused;

/// What the TUI and remote clients show of an agent.
#[derive(Component, Clone, Debug, Default, Serialize, Deserialize)]
pub struct Transcript(pub Vec<Entry>);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Entry {
    pub kind: EntryKind,
    pub text: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EntryKind {
    User,
    Assistant,
    Call,
    Result,
    Info,
    Error,
}

impl Transcript {
    pub fn log(&mut self, kind: EntryKind, text: impl Into<String>) {
        self.0.push(Entry {
            kind,
            text: text.into(),
        });
    }
}

/// Set by a plugin that wants the last sessions back after a quit, not only
/// after a reload.
#[derive(Resource)]
pub struct ResumeOnLaunch;

/// The raw conversation store, next to the [`SessionMemory`] requests are
/// loaded from (which a plugin may wrap to compact).
#[derive(Resource, Clone)]
pub struct Store(pub Arc<FileConversationMemory>);

impl Store {
    pub fn history(&self, id: &ConversationId) -> Vec<Message> {
        futures::executor::block_on(self.0.load(id)).unwrap_or_default()
    }
}

/// Everything about an agent that is not its conversation.
#[derive(Serialize, Deserialize)]
struct Sidecar {
    kind: Kind,
    name: String,
    model: ProviderRef,
    transcript: Vec<Entry>,
    follow_ups: VecDeque<String>,
    steering: Vec<String>,
    /// The model was about to be asked, or was being asked.
    pending_request: bool,
    /// An unfinished tool batch, in order; `done` holds finished results.
    calls: Vec<SavedCall>,
}

#[derive(Serialize, Deserialize, Clone)]
struct SavedCall {
    call: ToolCall,
    done: Option<Result<String, String>>,
}

pub struct SessionPlugin {
    /// The model the TUI session starts on, from the command line.
    pub model: Option<String>,
    /// Start a new TUI session even if the last one could be resumed.
    pub fresh: bool,
}

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        let env = app.world().resource::<Env>().clone();
        let store = FileConversationMemory::new(env.state.join("sessions"))
            .unwrap_or_else(|error| panic!("session store: {error}"));
        let store = Arc::new(store);
        app.insert_resource(Store(store.clone()))
            .insert_resource(SessionMemory(store))
            .insert_resource(LaunchOptions {
                model: self.model.clone(),
                fresh: self.fresh,
            })
            .add_systems(Startup, restore)
            .add_systems(Update, record_transcript.after(AgentSet::Run))
            .add_systems(Last, save_changed);
    }
}

#[derive(Resource)]
struct LaunchOptions {
    model: Option<String>,
    fresh: bool,
}

/// A new session id, unique within this state directory.
pub fn new_session_id(kind: Kind) -> ConversationId {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let prefix = match kind {
        Kind::Tui => "tui",
        Kind::Remote => "remote",
        Kind::Eval => "eval",
    };
    ConversationId::new(format!("{prefix}-{millis}-{n}"))
}

/// Spawn an agent with an empty conversation.
pub fn spawn_agent(
    commands: &mut Commands,
    preamble: String,
    kind: Kind,
    name: &str,
    model: ProviderRef,
    id: ConversationId,
) -> Entity {
    commands
        .spawn((
            Agent { preamble },
            Model(model),
            Session(id),
            kind,
            Name::new(name.to_owned()),
            Transcript::default(),
        ))
        .id()
}

pub fn default_model() -> ProviderRef {
    ProviderRef::parse(&presets()[0]).unwrap_or_else(|error| panic!("{error}"))
}

fn sidecar_dir(env: &Env) -> PathBuf {
    env.state.join("agents")
}

fn sidecar_path(env: &Env, id: &ConversationId) -> PathBuf {
    sidecar_dir(env).join(format!("{}.json", encode(id.as_str())))
}

fn encode(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

/// Turn agent events into transcript entries.
fn record_transcript(mut events: MessageReader<AgentEvent>, mut agents: Query<&mut Transcript>) {
    for event in events.read() {
        let Ok(mut transcript) = agents.get_mut(event.agent) else {
            continue;
        };
        match &event.kind {
            EventKind::Input(text) => transcript.log(EntryKind::User, text.clone()),
            EventKind::Text(text) => transcript.log(EntryKind::Assistant, text.clone()),
            EventKind::CallStarted { name, arguments } => {
                transcript.log(EntryKind::Call, format!("{name} {arguments}"))
            }
            EventKind::CallFinished { output, error, .. } => transcript.log(
                EntryKind::Result,
                if *error { format!("Error: {output}") } else { output.clone() },
            ),
            EventKind::Error(text) => transcript.log(EntryKind::Error, text.clone()),
            EventKind::TurnEnded => {}
        }
    }
}

type SavedAgent<'a> = (
    &'a Session,
    &'a Kind,
    &'a Name,
    &'a Model,
    &'a Transcript,
    &'a Inbox,
    &'a Turn,
    Option<&'a Calls>,
);

fn save_changed(
    env: Res<Env>,
    agents: Query<SavedAgent>,
    changed: Query<
        Entity,
        Or<(
            Changed<Transcript>,
            Changed<Model>,
            Changed<Inbox>,
            Changed<Turn>,
            Changed<Calls>,
        )>,
    >,
    done: Query<&CallOf, Changed<Done>>,
    calls: Query<(&Call, Option<&Done>)>,
    focused: Query<&Session, (With<Focused>, Changed<Focused>)>,
) {
    let dirty: Vec<Entity> = changed.iter().chain(done.iter().map(|c| c.0)).collect();
    for entity in dirty {
        if let Ok(agent) = agents.get(entity) {
            save(&env, agent, &calls);
        }
    }
    if let Ok(session) = focused.single() {
        let _ = std::fs::write(env.state.join("tui-session"), session.0.as_str());
    }
}

/// Save every agent now; a restart calls this before exiting.
pub fn save_all(world: &mut World) {
    let env = world.resource::<Env>().clone();
    let mut calls = world.query::<(&Call, Option<&Done>)>();
    let mut agents = world.query::<SavedAgent>();
    let calls = calls.query(world);
    for agent in agents.iter(world) {
        save(&env, agent, &calls);
    }
}

fn save(env: &Env, agent: SavedAgent, calls: &Query<(&Call, Option<&Done>)>) {
    let (session, kind, name, model, transcript, inbox, turn, owned) = agent;
    if *kind == Kind::Eval {
        return;
    }
    let mut saved: Vec<(usize, SavedCall)> = owned
        .into_iter()
        .flat_map(|owned| owned.iter())
        .filter_map(|entity| calls.get(entity).ok())
        .map(|(call, done)| {
            let done = done.map(|done| match &done.0 {
                Ok(output) => Ok(output.render()),
                Err(text) => Err(text.clone()),
            });
            (call.index, SavedCall { call: call.call.clone(), done })
        })
        .collect();
    saved.sort_by_key(|(index, _)| *index);
    let sidecar = Sidecar {
        kind: *kind,
        name: name.as_str().to_owned(),
        model: model.0.clone(),
        transcript: transcript.0.clone(),
        follow_ups: inbox.follow_ups.clone(),
        steering: inbox.steering.clone(),
        pending_request: matches!(turn, Turn::Ready | Turn::Thinking(_)),
        calls: saved.into_iter().map(|(_, call)| call).collect(),
    };
    let path = sidecar_path(env, &session.0);
    let result = std::fs::create_dir_all(sidecar_dir(env)).and_then(|()| {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&sidecar)?)?;
        std::fs::rename(tmp, &path)
    });
    if let Err(error) = result {
        error!("saving {}: {error}", path.display());
    }
}

/// Forget a session: its sidecar and its conversation.
pub fn forget(env: &Env, store: &Store, id: &ConversationId) {
    let _ = std::fs::remove_file(sidecar_path(env, id));
    let _ = futures::executor::block_on(store.0.clear(id));
}

fn load_sidecars(dir: &Path) -> Vec<(ConversationId, Sidecar)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| {
            let bytes = std::fs::read(entry.path()).ok()?;
            let sidecar: Sidecar = serde_json::from_slice(&bytes).ok()?;
            let stem = entry.path().file_stem()?.to_str()?.to_owned();
            Some((ConversationId::new(stem), sidecar))
        })
        .collect()
}

/// A saved session that is not running: its id, kind, name and model.
pub fn saved_sessions(env: &Env) -> Vec<(ConversationId, Kind, String, String)> {
    let mut saved: Vec<_> = load_sidecars(&sidecar_dir(env))
        .into_iter()
        .map(|(id, sidecar)| (id, sidecar.kind, sidecar.name, sidecar.model.to_string()))
        .collect();
    saved.sort_by(|a, b| b.0.cmp(&a.0));
    saved
}

/// Spawn the saved session `id`, as [`restore`] does after a restart.
pub fn resume_saved(world: &mut World, id: &ConversationId) -> Option<Entity> {
    let env = world.resource::<Env>().clone();
    let bytes = std::fs::read(sidecar_path(&env, id)).ok()?;
    let sidecar: Sidecar = serde_json::from_slice(&bytes).ok()?;
    let store = world.resource::<Store>().clone();
    let memory = world.resource::<SessionMemory>().clone();
    let preamble = crate::prompt::current(world);
    Some(revive(world, &store, &memory, &preamble, id.clone(), sidecar, &mut None))
}

/// Spawn the saved agents (after a reload, or when resuming), then make sure
/// there is a TUI session.
fn restore(world: &mut World) {
    let env = world.resource::<Env>().clone();
    let store = world.resource::<Store>().clone();
    let memory = world.resource::<SessionMemory>().clone();
    let options = world.remove_resource::<LaunchOptions>();
    let (cli_model, fresh) = options.map(|o| (o.model, o.fresh)).unwrap_or_default();
    let resume = env.restarted || (world.contains_resource::<ResumeOnLaunch>() && !fresh);
    let focused_id = std::fs::read_to_string(env.state.join("tui-session")).ok();
    let preamble = crate::prompt::current(world);
    let mut note = env.note.clone();
    let mut tui = None;

    if resume {
        for (id, sidecar) in load_sidecars(&sidecar_dir(&env)) {
            if sidecar.kind == Kind::Tui && focused_id.as_deref() != Some(id.as_str()) {
                continue;
            }
            let entity = revive(world, &store, &memory, &preamble, id.clone(), sidecar, &mut note);
            if focused_id.as_deref() == Some(id.as_str()) {
                tui = Some(entity);
            }
        }
    }
    let tui = tui.unwrap_or_else(|| {
        let mut commands = world.commands();
        let entity = spawn_agent(
            &mut commands,
            preamble.clone(),
            Kind::Tui,
            "tui",
            default_model(),
            new_session_id(Kind::Tui),
        );
        world.flush();
        entity
    });
    world.entity_mut(tui).insert(Focused);
    let mut agent = world.entity_mut(tui);
    if let Some(model) = cli_model {
        match crate::parse_model(&model) {
            Ok(model) => {
                agent.insert(Model(model.clone()));
                if let Some(mut transcript) = agent.get_mut::<Transcript>() {
                    transcript.log(EntryKind::Info, format!("model: {model}"));
                }
            }
            Err(error) => {
                if let Some(mut transcript) = agent.get_mut::<Transcript>() {
                    transcript.log(EntryKind::Error, error);
                }
            }
        }
    }
    if let Some(note) = note
        && let Some(mut transcript) = agent.get_mut::<Transcript>()
    {
        transcript.log(EntryKind::Info, note);
    }
}

/// Spawn one saved agent. An unfinished batch whose next call is `reload`
/// continues, with `note` as that call's result; any other unfinished batch
/// was cut short by a crash or quit, and its missing results say so.
fn revive(
    world: &mut World,
    store: &Store,
    memory: &SessionMemory,
    preamble: &str,
    id: ConversationId,
    sidecar: Sidecar,
    note: &mut Option<String>,
) -> Entity {
    let mut conversation = Conversation(store.history(&id));
    let session = Session(id);
    let resumes_reload = sidecar
        .calls
        .iter()
        .find(|call| call.done.is_none())
        .is_some_and(|call| call.call.function.name == crate::reload::TOOL);
    let mut turn = if sidecar.pending_request { Turn::Ready } else { Turn::Idle };
    let mut pending: Vec<(usize, SavedCall)> = Vec::new();
    if resumes_reload {
        let mut calls = sidecar.calls.clone();
        if let Some(call) = calls.iter_mut().find(|call| call.done.is_none()) {
            call.done = Some(Ok(note.take().unwrap_or_else(|| "The agent restarted.".into())));
        }
        pending = calls.into_iter().enumerate().collect();
        turn = Turn::Acting;
    } else if !sidecar.calls.is_empty() {
        let content: Vec<UserContent> = sidecar
            .calls
            .iter()
            .map(|saved| {
                let text = match &saved.done {
                    Some(Ok(text)) | Some(Err(text)) => text.clone(),
                    None => "interrupted: the agent stopped before this call finished".into(),
                };
                tool_result_message(saved.call.id.clone(), saved.call.function.name.clone(), text)
            })
            .collect();
        push_message(&mut conversation, Some(&session), Some(memory), Message::User { content });
        turn = Turn::Idle;
    }
    if !resumes_reload {
        repair(&mut conversation, &session, memory);
    }
    let name = Name::new(sidecar.name);
    // A turn cut by the restart goes on in a new span.
    let span = (!turn.is_idle()).then(|| turn_span(Some(&name), &sidecar.model));
    let entity = world
        .spawn((
            Agent {
                preamble: preamble.to_owned(),
            },
            Model(sidecar.model),
            conversation,
            session,
            sidecar.kind,
            name,
            Transcript(sidecar.transcript),
            Inbox {
                follow_ups: sidecar.follow_ups,
                steering: sidecar.steering,
            },
            turn,
        ))
        .id();
    if let Some(span) = span {
        world.entity_mut(entity).insert(span);
    }
    for (index, saved) in pending {
        let is_reload = saved.call.function.name == crate::reload::TOOL;
        let mut call = world.spawn((Call { call: saved.call, index }, CallOf(entity)));
        if let Some(done) = saved.done {
            call.insert(Done(done.map(ToolOutput::text)));
            // Reported before the restart; the reload call's note is new.
            if !is_reload {
                call.insert(Reported);
            }
        }
    }
    entity
}

/// Answer tool calls a restored history left open, so the next request is
/// valid; `validate_canonical` finds them.
fn repair(conversation: &mut Conversation, session: &Session, memory: &SessionMemory) {
    while let Err(TranscriptError::UnansweredToolCall { index, .. }) =
        validate_canonical(&conversation.0)
    {
        let Some(Message::Assistant { content, .. }) = conversation.0.get(index) else {
            break;
        };
        let answered: Vec<_> = match conversation.0.get(index + 1) {
            Some(Message::User { content }) => content
                .iter()
                .filter_map(|c| match c {
                    UserContent::ToolResult(result) => Some(result.call.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        let missing: Vec<UserContent> = content
            .iter()
            .filter_map(|c| match c {
                rig_core::message::AssistantContent::ToolCall(call) if !answered.contains(&call.id) => {
                    Some(tool_result_message(
                        call.id.clone(),
                        call.function.name.clone(),
                        "interrupted: the agent stopped before this call finished".into(),
                    ))
                }
                _ => None,
            })
            .collect();
        if missing.is_empty() || index + 1 != conversation.0.len() {
            break;
        }
        push_message(conversation, Some(session), Some(memory), Message::User { content: missing });
    }
}
