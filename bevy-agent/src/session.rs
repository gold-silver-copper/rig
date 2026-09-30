//! Agent sessions: what an agent is besides its Rig state. Each agent entity
//! also carries a [`SessionId`], an [`Origin`], the [`Transcript`] people see
//! and its queued [`Prompts`]. This module spawns agents, turns typed lines
//! into prompts and commands, and saves and restores sessions: the
//! conversation through Rig's `FileConversationMemory`, everything else in a
//! small JSON file next to it.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;

use bevy::prelude::*;
use futures::executor::block_on;
use rig_core::id::ConversationId;
use rig_core::memory::ConversationMemory;
use rig_core::message::Message;
use rig_memory::FileConversationMemory;
use serde::{Deserialize, Serialize};

use crate::glue::{Agent, AgentEvent, Conversation, GlueSystems, Instructions, Model, ModelFactory, Turn};
use crate::reload::Reload;

pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SlashCommands>()
            .add_message::<SlashCommand>()
            .add_systems(Update, start_turns.before(GlueSystems))
            .add_systems(Update, record_transcript.after(GlueSystems));
    }
}

/// The durable identity of an agent's session.
#[derive(Component, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

/// Who drives an agent.
#[derive(Component, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum Origin {
    Tui,
    Remote,
    Eval,
}

/// What people see of a session.
#[derive(Component, Default, Clone, Serialize, Deserialize)]
pub struct Transcript(pub Vec<Entry>);

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub enum Entry {
    User(String),
    Assistant(String),
    Call(String),
    Output(String),
    Notice(String),
    Error(String),
}

impl Transcript {
    pub fn log(&mut self, entry: Entry) {
        self.0.push(entry);
    }
}

/// Prompts and queued `/reload`s waiting for the agent to be idle, and
/// notes the model should hear with the next prompt.
#[derive(Component, Default, Clone, Serialize, Deserialize)]
pub struct Prompts {
    pub queue: VecDeque<String>,
    pub notes: Vec<String>,
}

/// Tokens a session has used, as providers report them.
#[derive(Component, Default, Clone, Copy)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
}

/// The system prompt new agents get.
#[derive(Resource, Clone)]
pub struct Preamble(pub String);

/// A new session id: the time, readable and sortable, plus a counter.
pub fn new_session_id() -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{millis}-{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Spawns an agent with an empty conversation.
pub fn spawn_agent(world: &mut World, id: String, origin: Origin, model: &str) -> Entity {
    let model = world.resource::<ModelFactory>().model(model);
    let instructions = Instructions(world.resource::<Preamble>().0.clone());
    world
        .spawn((
            Agent,
            SessionId(id),
            origin,
            model,
            instructions,
            Transcript::default(),
            Prompts::default(),
            Usage::default(),
        ))
        .id()
}

/// Slash commands plugins add, with a line of help each. A typed command in
/// this list becomes a [`SlashCommand`] message for its plugin.
#[derive(Resource, Default)]
pub struct SlashCommands(pub BTreeMap<String, String>);

#[derive(Message, Clone)]
pub struct SlashCommand {
    pub agent: Entity,
    pub name: String,
    pub argument: String,
}

/// Models `/model` offers and Ctrl+P cycles through.
pub const MODELS: [&str; 4] = [
    "openai:gpt-6.1-sol",
    "anthropic:claude-opus-5-5",
    "gemini:gemini-3.8-flash",
    "deepseek:deepseek-flash",
];

/// Handles a line typed into `agent`: a command now, or a prompt into its queue.
pub fn submit(world: &mut World, agent: Entity, text: &str) {
    let text = text.trim();
    let (command, argument) = text.split_once(' ').unwrap_or((text, ""));
    let argument = argument.trim();
    match command {
        "" => {}
        "/quit" => {
            world.write_message(AppExit::Success);
        }
        "/help" => {
            let mut help = String::from(
                "Enter sends (queued while busy) · Esc cancels · Ctrl+C clears · Ctrl+D quits · \
                 Ctrl+P next model · Ctrl+O expands tool output\n\
                 /model [vendor:model] · /reload · /new · /quit",
            );
            for (name, line) in &world.resource::<SlashCommands>().0 {
                help.push_str(&format!("\n/{name} {line}"));
            }
            log(world, agent, Entry::Notice(help));
        }
        "/model" if argument.is_empty() => {
            let current = world.get::<Model>(agent).map(|model| model.spec().to_owned());
            log(
                world,
                agent,
                Entry::Notice(format!(
                    "Model: {}. Switch with /model <vendor:model>, for example:\n{}",
                    current.unwrap_or_default(),
                    MODELS.join("\n")
                )),
            );
        }
        "/model" => switch_model(world, agent, argument),
        "/new" => {
            crate::glue::cancel(world, agent);
            let mut entity = world.entity_mut(agent);
            entity.insert((
                SessionId(new_session_id()),
                Conversation::default(),
                Transcript(vec![Entry::Notice("New session.".into())]),
                Prompts::default(),
            ));
        }
        _ if command.starts_with('/')
            && command != "/reload"
            && world
                .resource::<SlashCommands>()
                .0
                .contains_key(&command[1..]) =>
        {
            world.write_message(SlashCommand {
                agent,
                name: command[1..].to_owned(),
                argument: argument.to_owned(),
            });
        }
        _ if command.starts_with('/') && command != "/reload" => {
            log(
                world,
                agent,
                Entry::Error(format!("Unknown command {command}. /help lists them.")),
            );
        }
        _ => {
            if let Some(mut prompts) = world.get_mut::<Prompts>(agent) {
                prompts.queue.push_back(text.to_owned());
            }
        }
    }
}

/// Points an agent at another model; the conversation stays.
pub fn switch_model(world: &mut World, agent: Entity, spec: &str) {
    let model = world.resource::<ModelFactory>().model(spec);
    let entry = match model.error() {
        Some(error) => Entry::Error(format!("Cannot use {spec}: {error}")),
        None => Entry::Notice(format!("Model: {spec}")),
    };
    if model.error().is_none() {
        world.entity_mut(agent).insert(model);
    }
    log(world, agent, entry);
}

pub fn log(world: &mut World, agent: Entity, entry: Entry) {
    if let Some(mut transcript) = world.get_mut::<Transcript>(agent) {
        transcript.log(entry);
    }
}

/// Starts each idle agent's next prompt, or its queued `/reload`.
fn start_turns(
    mut agents: Query<(&mut Turn, &mut Prompts, &mut Conversation, &mut Transcript), With<Agent>>,
    mut reload: ResMut<Reload>,
) {
    for (mut turn, mut prompts, mut conversation, mut transcript) in &mut agents {
        if *turn != Turn::Idle || reload.building() {
            continue;
        }
        let Some(prompt) = prompts.queue.pop_front() else {
            continue;
        };
        if prompt == "/reload" {
            transcript.log(Entry::Notice("Building…".into()));
            reload.start(None);
            continue;
        }
        transcript.log(Entry::User(prompt.clone()));
        let notes = std::mem::take(&mut prompts.notes);
        let text = if notes.is_empty() {
            prompt
        } else {
            format!("[agent notes]\n{}\n\n{prompt}", notes.join("\n"))
        };
        conversation.0.push(Message::user(text));
        *turn = Turn::Request;
    }
}

fn record_transcript(
    mut events: MessageReader<AgentEvent>,
    mut agents: Query<&mut Transcript>,
    mut usage: Query<&mut Usage>,
) {
    for event in events.read() {
        match event {
            AgentEvent::Text { agent, text } => {
                let Ok(mut transcript) = agents.get_mut(*agent) else {
                    continue;
                };
                // Streamed text grows the reply's assistant entry.
                match transcript.0.last_mut() {
                    Some(Entry::Assistant(reply)) => reply.push_str(text),
                    _ => transcript.log(Entry::Assistant(text.clone())),
                }
            }
            AgentEvent::Replied { agent, response } => {
                if let Ok(mut usage) = usage.get_mut(*agent) {
                    usage.input += response.usage.input_tokens.unwrap_or_default();
                    usage.output += response.usage.output_tokens.unwrap_or_default();
                }
                let Ok(mut transcript) = agents.get_mut(*agent) else {
                    continue;
                };
                let text = response.text();
                match transcript.0.last_mut() {
                    Some(Entry::Assistant(reply)) if !text.is_empty() => *reply = text,
                    _ if !text.is_empty() => transcript.log(Entry::Assistant(text)),
                    _ => {}
                }
            }
            AgentEvent::CallStarted { agent, call } => {
                if let Ok(mut transcript) = agents.get_mut(*agent) {
                    transcript.log(Entry::Call(format!(
                        "{} {}",
                        call.function.name, call.function.arguments
                    )));
                }
            }
            AgentEvent::CallFinished { agent, output, .. } => {
                if let Ok(mut transcript) = agents.get_mut(*agent) {
                    transcript.log(Entry::Output(output.clone()));
                }
            }
            AgentEvent::Ended { agent, error } => {
                if let (Ok(mut transcript), Some(error)) = (agents.get_mut(*agent), error) {
                    transcript.log(Entry::Error(error.clone()));
                }
            }
        }
    }
}

/// Where sessions are saved: conversations through Rig's file memory, the
/// rest of each agent next to them.
#[derive(Resource)]
pub struct SessionStore {
    pub memory: FileConversationMemory,
}

/// Everything about an agent that is saved besides its conversation.
#[derive(Serialize, Deserialize)]
pub struct Saved {
    pub origin: Origin,
    pub model: String,
    pub transcript: Transcript,
    pub prompts: Prompts,
    pub turn: Turn,
}

impl SessionStore {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            memory: FileConversationMemory::new(dir),
        }
    }

    fn meta(&self, id: &str) -> PathBuf {
        self.memory
            .path(&ConversationId::new(id))
            .with_extension("agent.json")
    }

    /// Saves an agent whole: its conversation replaces the stored one.
    pub fn save(&self, id: &str, conversation: &[Message], saved: &Saved) -> std::io::Result<()> {
        block_on(
            self.memory
                .replace(&ConversationId::new(id), conversation.to_vec()),
        )
        .map_err(std::io::Error::other)?;
        self.save_meta(id, saved)
    }

    /// Saves what is not the conversation.
    pub fn save_meta(&self, id: &str, saved: &Saved) -> std::io::Result<()> {
        let path = self.meta(id);
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(saved)?)?;
        std::fs::rename(tmp, path)
    }

    pub fn load(&self, id: &str) -> Option<(Vec<Message>, Saved)> {
        let saved = serde_json::from_slice(&std::fs::read(self.meta(id)).ok()?).ok()?;
        let conversation = block_on(self.memory.load(&ConversationId::new(id))).ok()?;
        Some((conversation, saved))
    }

    pub fn remove(&self, id: &str) {
        let _ = block_on(self.memory.clear(&ConversationId::new(id)));
        let _ = std::fs::remove_file(self.meta(id));
    }

    /// Stored session ids, most recent first.
    pub fn list(&self) -> Vec<String> {
        self.memory
            .conversations()
            .unwrap_or_default()
            .into_iter()
            .map(ConversationId::into_string)
            .filter(|id| self.meta(id).exists())
            .collect()
    }
}

/// Saves one agent entity.
pub fn save_agent(world: &World, agent: Entity) -> std::io::Result<()> {
    let entity = world.entity(agent);
    let (Some(id), Some(conversation)) = (entity.get::<SessionId>(), entity.get::<Conversation>())
    else {
        return Ok(());
    };
    let saved = saved(world, agent);
    world
        .resource::<SessionStore>()
        .save(&id.0, &conversation.0, &saved)
}

pub fn saved(world: &World, agent: Entity) -> Saved {
    let entity = world.entity(agent);
    Saved {
        origin: entity.get::<Origin>().copied().unwrap_or(Origin::Tui),
        model: entity
            .get::<Model>()
            .map(|model| model.spec().to_owned())
            .unwrap_or_default(),
        transcript: entity.get::<Transcript>().cloned().unwrap_or_default(),
        prompts: entity.get::<Prompts>().cloned().unwrap_or_default(),
        turn: entity.get::<Turn>().cloned().unwrap_or_default(),
    }
}

/// Spawns an agent from a stored session.
pub fn restore_agent(world: &mut World, id: &str) -> Option<Entity> {
    let (conversation, saved) = world.resource::<SessionStore>().load(id)?;
    let agent = spawn_agent(world, id.to_owned(), saved.origin, &saved.model);
    world.entity_mut(agent).insert((
        Conversation(conversation),
        saved.transcript,
        saved.prompts,
        saved.turn,
    ));
    Some(agent)
}
