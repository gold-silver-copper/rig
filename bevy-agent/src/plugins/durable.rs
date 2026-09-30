//! Durable sessions. Every agent's session is saved as it changes, through
//! the same store reload uses (Rig's `FileConversationMemory`), and kept
//! after the agent quits. `--continue` resumes the latest TUI session,
//! `--resume <id>` a given one; `/sessions` lists them and `/resume <id>`
//! switches the TUI to one. A restored history is checked with
//! `rig_core::transcript` and tool calls left unanswered by a crash get a
//! result saying so. `/compact` and a token budget shrink a long history
//! with rig-memory's token window and template summary.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use futures::executor::block_on;
use rig_core::id::ConversationId;
use rig_core::memory::{Compactor, ConversationMemory};
use rig_core::transcript::{answer_unanswered, validate_canonical};
use rig_memory::{
    HeuristicTokenCounter, MemoryPolicy, TemplateCompactor, TokenCounter, TokenWindowMemory,
};

use crate::glue::{Agent, Conversation, Model, Turn};
use crate::session::{
    self, Entry, Origin, Prompts, SessionId, SessionStore, SlashCommand, SlashCommands, Transcript,
};
use crate::{KeepSessions, OpenSessions, ProcessState, State, save_everything};

/// Which session the TUI resumes on the first start.
#[derive(Clone, Debug)]
pub enum Resume {
    Latest,
    Id(String),
}

pub struct DurablePlugin {
    pub resume: Option<Resume>,
    /// Compact a session whose history is estimated above this many tokens.
    pub compact_at: usize,
}

impl Plugin for DurablePlugin {
    fn build(&self, app: &mut App) {
        let mut commands = app.world_mut().resource_mut::<SlashCommands>();
        commands
            .0
            .insert("sessions".into(), "lists saved sessions".into());
        commands
            .0
            .insert("resume".into(), "<id> switches to a saved session".into());
        commands
            .0
            .insert("compact".into(), "shrinks this session's history".into());
        app.insert_resource(KeepSessions)
            .insert_resource(Durable {
                resume: self.resume.clone(),
                compact_at: self.compact_at,
                last_save: None,
            })
            .add_systems(Startup, resume.before(OpenSessions))
            .add_systems(Update, (commands_, autosave).chain())
            .add_systems(Last, save_on_exit);
    }
}

#[derive(Resource)]
struct Durable {
    resume: Option<Resume>,
    compact_at: usize,
    last_save: Option<Instant>,
}

/// How much of an agent's conversation is already in the store.
#[derive(Component)]
struct Stored(usize);

fn resume(world: &mut World) {
    let state = world.resource::<State>().clone();
    let Some(target) = world.resource::<Durable>().resume.clone() else {
        return;
    };
    if state.generation > 0 || !state.primary {
        return;
    }
    let id = match target {
        Resume::Id(id) => Some(id),
        Resume::Latest => {
            let store = world.resource::<SessionStore>();
            store.list().into_iter().find(|id| {
                store
                    .load(id)
                    .is_some_and(|(_, saved)| saved.origin == Origin::Tui)
            })
        }
    };
    match id.and_then(|id| restore(world, &id)) {
        Some(agent) => {
            world.entity_mut(agent).insert(Origin::Tui);
        }
        None => eprintln!("no saved session to resume; starting a new one"),
    }
}

/// Restores a session, repairing a history a crash cut short.
fn restore(world: &mut World, id: &str) -> Option<Entity> {
    let agent = session::restore_agent(world, id)?;
    let mut conversation = world.get_mut::<Conversation>(agent)?;
    let mut notice = format!("Resumed session {id}.");
    if let Err(error) = validate_canonical(&conversation.0) {
        let repaired = answer_unanswered(&mut conversation.0, |_| {
            "Interrupted: the agent stopped before this tool finished.".into()
        });
        notice.push_str(&format!(" The saved history was not canonical ({error}); answered {repaired} interrupted tool call(s)."));
    }
    let length = conversation.0.len();
    // A saved turn in progress restarts from its conversation.
    if let Some(mut turn) = world.get_mut::<Turn>(agent)
        && matches!(*turn, Turn::Tools { .. })
    {
        *turn = Turn::Idle;
    }
    world.entity_mut(agent).insert(Stored(length));
    session::log(world, agent, Entry::Notice(notice));
    Some(agent)
}

fn commands_(mut reader: MessageReader<SlashCommand>, mut commands: Commands) {
    for command in reader.read().cloned() {
        commands.queue(move |world: &mut World| run_command(world, command));
    }
}

fn run_command(world: &mut World, command: SlashCommand) {
    let agent = command.agent;
    match command.name.as_str() {
        "sessions" => {
            let store = world.resource::<SessionStore>();
            let lines: Vec<String> = store
                .list()
                .into_iter()
                .take(20)
                .filter_map(|id| {
                    let (conversation, saved) = store.load(&id)?;
                    let first = saved.transcript.0.iter().find_map(|entry| match entry {
                        Entry::User(text) => Some(text.chars().take(60).collect::<String>()),
                        _ => None,
                    });
                    Some(format!(
                        "{id}  {:?}  {}  {} messages  {}",
                        saved.origin,
                        saved.model,
                        conversation.len(),
                        first.unwrap_or_default()
                    ))
                })
                .collect();
            session::log(
                world,
                agent,
                Entry::Notice(format!("Saved sessions:\n{}", lines.join("\n"))),
            );
        }
        "resume" => {
            let id = command.argument.trim();
            if world.resource::<SessionStore>().load(id).is_none() {
                session::log(
                    world,
                    agent,
                    Entry::Error(format!("No saved session {id}.")),
                );
                return;
            }
            let origin = world.get::<Origin>(agent).copied();
            let _ = session::save_agent(world, agent);
            crate::glue::cancel(world, agent);
            world.despawn(agent);
            if let Some(resumed) = restore(world, id)
                && let Some(origin) = origin
            {
                world.entity_mut(resumed).insert(origin);
            }
        }
        "compact" => compact(world, agent),
        _ => {}
    }
}

/// Replaces the history with a summary of its older part and a window of
/// its recent part.
fn compact(world: &mut World, agent: Entity) {
    let budget = world.resource::<Durable>().compact_at / 2;
    let Some(id) = world.get::<SessionId>(agent).map(|id| id.0.clone()) else {
        return;
    };
    let Some(conversation) = world.get::<Conversation>(agent).map(|c| c.0.clone()) else {
        return;
    };
    let policy = TokenWindowMemory::new(budget, HeuristicTokenCounter::default());
    let Ok((kept, demoted)) = policy.apply_with_demoted(conversation) else {
        return;
    };
    if demoted.is_empty() {
        session::log(world, agent, Entry::Notice("Nothing to compact.".into()));
        return;
    }
    let compactor = TemplateCompactor::new().with_max_bytes(16 * 1024);
    let Ok(summary) = block_on(compactor.compact(&ConversationId::new(&id), &demoted, None)) else {
        return;
    };
    let mut compacted = vec![rig_core::message::Message::from(summary)];
    compacted.extend(kept);
    let count = demoted.len();
    let store = world.resource::<SessionStore>();
    let _ = block_on(
        store
            .memory
            .replace(&ConversationId::new(&id), compacted.clone()),
    );
    let stored = compacted.len();
    world
        .entity_mut(agent)
        .insert((Conversation(compacted), Stored(stored)));
    session::log(
        world,
        agent,
        Entry::Notice(format!("Compacted: {count} older messages summarized.")),
    );
}

/// Saves changed sessions at most once a second, appending new messages to
/// the conversation file and rewriting the rest. Idle sessions over the
/// token budget are compacted first.
fn autosave(world: &mut World) {
    let now = Instant::now();
    if world
        .resource::<Durable>()
        .last_save
        .is_some_and(|at| now - at < Duration::from_secs(1))
    {
        return;
    }
    world.resource_mut::<Durable>().last_save = Some(now);
    let compact_at = world.resource::<Durable>().compact_at;
    let counter = HeuristicTokenCounter::default();
    let changed: Vec<(Entity, bool)> = world
        .query_filtered::<(Entity, &Conversation, &Turn), (
            With<Agent>,
            Or<(
                Changed<Conversation>,
                Changed<Transcript>,
                Changed<Turn>,
                Changed<Prompts>,
                Changed<Model>,
            )>,
        )>()
        .iter(world)
        .map(|(entity, conversation, turn)| {
            let tokens: usize = conversation
                .0
                .iter()
                .map(|message| counter.count(message))
                .sum();
            (entity, *turn == Turn::Idle && tokens > compact_at)
        })
        .collect();
    for (agent, over_budget) in changed {
        if over_budget {
            compact(world, agent);
        }
        save(world, agent);
    }
    let agents: Vec<String> = world
        .query_filtered::<&SessionId, With<Agent>>()
        .iter(world)
        .map(|id| id.0.clone())
        .collect();
    let dir = world.resource::<State>().dir.clone();
    let mut process = world.resource_mut::<ProcessState>();
    process.0.insert("agents".into(), serde_json::json!(agents));
    let _ = process.save(&dir);
}

fn save(world: &mut World, agent: Entity) {
    let entity = world.entity(agent);
    let (Some(id), Some(conversation)) = (entity.get::<SessionId>(), entity.get::<Conversation>())
    else {
        return;
    };
    let stored = entity.get::<Stored>().map(|stored| stored.0);
    let id = id.0.clone();
    let messages = conversation.0.clone();
    let saved = session::saved(world, agent);
    let store = world.resource::<SessionStore>();
    let conversation_id = ConversationId::new(&id);
    let result = match stored {
        Some(stored) if stored <= messages.len() => block_on(
            store
                .memory
                .append(&conversation_id, messages[stored..].to_vec()),
        ),
        _ => block_on(store.memory.replace(&conversation_id, messages.clone())),
    };
    let result = result
        .map_err(std::io::Error::other)
        .and_then(|()| store.save_meta(&id, &saved));
    match result {
        Ok(()) => {
            world.entity_mut(agent).insert(Stored(messages.len()));
        }
        Err(error) => eprintln!("saving session {id} failed: {error}"),
    }
}

fn save_on_exit(world: &mut World) {
    if world.resource::<Messages<AppExit>>().is_empty() {
        return;
    }
    let _ = save_everything(world);
}
