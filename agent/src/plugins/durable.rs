//! Durable sessions: sessions outlive the process, not just a reload.
//!
//! With this plugin the last TUI session and every remote session come back
//! after a quit (`--new` starts fresh), `/resume` lists saved sessions and
//! reopens one in the TUI, and requests are compacted: rig-memory's
//! `CompactingMemory` keeps a token window of recent messages verbatim and
//! folds older ones into a summary, while the file keeps everything.

use std::sync::Arc;

use bevy::prelude::*;
use rig_core::id::ConversationId;
use rig_memory::{CompactingMemory, HeuristicTokenCounter, TemplateCompactor, TokenWindowMemory};

use crate::Env;
use crate::glue::{Agent, Session, SessionMemory};
use crate::session::{EntryKind, Focused, ResumeOnLaunch, Store, Transcript, resume_saved, saved_sessions};
use crate::tui::{SlashCommand, SlashCommands};

/// Messages kept verbatim in a request, by estimated tokens.
const WINDOW_TOKENS: usize = 120_000;

pub struct DurablePlugin;

impl Plugin for DurablePlugin {
    fn build(&self, app: &mut App) {
        let store = app.world().resource::<Store>().0.clone();
        let compacting = CompactingMemory::new(
            store,
            TokenWindowMemory::new(WINDOW_TOKENS, HeuristicTokenCounter::default()),
            TemplateCompactor::new(),
        );
        app.insert_resource(SessionMemory(Arc::new(compacting)))
            .insert_resource(ResumeOnLaunch);
        app.world_mut()
            .get_resource_or_init::<SlashCommands>()
            .0
            .insert("resume".into(), "list saved sessions, or reopen one: /resume <n>".into());
        app.add_systems(Update, resume);
    }
}

fn resume(mut requests: MessageReader<SlashCommand>, mut commands: Commands) {
    for request in requests.read().filter(|command| command.name == "resume").cloned() {
        commands.queue(move |world: &mut World| reopen(world, request));
    }
}

fn reopen(world: &mut World, request: SlashCommand) {
    let env = world.resource::<Env>().clone();
    let running: Vec<ConversationId> = world
        .query_filtered::<&Session, With<Agent>>()
        .iter(world)
        .map(|session| session.0.clone())
        .collect();
    let saved: Vec<_> = saved_sessions(&env)
        .into_iter()
        .filter(|(id, ..)| !running.contains(id))
        .collect();
    let chosen = request.args.parse::<usize>().ok().and_then(|n| saved.get(n.wrapping_sub(1)));
    let Some((id, ..)) = chosen else {
        let list: Vec<String> = saved
            .iter()
            .enumerate()
            .map(|(i, (id, kind, name, model))| format!("{}. {name} {kind:?} {id} {model}", i + 1))
            .collect();
        if let Some(mut transcript) = world.get_mut::<Transcript>(request.agent) {
            transcript.log(EntryKind::Info, format!("saved sessions:\n{}\n/resume <n>", list.join("\n")));
        }
        return;
    };
    if let Some(entity) = resume_saved(world, id) {
        world.entity_mut(request.agent).remove::<Focused>();
        world.entity_mut(entity).insert(Focused);
    }
}
