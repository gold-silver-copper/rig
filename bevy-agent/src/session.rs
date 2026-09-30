//! Everything that survives a reload: the conversation, the transcript, the
//! prompt queue, the model choice, the BRP port and external tool definitions.
//! It lives in one JSON file in the state directory, rewritten when it changes.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use rig_core::completion::ToolDefinition;
use rig_core::message::{Message, ToolCall, ToolResult};
use serde::{Deserialize, Serialize};

#[derive(Resource, Serialize, Deserialize, Default)]
pub struct Session {
    /// `vendor[/format]:model`, as `rig_core::providers::registry::ProviderRef` parses it.
    pub model: String,
    pub brp_port: u16,
    pub messages: Vec<Message>,
    pub transcript: Vec<Entry>,
    /// Prompts and queued `/reload` commands waiting for the agent to be idle.
    pub queue: VecDeque<String>,
    pub turn: Turn,
    /// Tools registered by external BRP plugins, kept so they stay callable
    /// while their plugin reconnects after a reload.
    pub external_tools: Vec<ToolDefinition>,
    /// Things the model should hear about with the next prompt, such as a
    /// user-triggered reload that fell back to the previous binary.
    pub notes: Vec<String>,
}

#[derive(Serialize, Deserialize, Default, Clone, PartialEq)]
pub enum Turn {
    #[default]
    Idle,
    /// Waiting on the model. A restart in this state sends the request again.
    Request,
    /// Running the tool calls of the last assistant message, in order.
    Tools {
        calls: Vec<ToolCall>,
        results: Vec<ToolResult>,
    },
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Entry {
    User(String),
    Assistant(String),
    Call(String),
    Output(String),
    Notice(String),
    Error(String),
}

impl Session {
    pub fn log(&mut self, entry: Entry) {
        self.transcript.push(entry);
    }

    pub fn busy(&self) -> bool {
        self.turn != Turn::Idle
    }
}

/// Where the session and the supervisor's files live.
#[derive(Resource, Clone)]
pub struct StateDir(pub PathBuf);

impl StateDir {
    pub fn session(&self) -> PathBuf {
        self.0.join("session.json")
    }
}

pub fn load(path: &Path) -> Option<Session> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn save(session: &Session, path: &Path) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(session)?)?;
    std::fs::rename(tmp, path)
}

/// Writes the session at most once a second while it keeps changing, so a
/// crash loses little.
pub fn autosave(
    session: Res<Session>,
    dir: Res<StateDir>,
    mut dirty: Local<bool>,
    mut last: Local<Option<Instant>>,
) {
    *dirty |= session.is_changed();
    if !*dirty || last.is_some_and(|at| at.elapsed() < Duration::from_secs(1)) {
        return;
    }
    *dirty = false;
    *last = Some(Instant::now());
    if let Err(error) = save(&session, &dir.session()) {
        eprintln!("saving the session failed: {error}");
    }
}

/// Writes the session in the frame the app decides to exit, reload included.
pub fn save_on_exit(mut exits: MessageReader<AppExit>, session: Res<Session>, dir: Res<StateDir>) {
    if exits.read().next().is_some()
        && let Err(error) = save(&session, &dir.session())
    {
        eprintln!("saving the session failed: {error}");
    }
}
