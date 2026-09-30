//! Everything that survives a reload, and where it is kept.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use anyhow::Context;
use bevy::prelude::Resource;
use rig_core::message::{Message, ToolCall};
use serde::{Deserialize, Serialize};

use crate::Options;

/// Files in the state directory.
#[derive(Clone, Debug)]
pub struct Paths {
    pub dir: PathBuf,
}

impl Paths {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
    /// The saved conversation.
    pub fn session(&self) -> PathBuf {
        self.dir.join("session.json")
    }
    /// Written by a child once its TUI and BRP server are up.
    pub fn ready(&self) -> PathBuf {
        self.dir.join("ready")
    }
    /// Written by a child that exits to reload: which binary to start next.
    pub fn reload(&self) -> PathBuf {
        self.dir.join("reload.json")
    }
    /// Copies of every binary built or started, so a later build cannot
    /// overwrite the last good one.
    pub fn bin(&self) -> PathBuf {
        self.dir.join("bin")
    }
    /// Children's stderr.
    pub fn log(&self) -> PathBuf {
        self.dir.join("agent.log")
    }
}

/// One line of the transcript shown in the TUI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub kind: EntryKind,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryKind {
    User,
    Assistant,
    ToolCall,
    ToolResult,
    Info,
    Error,
}

/// A tool call of the batch in flight, with its output once it has one.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CallState {
    pub call: ToolCall,
    pub output: Option<String>,
}

/// A tool an external process registered over BRP.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalTool {
    pub plugin: String,
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// The whole agent state that a reload carries over. Missing fields default,
/// so a binary can read a session written by an older or newer build.
#[derive(Resource, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Session {
    /// The selected model, as an alias or Rig provider reference.
    pub model: String,
    pub messages: Vec<Message>,
    pub transcript: Vec<Entry>,
    pub queue: VecDeque<String>,
    /// Tool calls of an unfinished turn.
    pub batch: Vec<CallState>,
    /// The call id of the `reload` tool call awaiting the reload's outcome.
    pub reload_call: Option<String>,
    pub external_tools: Vec<ExternalTool>,
    /// What the model should be told with the next prompt.
    pub notes: Vec<String>,
}

impl Session {
    /// The saved session when resuming; otherwise a fresh one that keeps the
    /// previously selected model unless `--model` names another.
    pub fn open(options: &Options) -> anyhow::Result<Self> {
        let paths = Paths::new(options.state_dir());
        let saved = Self::load(&paths.session());
        if options.resume {
            let mut session = saved.context("no saved session to continue")?;
            if let Some(model) = &options.model {
                session.model = model.clone();
            }
            return Ok(session);
        }
        let model = options
            .model
            .clone()
            .or_else(|| saved.ok().map(|session| session.model))
            .unwrap_or_else(|| crate::agent::DEFAULT_MODEL.to_string());
        Ok(Self {
            model,
            ..Self::default()
        })
    }

    fn load(path: &Path) -> anyhow::Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// Write atomically, so a crash mid-write leaves the previous session.
    pub fn save(&self, paths: &Paths) -> anyhow::Result<()> {
        std::fs::create_dir_all(&paths.dir)?;
        let tmp = paths.dir.join("session.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&tmp, paths.session())?;
        Ok(())
    }
}
