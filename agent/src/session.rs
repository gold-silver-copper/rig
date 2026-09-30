//! The session: everything that must survive a reload, saved to
//! `<state>/session.json` whenever it changes.

use std::collections::VecDeque;
use std::path::Path;

use bevy::prelude::*;
use rig_core::completion::{Message, ToolDefinition};
use rig_core::message::{ToolCall, ToolResult};
use rig_core::providers::registry::ProviderRef;
use serde::{Deserialize, Serialize};

use crate::{Env, presets};

#[derive(Resource, Serialize, Deserialize)]
pub struct Session {
    /// The model every request goes to; `/model` changes it mid-conversation.
    pub model: ProviderRef,
    /// The conversation sent to the model.
    pub messages: Vec<Message>,
    /// What the TUI shows.
    pub transcript: Vec<Entry>,
    /// Prompts waiting for the current turn to finish.
    pub queue: VecDeque<String>,
    pub phase: Phase,
    /// Tools external processes registered over BRP, kept so the model can
    /// still call them while their plugin reconnects after a reload.
    pub remote_tools: Vec<RemoteTool>,
}

/// Where the current turn is. Persisted so a restart resumes it.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub enum Phase {
    #[default]
    Idle,
    /// Waiting on the model. A restart sends the request again.
    Model,
    /// Running the model's tool calls one at a time. A restart continues at
    /// `next`; if that is the `reload` call, the restart is its result.
    Tools {
        calls: Vec<ToolCall>,
        next: usize,
        results: Vec<ToolResult>,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RemoteTool {
    pub plugin: String,
    pub def: ToolDefinition,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Entry {
    pub kind: Kind,
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    User,
    Assistant,
    Call,
    Result,
    Info,
    Error,
}

impl Session {
    fn new(model: ProviderRef) -> Self {
        Self {
            model,
            messages: Vec::new(),
            transcript: Vec::new(),
            queue: VecDeque::new(),
            phase: Phase::Idle,
            remote_tools: Vec::new(),
        }
    }

    pub fn log(&mut self, kind: Kind, text: impl Into<String>) {
        self.transcript.push(Entry {
            kind,
            text: text.into(),
        });
    }

    pub fn busy(&self) -> bool {
        !matches!(self.phase, Phase::Idle)
    }

    /// Switch to `reference` (`vendor[/format]:model`, or a preset number).
    pub fn switch_model(&mut self, reference: &str) -> Result<(), String> {
        let presets = presets();
        let reference = match reference.parse::<usize>() {
            Ok(n) if (1..=presets.len()).contains(&n) => presets[n - 1].as_str(),
            _ => reference,
        };
        let model = ProviderRef::parse(reference).map_err(|error| error.to_string())?;
        self.log(Kind::Info, format!("model: {model}"));
        self.model = model;
        Ok(())
    }

    fn load(path: &Path) -> Result<Option<Self>, String> {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|error| format!("{}: {error}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("{}: {error}", path.display())),
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(tmp, path)
    }
}

pub struct SessionPlugin {
    /// Model chosen on the command line; overrides the saved one.
    pub model: Option<String>,
}

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        let env = app.world().resource::<Env>().clone();
        let _ = std::fs::create_dir_all(&env.state);
        let path = env.state.join("session.json");
        let default_model = || ProviderRef::parse(&presets()[0]).map_err(|e| e.to_string());
        let mut session = match Session::load(&path) {
            Ok(Some(session)) => session,
            Ok(None) => Session::new(default_model().unwrap_or_else(|e| panic!("{e}"))),
            Err(error) => {
                let _ = std::fs::rename(&path, path.with_extension("json.bad"));
                let mut session = Session::new(default_model().unwrap_or_else(|e| panic!("{e}")));
                session.log(Kind::Error, format!("could not read the session: {error}"));
                session
            }
        };
        if let Some(model) = &self.model
            && let Err(error) = session.switch_model(model)
        {
            session.log(Kind::Error, error);
        }
        app.insert_resource(session)
            .add_systems(Last, save.run_if(resource_changed::<Session>));
    }
}

fn save(session: Res<Session>, env: Res<Env>) {
    if let Err(error) = session.save(&env.state.join("session.json")) {
        error!("saving the session: {error}");
    }
}
