//! Everything that survives a reload: the conversation, the transcript, queued
//! prompts, the model choice and the BRP port, written to one JSON file.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use rig_core::completion::{Message, ToolDefinition};
use rig_core::message::{ToolCall, UserContent};
use rig_core::providers::registry::{ProviderId, ProviderRef};
use serde::{Deserialize, Serialize};

/// Where the agent keeps its state, binaries and logs, and where its source is.
#[derive(Resource, Clone, Debug)]
pub struct Paths {
    pub home: PathBuf,
    pub source: PathBuf,
}

impl Paths {
    pub fn from_env() -> Self {
        let home = std::env::var_os("BEVY_AGENT_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".bevy-agent"));
        let home = std::path::absolute(&home).unwrap_or(home);
        let source = std::env::var_os("BEVY_AGENT_SRC")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        Self { home, source }
    }
    pub fn session(&self) -> PathBuf {
        self.home.join("session.json")
    }
    pub fn bin(&self, name: &str) -> PathBuf {
        self.home.join("bin").join(name)
    }
    pub fn fallback_note(&self) -> PathBuf {
        self.home.join("fallback.txt")
    }
    pub fn child_stderr(&self) -> PathBuf {
        self.home.join("stderr.log")
    }
}

/// One line of the visible transcript.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Entry {
    User(String),
    Assistant(String),
    ToolCall(String),
    ToolResult(String),
    Info(String),
    Error(String),
}

/// A tool call of the current turn and, once it finished, its output.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingCall {
    pub call: ToolCall,
    pub result: Option<String>,
}

/// A tool an external process registered over BRP.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemoteTool {
    pub plugin: String,
    pub def: ToolDefinition,
}

#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub model: ProviderRef,
    pub port: u16,
    pub history: Vec<Message>,
    pub transcript: Vec<Entry>,
    pub queue: VecDeque<String>,
    pub remote_tools: Vec<RemoteTool>,
    /// `Some` while a turn is in flight. Empty calls: the model is due next.
    pub turn: Option<Vec<PendingCall>>,
}

impl Session {
    pub fn new(model: ProviderRef, port: u16) -> Self {
        Self {
            model,
            port,
            history: Vec::new(),
            transcript: Vec::new(),
            queue: VecDeque::new(),
            remote_tools: Vec::new(),
            turn: None,
        }
    }

    pub fn load(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Write atomically, so a crash mid-write never loses the conversation.
    pub fn save(&self, path: &Path) {
        let Ok(text) = serde_json::to_string(self) else {
            return;
        };
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }

    /// Append user content, merging into a trailing user message so a failed
    /// turn never leaves two user messages in a row.
    pub fn push_user(&mut self, content: Vec<UserContent>) {
        if let Some(Message::User { content: last }) = self.history.last_mut() {
            last.extend(content);
        } else {
            self.history.push(Message::User { content });
        }
    }

    pub fn info(&mut self, text: impl Into<String>) {
        self.transcript.push(Entry::Info(text.into()));
    }

    pub fn error(&mut self, text: impl Into<String>) {
        self.transcript.push(Entry::Error(text.into()));
    }
}

/// Models offered by `/model`; any other `vendor:model` reference works too.
pub const PRESETS: &[&str] = &[
    "openai:gpt-6.1-sol",
    "anthropic:claude-opus-5-5",
    "gcp.gemini:gemini-3.8-flash",
    "deepseek:deepseek-flash",
];

/// A preset number (1-based), a preset's model name, or a `vendor:model` reference.
pub fn parse_model(text: &str) -> Result<ProviderRef, String> {
    let text = text.trim();
    let preset = text
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_sub(1))
        .and_then(|n| PRESETS.get(n))
        .or_else(|| PRESETS.iter().find(|p| p.ends_with(&format!(":{text}"))));
    let text = preset.copied().unwrap_or(text);
    ProviderRef::parse(text).or_else(|error| {
        // Accept `gemini:…` for `gcp.gemini:…`: a unique dotted vendor suffix.
        let (vendor, model) = text.split_once(':').ok_or_else(|| error.to_string())?;
        let suffix = format!(".{vendor}");
        let mut vendors: Vec<&str> = ProviderId::all().map(|id| id.vendor()).filter(|v| v.ends_with(&suffix)).collect();
        vendors.dedup();
        match vendors[..] {
            [vendor] => ProviderRef::parse(&format!("{vendor}:{model}")).map_err(|e| e.to_string()),
            _ => Err(error.to_string()),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::parse_model;

    #[test]
    fn presets_names_and_references_parse() {
        for (text, expected) in [
            ("1", "openai/openai:gpt-6.1-sol"),
            ("claude-opus-5-5", "anthropic/anthropic:claude-opus-5-5"),
            ("gemini:gemini-3.8-flash", "gcp.gemini/gemini:gemini-3.8-flash"),
            ("deepseek:deepseek-flash", "deepseek/openai:deepseek-flash"),
        ] {
            assert_eq!(parse_model(text).map(|r| r.to_string()), Ok(expected.to_owned()));
        }
        assert!(parse_model("nosuch:model").is_err());
    }
}
