//! Self-reload. `cargo build` runs on the agent's own source; if it fails
//! the errors go back to whoever asked and nothing restarts. If it succeeds
//! the agent saves its session and exits with [`RELOAD_CODE`], and the
//! supervisor starts the new binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use async_channel::{Receiver, Sender};
use bevy::prelude::*;
use serde_json::{Value, json};

use crate::agent::{AddTool, ToolReply};
use crate::session::{Entry, Session, StateDir};
use crate::supervisor::{self, RELOAD_CODE};

pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Reload>()
            .add_tool(
                "reload",
                "Rebuild yourself from your source and restart into the new binary, keeping \
                 this conversation. Returns the compiler errors if the build fails.",
                json!({"type": "object", "properties": {}}),
                reload_tool,
            )
            .add_systems(Update, finish);
    }
}

#[derive(Resource, Default)]
pub struct Reload {
    build: Option<Receiver<Result<PathBuf, String>>>,
    /// Where the `reload` tool waits for its result; `None` for `/reload`.
    caller: Option<Sender<String>>,
}

impl Reload {
    pub fn building(&self) -> bool {
        self.build.is_some()
    }

    pub fn start(&mut self, caller: Option<Sender<String>>, state: &Path) {
        let (sender, receiver) = async_channel::bounded(1);
        let state = state.to_owned();
        std::thread::spawn(move || {
            let _ = sender.send_blocking(build(&state));
        });
        self.build = Some(receiver);
        self.caller = caller;
    }
}

fn reload_tool(In(_): In<Value>, mut reload: ResMut<Reload>, state: Res<StateDir>) -> ToolReply {
    if reload.building() {
        return ToolReply::Done("error: a reload is already building".into());
    }
    let (sender, receiver) = async_channel::bounded(1);
    reload.start(Some(sender), &state.0);
    ToolReply::Pending(receiver)
}

fn finish(
    mut reload: ResMut<Reload>,
    mut session: ResMut<Session>,
    state: Res<StateDir>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(Ok(result)) = reload.build.as_ref().map(Receiver::try_recv) else {
        return;
    };
    reload.build = None;
    if reload.caller.as_ref().is_some_and(Sender::is_closed) {
        reload.caller = None;
        session.log(Entry::Notice("Reload cancelled.".into()));
        return;
    }
    let failure = match result {
        Ok(binary) => {
            match std::fs::write(
                state.0.join("next-binary"),
                binary.to_string_lossy().as_bytes(),
            ) {
                Ok(()) => {
                    // The `reload` call stays unanswered: the new process answers it.
                    session.log(Entry::Notice("Build succeeded. Restarting…".into()));
                    exit.write(AppExit::from_code(RELOAD_CODE));
                    return;
                }
                Err(error) => format!("cannot hand the new binary over: {error}"),
            }
        }
        Err(errors) => errors,
    };
    let message = format!("Reload failed; still running the previous binary.\n{failure}");
    match reload.caller.take() {
        Some(caller) => {
            let _ = caller.try_send(message);
        }
        None => session.log(Entry::Error(message)),
    }
}

/// Builds the agent's source and returns a private copy of the new binary,
/// or the compiler errors.
fn build(state: &Path) -> Result<PathBuf, String> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let mut command = Command::new("cargo");
    command
        .arg("build")
        .arg("--manifest-path")
        .arg(&manifest)
        .args(["--bin", env!("CARGO_BIN_NAME")])
        .arg("--message-format=json-render-diagnostics")
        .env("CARGO_TERM_COLOR", "never")
        .stdin(Stdio::null());
    if !cfg!(debug_assertions) {
        command.arg("--release");
    }
    let output = command
        .output()
        .map_err(|error| format!("cannot run cargo: {error}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let binary = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|message| message["reason"] == "compiler-artifact")
        .filter(|message| message["target"]["name"] == env!("CARGO_BIN_NAME"))
        .find_map(|message| message["executable"].as_str().map(PathBuf::from));
    match binary {
        Some(binary) if output.status.success() => supervisor::install(&binary, state)
            .map_err(|error| format!("cannot copy the new binary: {error}")),
        _ => Err(errors(&stderr)),
    }
}

/// The error diagnostics in cargo's rendered output, or its tail when it has none.
fn errors(stderr: &str) -> String {
    let stderr: String = stderr
        .lines()
        .filter(|line| !line.starts_with("   Compiling ") && !line.starts_with("    Blocking "))
        .map(|line| format!("{line}\n"))
        .collect();
    let blocks: Vec<&str> = stderr
        .split("\n\n")
        .filter(|block| block.trim_start().starts_with("error"))
        .collect();
    let text = if blocks.is_empty() {
        stderr.clone()
    } else {
        blocks.join("\n\n")
    };
    let mut start = text.len().saturating_sub(20_000);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_owned()
}
