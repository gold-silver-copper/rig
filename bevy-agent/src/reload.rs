//! Self-reload. `cargo build` runs on the agent's own source; if it fails
//! the errors go back to whoever asked and nothing restarts. If it succeeds
//! every session is saved and the process exits with [`RELOAD_CODE`], and
//! the supervisor starts the new binary, which restores the sessions and
//! answers the `reload` call that was waiting.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use async_channel::Receiver;
use bevy::prelude::*;
use rig_core::completion::ToolDefinition;
use serde_json::{Value, json};

use crate::glue::{AddTool, CallOf, ToolCall, ToolOutput};
use crate::session::{Entry, Origin, Transcript};
use crate::supervisor::{self, RELOAD_CODE};
use crate::{State, save_everything};

pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        let state = app.world().resource::<State>().dir.clone();
        app.insert_resource(Reload {
            build: None,
            caller: None,
            state,
        })
        .add_system_tool(ToolDefinition {
            name: "reload".into(),
            description: "Rebuild yourself from your source and restart into the new binary, \
                          keeping every conversation. Returns the compiler errors if the build \
                          fails, and what happened if the new binary does not start."
                .into(),
            parameters: json!({"type": "object", "properties": {}}),
        })
        .add_systems(Update, (reload_calls, finish));
    }
}

#[derive(Resource)]
pub struct Reload {
    build: Option<Receiver<Result<PathBuf, String>>>,
    /// The `reload` call waiting for the build; `None` for `/reload`.
    caller: Option<Entity>,
    state: PathBuf,
}

impl Reload {
    pub fn building(&self) -> bool {
        self.build.is_some()
    }

    /// Starts a build for `caller`, a `reload` call entity, or for `/reload`.
    pub fn start(&mut self, caller: Option<Entity>) {
        let (sender, receiver) = async_channel::bounded(1);
        let state = self.state.clone();
        std::thread::spawn(move || {
            let _ = sender.send_blocking(build(&state));
        });
        self.build = Some(receiver);
        self.caller = caller;
    }
}

fn reload_calls(
    mut commands: Commands,
    calls: Query<(Entity, &ToolCall), Added<ToolCall>>,
    mut reload: ResMut<Reload>,
) {
    for (entity, ToolCall(call)) in &calls {
        if call.function.name != "reload" {
            continue;
        }
        if reload.building() {
            commands
                .entity(entity)
                .insert(ToolOutput("error: a reload is already building".into()));
        } else {
            reload.start(Some(entity));
        }
    }
}

fn finish(world: &mut World) {
    let result = {
        let mut reload = world.resource_mut::<Reload>();
        let Some(Ok(result)) = reload.build.as_ref().map(Receiver::try_recv) else {
            return;
        };
        reload.build = None;
        result
    };
    let caller = world.resource_mut::<Reload>().caller.take();
    let caller_agent = caller.and_then(|call| world.get::<CallOf>(call).map(|of| of.0));
    if caller.is_some() && caller_agent.is_none() {
        notify(world, None, Entry::Notice("Reload cancelled.".into()));
        return;
    }
    let failure = match result {
        Ok(binary) => {
            let state = world.resource::<Reload>().state.clone();
            notify(world, caller_agent, Entry::Notice("Build succeeded. Restarting…".into()));
            // The `reload` call stays unanswered: the new process answers it.
            match save_everything(world).and_then(|()| {
                std::fs::write(state.join("next-binary"), binary.to_string_lossy().as_bytes())
            }) {
                Ok(()) => {
                    world.write_message(AppExit::from_code(RELOAD_CODE));
                    return;
                }
                Err(error) => format!("cannot hand over to the new binary: {error}"),
            }
        }
        Err(errors) => errors,
    };
    let message = format!("Reload failed; still running the previous binary.\n{failure}");
    match caller {
        Some(call) => {
            world.entity_mut(call).insert(ToolOutput(message));
        }
        None => notify(world, None, Entry::Error(message)),
    }
}

/// Shows `entry` in `agent`'s transcript, or the TUI session's.
fn notify(world: &mut World, agent: Option<Entity>, entry: Entry) {
    let agent = agent.or_else(|| {
        world
            .query::<(Entity, &Origin)>()
            .iter(world)
            .find(|(_, origin)| **origin == Origin::Tui)
            .map(|(entity, _)| entity)
    });
    if let Some(mut transcript) = agent.and_then(|agent| world.get_mut::<Transcript>(agent)) {
        transcript.log(entry);
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
        .args(["--bin", env!("CARGO_PKG_NAME")])
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
        .filter(|message| message["target"]["name"] == env!("CARGO_PKG_NAME"))
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
