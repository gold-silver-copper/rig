//! Reloading: rebuild the agent's own crate, and only if that succeeds, save
//! the session and exit so the supervisor starts the new binary.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use bevy::prelude::*;
use crossbeam_channel::Receiver;
use serde_json::{Value, json};

use crate::agent::{
    AgentAppExt, CallOutput, Config, PendingCall, ToolInvocation, ToolTasks, Transcript, Turn,
    call_key, save_session,
};
use crate::session::EntryKind;
use crate::supervisor::{ReloadRequest, stash_binary};

pub struct ReloadPlugin;

/// `/reload` from the user.
#[derive(Message)]
pub struct ReloadCommand;

/// A `reload` call waiting for the rest of its batch, so that edits made in
/// the same response are on disk before the build starts.
#[derive(Resource)]
struct Queued(Entity);

/// A build in progress, for the `reload` call `call` or for the user.
#[derive(Resource)]
struct Building {
    call: Option<Entity>,
    result: Receiver<Result<PathBuf, String>>,
}

/// A successful build waiting to be restarted into: once the rest of the
/// tool batch (or the user's turn) is finished.
#[derive(Resource)]
pub struct RestartPending {
    /// The `reload` call to answer after the restart.
    pub call: Option<String>,
    entity: Option<Entity>,
    binary: PathBuf,
}

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<ReloadCommand>()
            .add_tool(
                "reload",
                "Rebuild your own source code and restart into the new binary. The \
                 conversation survives the restart and the turn continues. If the build fails, \
                 returns the compiler errors and keeps running the current binary.",
                json!({ "type": "object", "properties": {} }),
                reload_tool,
            )
            .add_systems(
                Update,
                (start_user_reload, start_queued, finish_build, restart_when_ready).chain(),
            );
    }
}

fn reload_tool(
    In(invocation): In<ToolInvocation>,
    building: Option<Res<Building>>,
    queued: Option<Res<Queued>>,
    mut commands: Commands,
) {
    if building.is_some() || queued.is_some() {
        commands
            .entity(invocation.entity)
            .insert(CallOutput("error: a reload is already in progress".into()));
        return;
    }
    commands.insert_resource(Queued(invocation.entity));
}

fn start_queued(
    queued: Option<Res<Queued>>,
    calls: Query<(Entity, Option<&CallOutput>), With<PendingCall>>,
    config: Res<Config>,
    tasks: Res<ToolTasks>,
    mut commands: Commands,
) {
    let Some(queued) = queued else { return };
    if calls
        .iter()
        .any(|(entity, output)| entity != queued.0 && output.is_none())
    {
        return;
    }
    commands.remove_resource::<Queued>();
    commands.insert_resource(start_build(&config, &tasks, Some(queued.0)));
}

fn start_user_reload(
    mut requests: MessageReader<ReloadCommand>,
    building: Option<Res<Building>>,
    config: Res<Config>,
    tasks: Res<ToolTasks>,
    mut transcript: ResMut<Transcript>,
    mut commands: Commands,
) {
    if requests.read().count() == 0 {
        return;
    }
    if building.is_some() {
        transcript.push(EntryKind::Info, "a reload is already building");
        return;
    }
    transcript.push(EntryKind::Info, "reload: building…");
    commands.insert_resource(start_build(&config, &tasks, None));
}

fn start_build(config: &Config, tasks: &ToolTasks, call: Option<Entity>) -> Building {
    let (sender, result) = crossbeam_channel::bounded(1);
    let source = config.source_dir.clone();
    let paths = config.paths.clone();
    tasks.handle().spawn(async move {
        let built = cargo_build(&source)
            .await
            .and_then(|binary| stash_binary(&paths, &binary).map_err(|e| format!("{e:#}")));
        let _ = sender.send(built);
    });
    Building { call, result }
}

fn finish_build(
    building: Option<Res<Building>>,
    calls: Query<&PendingCall>,
    mut transcript: ResMut<Transcript>,
    mut commands: Commands,
) {
    let Some(building) = building else { return };
    let Ok(result) = building.result.try_recv() else {
        return;
    };
    commands.remove_resource::<Building>();
    match (result, building.call) {
        (Ok(binary), call) => {
            transcript.push(EntryKind::Info, "reload: build succeeded, restarting…");
            commands.insert_resource(RestartPending {
                call: call.and_then(|entity| calls.get(entity).ok()).map(|p| call_key(&p.call)),
                entity: call,
                binary,
            });
        }
        (Err(errors), Some(entity)) => {
            commands.entity(entity).insert(CallOutput(format!(
                "Build failed, so nothing was reloaded: the current binary keeps running.\n{errors}"
            )));
        }
        (Err(errors), None) => {
            transcript.push(EntryKind::Error, format!("reload: build failed\n{errors}"));
        }
    }
}

/// Restart once nothing else is in flight: for a `reload` call, when the other
/// calls of its batch have their outputs; for `/reload`, between turns.
fn restart_when_ready(world: &mut World) {
    let Some(pending) = world.get_resource::<RestartPending>() else {
        return;
    };
    let (entity, binary) = (pending.entity, pending.binary.clone());
    let ready = match entity {
        Some(reload_call) => world
            .query_filtered::<(Entity, Option<&CallOutput>), With<PendingCall>>()
            .iter(world)
            .all(|(entity, output)| entity == reload_call || output.is_some()),
        None => world.resource::<Turn>().is_idle(),
    };
    if !ready {
        return;
    }
    let paths = world.resource::<Config>().paths.clone();
    let written = save_session(world).and_then(|()| {
        std::fs::write(paths.reload(), serde_json::to_vec(&ReloadRequest { binary })?)?;
        Ok(())
    });
    match written {
        Ok(()) => {
            world.write_message(AppExit::from_code(crate::RELOAD_EXIT_CODE));
        }
        Err(error) => {
            world.remove_resource::<RestartPending>();
            let message = format!("reload: could not save the session: {error:#}");
            if let Some(entity) = entity {
                world.entity_mut(entity).insert(CallOutput(message));
            } else {
                world.resource_mut::<Transcript>().push(EntryKind::Error, message);
            }
        }
    }
}

/// `cargo build` the agent crate. Returns the new executable, or the
/// compiler's errors.
async fn cargo_build(source: &Path) -> Result<PathBuf, String> {
    let mut command = tokio::process::Command::new("cargo");
    command
        .arg("build")
        .arg("--message-format=json-diagnostic-short")
        .current_dir(source)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if !cfg!(debug_assertions) {
        command.arg("--release");
    }
    let output = command
        .output()
        .await
        .map_err(|error| format!("cannot run cargo: {error}"))?;

    let mut errors = Vec::new();
    let mut executable = None;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Ok(message) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match message["reason"].as_str() {
            Some("compiler-message") if message["message"]["level"] == "error" => {
                if let Some(rendered) = message["message"]["rendered"].as_str() {
                    errors.push(rendered.trim_end().to_string());
                }
            }
            Some("compiler-artifact") if message["target"]["name"] == env!("CARGO_PKG_NAME") => {
                if let Some(path) = message["executable"].as_str() {
                    executable = Some(PathBuf::from(path));
                }
            }
            _ => {}
        }
    }
    match executable {
        Some(executable) if output.status.success() => Ok(executable),
        _ => {
            if errors.is_empty() {
                errors.push(String::from_utf8_lossy(&output.stderr).trim().to_string());
            }
            Err(crate::plugins::truncate(&errors.join("\n"), 8_000))
        }
    }
}
