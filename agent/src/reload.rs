//! Reload: build the agent's own crate, and only if that succeeds, hand the
//! new binary to the supervisor and exit. The `reload` tool and `/reload`
//! both come here.

use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use serde_json::json;

use crate::agent::InFlight;
use crate::session::{Kind, Session};
use crate::tools::{Run, Tools, clip};
use crate::{Env, RELOAD_EXIT};

pub const TOOL: &str = "reload";

/// A build started by `/reload`, and the binary it produced once it is safe
/// to restart (no tool running).
#[derive(Resource, Default)]
pub struct UserReload {
    build: Option<Task<Result<PathBuf, String>>>,
    ready: Option<PathBuf>,
}

impl UserReload {
    pub fn building(&self) -> bool {
        self.build.is_some() || self.ready.is_some()
    }

    pub fn start(&mut self, session: &mut Session, env: &Env) {
        if self.build.is_some() || self.ready.is_some() {
            return session.log(Kind::Info, "a reload is already in progress");
        }
        session.log(Kind::Info, "building…");
        self.build = Some(spawn_build(env));
    }
}

pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        let def = ToolDefinition::new(
            ToolName::new(TOOL).unwrap_or_else(|e| panic!("{e}")),
            "Rebuild your own source with cargo and restart into the new binary. The conversation \
             continues after the restart. If the build fails, returns the compiler errors and the \
             current version keeps running.",
            json!({"type": "object", "properties": {}}),
        );
        app.world_mut()
            .get_resource_or_init::<Tools>()
            .insert(def, Run::Reload);
        app.init_resource::<UserReload>()
            .add_systems(Update, user_reload.before(crate::agent::drive));
    }
}

fn user_reload(
    mut user: ResMut<UserReload>,
    mut session: ResMut<Session>,
    inflight: Res<InFlight>,
    env: Res<Env>,
    mut exit: MessageWriter<AppExit>,
) {
    if let Some(task) = &mut user.build
        && let Some(result) = check_ready(task)
    {
        user.build = None;
        match result {
            Ok(binary) => user.ready = Some(binary),
            Err(errors) => session.log(Kind::Error, errors),
        }
    }
    if !inflight.tool_running()
        && let Some(binary) = user.ready.take()
    {
        if let Err(error) = restart(&mut session, &env, &binary, &mut exit) {
            session.log(Kind::Error, error);
        }
    }
}

pub fn spawn_build(env: &Env) -> Task<Result<PathBuf, String>> {
    let (source, state) = (env.source.clone(), env.state.clone());
    AsyncComputeTaskPool::get().spawn(async move { build(&source, &state) })
}

/// `cargo build` the agent in the profile this binary was built with, then
/// copy the result out of `target/` so later builds cannot overwrite a
/// binary the supervisor may fall back to.
fn build(source: &Path, state: &Path) -> Result<PathBuf, String> {
    let release = !cfg!(debug_assertions);
    let mut cargo = Command::new("cargo");
    cargo.arg("build").arg("--color=never");
    if release {
        cargo.arg("--release");
    }
    let output = cargo
        .current_dir(source)
        .env("CARGO_TERM_PROGRESS_WHEN", "never")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Build failed; the current version keeps running.\n{}",
            errors_only(&stderr)
        ));
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| source.join("target"));
    let built = target
        .join(if release { "release" } else { "debug" })
        .join("rig-pi");
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    let bin = state.join("bin");
    std::fs::create_dir_all(&bin).map_err(|e| e.to_string())?;
    let copy = bin.join(format!("rig-pi-{millis}"));
    std::fs::copy(&built, &copy).map_err(|e| format!("{}: {e}", built.display()))?;
    Ok(copy)
}

/// The error diagnostics of a failed build: rustc separates diagnostics
/// with blank lines, and warnings would crowd the errors out.
fn errors_only(stderr: &str) -> String {
    let stderr: String = stderr
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            !["Compiling ", "Checking ", "Blocking ", "Building "]
                .iter()
                .any(|progress| line.starts_with(progress))
        })
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
    clip(&text, 12_000, false)
}

/// Hand `binary` to the supervisor and exit. The session is saved first so
/// the new process resumes exactly here.
pub fn restart(
    session: &mut Session,
    env: &Env,
    binary: &Path,
    exit: &mut MessageWriter<AppExit>,
) -> Result<(), String> {
    session.log(Kind::Info, format!("restarting into {}", binary.display()));
    std::fs::write(
        env.state.join("next-binary"),
        binary.as_os_str().as_encoded_bytes(),
    )
    .and_then(|()| session.save(&env.state.join("session.json")))
    .map_err(|error| format!("cannot restart: {error}"))?;
    exit.write(AppExit::Error(
        NonZero::new(RELOAD_EXIT as u8).unwrap_or(NonZero::<u8>::MIN),
    ));
    Ok(())
}
