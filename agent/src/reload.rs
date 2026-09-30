//! Reload: build the agent's own crate, and only if that succeeds, save
//! every agent, hand the new binary to the supervisor and exit.
//!
//! The model asks with the `reload` tool, the user with `/reload`. A failed
//! build answers the tool call with the compiler errors and the agent keeps
//! running. After a successful build the `reload` call stays unfinished in
//! its agent's saved batch; the new process answers it with the
//! supervisor's note and the turn goes on.

use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use rig_core::completion::ToolDefinition;
use serde_json::json;

use crate::glue::{AgentSet, Call, Done, Ready, Running, Tool, Tools};
use crate::prompt::Prompt;
use crate::session::{Focused, Transcript, EntryKind, save_all};
use crate::{Env, RELOAD_EXIT};

pub const TOOL: &str = "reload";

/// A `/reload` from the user.
#[derive(Message, Clone, Copy, Debug)]
pub struct ReloadRequest;

/// A build in flight for a `reload` call.
#[derive(Component)]
struct Building(Task<Result<PathBuf, String>>);

/// The call's build succeeded; the restart is pending.
#[derive(Component)]
struct Built;

/// The build `/reload` started, and the binary to restart into once no tool
/// is running.
#[derive(Resource, Default)]
pub struct Reloading {
    build: Option<Task<Result<PathBuf, String>>>,
    ready: Option<PathBuf>,
}

impl Reloading {
    pub fn busy(&self) -> bool {
        self.build.is_some() || self.ready.is_some()
    }
}

pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        let source = relative_source(&app.world().resource::<Env>().source);
        app.world_mut().get_resource_or_init::<Tools>().add(Tool::World(ToolDefinition {
            name: TOOL.into(),
            description: "Rebuild your own source with cargo and restart into the new binary. The conversation \
                          continues after the restart. If the build fails, returns the compiler errors and the \
                          current version keeps running."
                .into(),
            parameters: json!({"type": "object", "properties": {}}),
        }));
        let mut prompt = app.world_mut().get_resource_or_init::<Prompt>();
        prompt.tool(
            TOOL,
            "Rebuild your own source and restart into it; the conversation continues",
            &[],
        );
        prompt.section(
            "self",
            format!(
                "Your own source code is the Rust crate at `{source}` (a Bevy app; Rig talks to the models; \
                 ratatui draws the terminal). Native plugins are ordinary Bevy plugins in `{source}/src/plugins`, \
                 and the rig crates you use are in `{source}/../crates`. You may edit any of it, including Cargo \
                 manifests, then call reload to rebuild and restart into the new version and keep working in \
                 the same turn. If the build fails you get the compiler errors and keep running the old version."
            ),
        );
        app.init_resource::<Reloading>()
            .add_message::<ReloadRequest>()
            .add_systems(Update, (reload_calls, user_reload, restart).chain().after(AgentSet::Run));
    }
}

/// `source` relative to the working directory, so the prompt holds no
/// machine-specific path.
fn relative_source(source: &Path) -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let (Ok(source), Ok(cwd)) = (source.canonicalize(), cwd.canonicalize()) else {
        return "agent".into();
    };
    let common = source.components().zip(cwd.components()).take_while(|(a, b)| a == b).count();
    let ups = cwd.components().count() - common;
    let mut parts: Vec<String> = std::iter::repeat_n("..".to_owned(), ups).collect();
    parts.extend(source.components().skip(common).map(|c| c.as_os_str().to_string_lossy().into_owned()));
    if parts.is_empty() { ".".into() } else { parts.join("/") }
}

fn reload_calls(
    calls: Query<(Entity, &Call, Option<&mut Building>), (With<Ready>, Without<Done>, Without<Built>)>,
    mut reloading: ResMut<Reloading>,
    env: Res<Env>,
    mut commands: Commands,
) {
    let mut calls = calls;
    for (entity, call, building) in &mut calls {
        if call.call.function.name != TOOL {
            continue;
        }
        match building {
            None => {
                commands.entity(entity).insert(Building(spawn_build(&env)));
            }
            Some(mut building) => match check_ready(&mut building.0) {
                None => {}
                Some(Ok(binary)) => {
                    reloading.ready = Some(binary);
                    commands.entity(entity).remove::<Building>().insert(Built);
                }
                Some(Err(errors)) => {
                    commands.entity(entity).remove::<Building>().try_insert(Done(Err(errors)));
                }
            },
        }
    }
}

fn user_reload(
    mut requests: MessageReader<ReloadRequest>,
    mut reloading: ResMut<Reloading>,
    mut focused: Query<&mut Transcript, With<Focused>>,
    env: Res<Env>,
) {
    for _ in requests.read() {
        if reloading.busy() {
            continue;
        }
        if let Ok(mut transcript) = focused.single_mut() {
            transcript.log(EntryKind::Info, "building…");
        }
        reloading.build = Some(spawn_build(&env));
    }
    if let Some(task) = &mut reloading.build
        && let Some(result) = check_ready(task)
    {
        reloading.build = None;
        match result {
            Ok(binary) => reloading.ready = Some(binary),
            Err(errors) => {
                if let Ok(mut transcript) = focused.single_mut() {
                    transcript.log(EntryKind::Error, errors);
                }
            }
        }
    }
}

/// Once a build is ready and no tool is running anywhere, save everything
/// and exit for the supervisor to restart us.
fn restart(world: &mut World) {
    if world.resource::<Reloading>().ready.is_none() {
        return;
    }
    if world.query::<&Running>().iter(world).next().is_some() {
        return;
    }
    let Some(binary) = world.resource_mut::<Reloading>().ready.take() else {
        return;
    };
    let env = world.resource::<Env>().clone();
    save_all(world);
    if let Err(error) = std::fs::write(env.state.join("next-binary"), binary.as_os_str().as_encoded_bytes()) {
        error!("cannot restart: {error}");
        return;
    }
    world.write_message(AppExit::Error(NonZero::new(RELOAD_EXIT as u8).unwrap_or(NonZero::<u8>::MIN)));
}

pub fn spawn_build(env: &Env) -> Task<Result<PathBuf, String>> {
    let (source, state) = (env.source.clone(), env.state.clone());
    AsyncComputeTaskPool::get().spawn(async move { build(&source, &state) })
}

/// `cargo build` the agent in the profile this binary was built with, then
/// copy the result out of `target/`, so later builds cannot overwrite a
/// binary the supervisor may fall back to.
fn build(source: &Path, state: &Path) -> Result<PathBuf, String> {
    let release = !cfg!(debug_assertions);
    let mut cargo = Command::new("cargo");
    cargo.arg("build").arg("--color=never").arg("--bin").arg("rig-pi");
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
    let built = target.join(if release { "release" } else { "debug" }).join("rig-pi");
    let millis = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or_default();
    let bin = state.join("bin");
    std::fs::create_dir_all(&bin).map_err(|e| e.to_string())?;
    let copy = bin.join(format!("rig-pi-{millis}"));
    std::fs::copy(&built, &copy).map_err(|e| format!("{}: {e}", built.display()))?;
    Ok(copy)
}

/// The error diagnostics of a failed build, without cargo's progress lines
/// or warnings, which would crowd the errors out.
fn errors_only(stderr: &str) -> String {
    let stderr: String = stderr
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            !["Compiling ", "Checking ", "Blocking ", "Building "].iter().any(|p| line.starts_with(p))
        })
        .map(|line| format!("{line}\n"))
        .collect();
    let blocks: Vec<&str> = stderr
        .split("\n\n")
        .filter(|block| block.trim_start().starts_with("error"))
        .collect();
    let text = if blocks.is_empty() { stderr.clone() } else { blocks.join("\n\n") };
    if text.len() > 12_000 {
        let cut = (0..=12_000).rev().find(|i| text.is_char_boundary(*i)).unwrap_or(0);
        format!("{}\n[truncated]", &text[..cut])
    } else {
        text
    }
}
