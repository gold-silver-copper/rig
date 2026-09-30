//! bevy-agent: a minimal coding agent. Rig talks to models, Bevy runs
//! everything else, ratatui draws it, and a small supervisor restarts it
//! into a freshly built binary when it reloads itself.
//!
//! The core is [`CorePlugins`]: the Bevy-Rig [`glue`], sessions, the native
//! [`tools`], reload and, when interactive, the [`tui`]. Every other feature
//! is a plugin in [`plugins`] that can be left out.

// Bevy queries name their components in their types.
#![allow(clippy::type_complexity)]

pub mod glue;
pub mod plugins;
pub mod reload;
pub mod session;
pub mod supervisor;
pub mod tools;
pub mod tui;

use std::path::{Path, PathBuf};
use std::time::Duration;

use bevy::app::{PluginGroupBuilder, ScheduleRunnerPlugin};
use bevy::prelude::*;
use serde_json::{Map, Value};

use glue::{Agent, GluePlugin};
use session::{Origin, Preamble, SessionId, SessionPlugin, SessionStore};

/// The model new sessions start with unless told otherwise.
pub const DEFAULT_MODEL: &str = "anthropic:claude-opus-5-5";

/// How this process runs.
#[derive(Resource, Clone)]
pub struct State {
    /// Where the supervisor, sessions and binaries keep their files.
    pub dir: PathBuf,
    /// How many restarts came before this process; 0 is the first start.
    pub generation: u32,
    /// The working directory as the footer shows it.
    pub cwd_label: String,
    /// The model new sessions get.
    pub model: String,
    /// Whether the TUI session is opened.
    pub tui: bool,
}

/// Values plugins keep across a restart, such as the BRP port, saved with
/// the sessions.
#[derive(Resource, Default)]
pub struct ProcessState(pub Map<String, Value>);

impl ProcessState {
    fn path(dir: &Path) -> PathBuf {
        dir.join(supervisor::SESSIONS).join("process.json")
    }

    pub fn load(dir: &Path) -> Self {
        let values = std::fs::read(Self::path(dir))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        Self(values)
    }

    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        let path = Self::path(dir);
        std::fs::create_dir_all(path.parent().unwrap_or(dir))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(&self.0)?)?;
        std::fs::rename(tmp, path)
    }
}

/// The port the BRP plugin serves, when it runs.
#[derive(Resource, Clone, Copy)]
pub struct BrpPort(pub u16);

/// Marks a process whose sessions outlive it; without it, a clean quit
/// deletes what reloads saved.
#[derive(Resource)]
pub struct KeepSessions;

/// The core: the glue, sessions, native tools, reload and, if `tui`, the TUI.
pub struct CorePlugins {
    pub tui: bool,
}

impl PluginGroup for CorePlugins {
    fn build(self) -> PluginGroupBuilder {
        let group = PluginGroupBuilder::start::<Self>()
            .add(GluePlugin)
            .add(SessionPlugin)
            .add(reload::ReloadPlugin)
            .add(CorePlugin)
            .add_group(tools::ToolPlugins);
        if self.tui {
            group.add(tui::TuiPlugin)
        } else {
            group
        }
    }
}

struct CorePlugin;

impl Plugin for CorePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, open_sessions.in_set(OpenSessions))
            .add_systems(Update, supervisor::mark_started)
            .add_systems(Last, on_exit);
    }
}

/// The startup step that restores or opens sessions; plugins that open
/// their own sessions run before it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpenSessions;

/// A headless app with the core but without the TUI, running at 30 frames
/// a second: the base of the agent binary, evals and tests.
pub fn app(state: State, deterministic: bool) -> App {
    let mut app = App::new();
    let preamble = preamble(deterministic);
    let dir = state.dir.clone();
    let tui = state.tui;
    app.add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(
        Duration::from_secs_f64(1.0 / 30.0),
    )))
    .insert_resource(ProcessState::load(&dir))
    .insert_resource(SessionStore::new(dir.join(supervisor::SESSIONS)))
    .insert_resource(Preamble(preamble))
    .insert_resource(state)
    .add_plugins(CorePlugins { tui });
    app
}

/// Restores the sessions of the process that reloaded, then opens the TUI
/// session if there is none yet.
fn open_sessions(world: &mut World) {
    let state = world.resource::<State>().clone();
    if state.generation > 0 {
        let ids: Vec<String> = world
            .resource::<ProcessState>()
            .0
            .get("agents")
            .and_then(|ids| serde_json::from_value(ids.clone()).ok())
            .unwrap_or_default();
        for id in ids {
            session::restore_agent(world, &id);
        }
    }
    let has_tui = world
        .query::<&Origin>()
        .iter(world)
        .any(|origin| *origin == Origin::Tui);
    if state.tui && !has_tui {
        session::spawn_agent(world, session::new_session_id(), Origin::Tui, &state.model);
    }
    supervisor::settle_restart(world);
}

/// Saves every session and the process state, for a reload.
pub fn save_everything(world: &mut World) -> std::io::Result<()> {
    let agents: Vec<(Entity, String)> = world
        .query_filtered::<(Entity, &SessionId), With<Agent>>()
        .iter(world)
        .map(|(entity, id)| (entity, id.0.clone()))
        .collect();
    for (agent, _) in &agents {
        session::save_agent(world, *agent)?;
    }
    let ids: Vec<String> = agents.into_iter().map(|(_, id)| id).collect();
    world
        .resource_mut::<ProcessState>()
        .0
        .insert("agents".into(), Value::from(ids));
    let dir = world.resource::<State>().dir.clone();
    world.resource::<ProcessState>().save(&dir)
}

/// On a clean quit, deletes what reloads saved unless sessions are kept.
fn on_exit(
    mut exits: MessageReader<AppExit>,
    keep: Option<Res<KeepSessions>>,
    ids: Query<&SessionId>,
    store: Res<SessionStore>,
    state: Res<State>,
) {
    if !exits.read().any(AppExit::is_success) || keep.is_some() {
        return;
    }
    for id in &ids {
        store.remove(&id.0);
    }
    let _ = std::fs::remove_file(ProcessState::path(&state.dir));
}

/// The system prompt, after pi's. In deterministic mode, used for recording
/// and replaying cassettes, it names paths relative to the working
/// directory, so the same session sends the same requests on any machine.
pub fn preamble(deterministic: bool) -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let source = Path::new(env!("CARGO_MANIFEST_DIR"));
    let show = |path: &Path| {
        if deterministic {
            relative(path, &cwd)
        } else {
            path.display().to_string()
        }
    };
    let cwd_text = if deterministic {
        ".".to_owned()
    } else {
        cwd.display().to_string()
    };
    format!(
        "You are an expert coding assistant operating inside bevy-agent, a coding agent harness. \
You help users by reading files, executing commands, editing code, and writing new files.

<tools>
- read: Read file contents
- bash: Execute bash commands (ls, grep, find, etc.)
- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
- write: Create or overwrite files
- reload: Rebuild yourself from your source and restart into the new binary

In addition to the tools above, you may have access to other custom tools depending on the project.
</tools>

<rules>
- Use bash for file operations like ls, rg, find
- Use read to examine files instead of cat or sed.
- Use edit for precise changes (edits[].oldText must match exactly)
- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls
- Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.
- Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.
- Use write only for new files or complete rewrites.
- Never print environment variables or credentials.
- Be concise in your responses
- Show file paths clearly when working with files
</rules>

<self>
Your own source code is the Rust crate at {source}, a Bevy app. Each native tool is an ordinary \
Bevy plugin in {source}/src/tools/, and optional features are plugins in {source}/src/plugins/. \
You use Rig for models; its crates are in {rig}. To change yourself, edit that source and call \
`reload`. It rebuilds you. If the build fails you get the compiler errors and keep running \
unchanged. If it succeeds you restart into the new binary with this conversation intact and the \
`reload` call returns, so you can use your changes in the same turn.
</self>

<cwd>
{cwd_text}
</cwd>",
        source = show(source),
        rig = show(&source.join("../crates")),
    )
}

/// `path` relative to `base`, both made absolute first.
fn relative(path: &Path, base: &Path) -> String {
    let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let base = std::fs::canonicalize(base).unwrap_or_else(|_| base.to_path_buf());
    let common = path
        .components()
        .zip(base.components())
        .take_while(|(a, b)| a == b)
        .count();
    let mut relative = PathBuf::new();
    for _ in base.components().skip(common) {
        relative.push("..");
    }
    for part in path.components().skip(common) {
        relative.push(part);
    }
    if relative.as_os_str().is_empty() {
        ".".into()
    } else {
        relative.display().to_string()
    }
}
