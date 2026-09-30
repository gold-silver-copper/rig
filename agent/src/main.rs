//! rig-pi: a minimal coding agent. Bevy runs the app, Rig talks to the
//! models, ratatui draws the terminal, and BRP lets other processes plug in.
//!
//! `rig-pi` starts a supervisor that runs the agent as a child process. The
//! agent reloads by rebuilding itself and exiting with [`RELOAD_EXIT`]; the
//! supervisor restarts it into the new binary, or into the last good one if
//! the new binary does not start.

mod agent;
mod brp;
mod plugins;
mod reload;
mod session;
mod supervisor;
mod tools;
mod tui;

use std::path::PathBuf;
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::log::LogPlugin;
use bevy::prelude::*;

/// Exit code the agent uses to ask the supervisor for a restart into
/// `<state>/next-binary`.
pub const RELOAD_EXIT: i32 = 75;

/// The models `/model` offers by number; the first is the default. Any
/// `vendor[/format]:model` that rig's provider registry resolves also works.
pub fn presets() -> [String; 4] {
    use rig_core::providers::{anthropic, deepseek, gemini, openai};
    [
        format!("openai:{}", openai::GPT_6_1_SOL),
        format!("anthropic:{}", anthropic::completion::CLAUDE_OPUS_5_5),
        format!(
            "{}:{}",
            gemini::PROVIDER_NAME,
            gemini::completion::GEMINI_3_8_FLASH
        ),
        format!("deepseek:{}", deepseek::DEEPSEEK_V4_1_FLASH),
    ]
}

/// Where this agent lives and what it was asked to do, fixed for the life of
/// one process.
#[derive(Resource, Clone, Debug)]
pub struct Env {
    /// The agent's own crate: what `reload` builds.
    pub source: PathBuf,
    /// Session file, binaries, logs.
    pub state: PathBuf,
    /// BRP HTTP port, stable across reloads.
    pub port: u16,
    /// What the supervisor wants the agent told on startup.
    pub note: Option<String>,
    /// Marker file the agent creates once it is up.
    pub ready_file: Option<PathBuf>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--child") {
        std::process::exit(run_agent());
    }
    std::process::exit(supervisor::run(&args));
}

/// Run the agent app in this process and return its exit code.
fn run_agent() -> i32 {
    let var = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
    let env = Env {
        source: var("RIG_PI_SOURCE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
        state: var("RIG_PI_STATE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".rig-pi")),
        port: var("RIG_PI_PORT")
            .and_then(|port| port.parse().ok())
            .unwrap_or(0),
        note: var("RIG_PI_NOTE"),
        ready_file: var("RIG_PI_READY").map(PathBuf::from),
    };
    let model = var("RIG_PI_MODEL");

    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_millis(16))),
        LogPlugin {
            filter: "warn,rig_pi=info".into(),
            ..default()
        },
    ))
    .insert_resource(env.clone())
    .add_plugins((
        session::SessionPlugin { model },
        tools::ToolsPlugin,
        agent::AgentPlugin,
        reload::ReloadPlugin,
        brp::BrpPlugin { port: env.port },
        tui::TuiPlugin,
        plugins::NativePlugins,
        supervisor::ReadyPlugin,
    ));

    let exit = app.run();
    tui::restore();
    match exit {
        AppExit::Success => 0,
        AppExit::Error(code) => i32::from(code.get()),
    }
}
