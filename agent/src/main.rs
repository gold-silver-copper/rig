//! bevy-agent: a minimal coding agent on Bevy, Rig and ratatui.
//!
//! The binary runs in two roles. Started by a user it is the *supervisor*: it
//! picks the BRP port, copies itself aside as the first known-good binary and
//! launches itself again with `--child`. The *child* is the Bevy app that owns
//! the terminal. A reload is the child exiting with [`RELOAD_EXIT_CODE`] after
//! building its successor; the supervisor starts the successor, and falls back
//! to the last binary that started if the successor does not.

mod agent;
mod brp;
mod plugins;
mod reload;
mod session;
mod supervisor;
mod tui;

use std::path::PathBuf;
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;

/// Exit code a child uses to ask the supervisor to start the binary named in
/// `reload.json`.
pub const RELOAD_EXIT_CODE: u8 = 75;

/// Command-line options shared by the supervisor and the child.
#[derive(Clone, Debug, Default)]
pub struct Options {
    pub child: bool,
    pub model: Option<String>,
    pub state_dir: Option<PathBuf>,
    pub brp_port: Option<u16>,
    /// Continue the conversation saved in the state directory.
    pub resume: bool,
    /// How the reload that started this child went, if one did.
    pub reload: Option<ReloadStatus>,
}

#[derive(Clone, Debug)]
pub enum ReloadStatus {
    /// This process is the freshly built binary.
    Succeeded,
    /// The new binary did not start; this is the last good one. Holds why.
    Failed(String),
}

impl Options {
    fn parse() -> anyhow::Result<Self> {
        let mut options = Options::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))
            };
            match arg.as_str() {
                "--child" => options.child = true,
                "--model" | "-m" => options.model = Some(value()?),
                "--state-dir" => options.state_dir = Some(PathBuf::from(value()?)),
                "--brp-port" => options.brp_port = Some(value()?.parse()?),
                "--continue" | "--resume" => options.resume = true,
                "--reload-ok" => options.reload = Some(ReloadStatus::Succeeded),
                "--reload-failed" => options.reload = Some(ReloadStatus::Failed(value()?)),
                "--help" | "-h" => {
                    println!(
                        "usage: bevy-agent [--model MODEL] [--state-dir DIR] [--brp-port PORT] [--continue]\n\
                         MODEL is an alias ({}) or any Rig provider reference `vendor[/format]:model`.",
                        agent::MODEL_ALIASES
                            .iter()
                            .map(|(alias, ..)| *alias)
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument `{other}` (try --help)"),
            }
        }
        Ok(options)
    }

    /// Where sessions, built binaries and logs live.
    pub fn state_dir(&self) -> PathBuf {
        self.state_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(".bevy-agent"))
    }
}

fn main() -> std::process::ExitCode {
    let options = match Options::parse() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("bevy-agent: {error}");
            return std::process::ExitCode::from(2);
        }
    };
    if !options.child {
        return supervisor::run(options);
    }
    match run_child(options) {
        AppExit::Success => std::process::ExitCode::SUCCESS,
        AppExit::Error(code) => std::process::ExitCode::from(code.get()),
    }
}

fn run_child(options: Options) -> AppExit {
    let session = match session::Session::open(&options) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("bevy-agent: cannot open session: {error:#}");
            return AppExit::error();
        }
    };
    let exit = App::new()
        .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_millis(16))))
        .insert_resource(session)
        .add_plugins((
            agent::AgentPlugin(options.clone()),
            tui::TuiPlugin,
            brp::BrpPlugin,
            reload::ReloadPlugin,
            plugins::NativePlugins,
        ))
        .run();
    tui::restore_terminal();
    exit
}
