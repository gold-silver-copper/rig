//! bevy-agent: a minimal coding agent. Rig talks to the model, Bevy runs
//! everything else as plugins, ratatui draws it, and a small supervisor
//! restarts it into a freshly built binary when it reloads itself.

mod agent;
mod brp;
mod plugins;
mod reload;
mod session;
mod supervisor;
mod tui;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;

use session::{Session, StateDir};

/// Set in the environment of the agent process the supervisor starts.
const CHILD_ENV: &str = "BEVY_AGENT_CHILD";
/// How many restarts came before this process; 0 is the first start.
const GENERATION_ENV: &str = "BEVY_AGENT_GENERATION";
const DEFAULT_MODEL: &str = "anthropic:claude-opus-5-5";

#[derive(Default)]
struct Args {
    model: Option<String>,
    state_dir: Option<PathBuf>,
    brp_port: Option<u16>,
    resume: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        let mut value = || iter.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--model" => args.model = Some(value()?),
            "--state-dir" => args.state_dir = Some(value()?.into()),
            "--brp-port" => {
                let port = value()?;
                args.brp_port = Some(port.parse().map_err(|_| format!("bad port `{port}`"))?);
            }
            "--continue" => args.resume = true,
            _ => {
                return Err(format!(
                    "unknown argument `{arg}`\nusage: bevy-agent [--model vendor:model] \
                     [--state-dir dir] [--brp-port port] [--continue]"
                ));
            }
        }
    }
    Ok(args)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };
    let state = std::env::var_os(supervisor::STATE_ENV)
        .map(PathBuf::from)
        .or(args.state_dir.clone())
        .unwrap_or_else(|| ".bevy-agent".into());
    let state = std::path::absolute(&state).unwrap_or(state);
    if std::env::var_os(CHILD_ENV).is_some() {
        run_agent(args, StateDir(state))
    } else {
        supervisor::run(&state)
    }
}

/// What this process knows about how it was started.
#[derive(Resource)]
pub struct Boot {
    pub generation: u32,
}

fn run_agent(args: Args, state: StateDir) -> ExitCode {
    let generation = std::env::var(GENERATION_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let mut session = if generation > 0 || args.resume {
        session::load(&state.session()).unwrap_or_default()
    } else {
        Session::default()
    };
    // Command-line choices apply to the first start only: after that the
    // session holds whatever the user switched to.
    if generation == 0 {
        if let Some(model) = args.model {
            session.model = model;
        }
        if let Some(port) = args.brp_port {
            session.brp_port = port;
        }
    }
    if session.model.is_empty() {
        session.model = DEFAULT_MODEL.into();
    }
    if session.brp_port == 0 {
        session.brp_port = brp::free_port();
    }
    supervisor::resume(&mut session, &state, generation);
    let port = session.brp_port;

    let mut app = App::new();
    app.add_plugins(
        MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 30.0,
        ))),
    )
    .insert_resource(Boot { generation })
    .insert_resource(state.clone())
    .insert_resource(session)
    .add_plugins((
        agent::AgentPlugin,
        tui::TuiPlugin,
        reload::ReloadPlugin,
        brp::BrpPlugin { port },
        plugins::NativePlugins,
    ))
    .add_systems(Update, supervisor::mark_started);

    let exit = app.run();
    ratatui::restore();
    match exit {
        AppExit::Success => ExitCode::SUCCESS,
        AppExit::Error(code) => ExitCode::from(code.get()),
    }
}
