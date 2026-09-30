//! The bevy-agent binary: the supervisor the user starts, and the agent
//! process it runs.

use std::path::PathBuf;
use std::process::ExitCode;

use bevy::prelude::*;
use bevy_agent::supervisor::{self, CHILD_ENV, GENERATION_ENV, STATE_ENV};
use bevy_agent::{DEFAULT_MODEL, State, plugins};

#[derive(Default)]
struct Args {
    model: Option<String>,
    state_dir: Option<PathBuf>,
    disabled: Vec<String>,
    brp_port: Option<u16>,
}

const USAGE: &str = "usage: bevy-agent [--model vendor:model] [--state-dir dir] [--brp-port port]
                  [--disable plugin,...]
plugins: brp, remote";

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
            "--disable" => args
                .disabled
                .extend(value()?.split(',').map(|name| name.trim().to_owned())),
            _ => return Err(format!("unknown argument `{arg}`\n{USAGE}")),
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
    let dir = std::env::var_os(STATE_ENV)
        .map(PathBuf::from)
        .or(args.state_dir.clone())
        .unwrap_or_else(|| ".bevy-agent".into());
    let dir = std::path::absolute(&dir).unwrap_or(dir);
    if std::env::var_os(CHILD_ENV).is_none() {
        return supervisor::run(&dir);
    }
    let generation = std::env::var(GENERATION_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let state = State {
        dir,
        generation,
        cwd_label: cwd_label(),
        model: args.model.clone().unwrap_or_else(|| DEFAULT_MODEL.into()),
        tui: true,
    };
    let mut app = bevy_agent::app(state, false);
    let enabled = |name: &str| !args.disabled.iter().any(|disabled| disabled == name);
    if enabled("brp") {
        app.add_plugins(plugins::brp::BrpPlugin {
            port: args.brp_port,
        });
        if enabled("remote") {
            app.add_plugins(plugins::remote::RemoteSessionsPlugin);
        }
    }
    let exit = app.run();
    ratatui::restore();
    match exit {
        AppExit::Success => ExitCode::SUCCESS,
        AppExit::Error(code) => ExitCode::from(code.get()),
    }
}

/// The working directory with the home directory shortened to `~`, as pi shows it.
fn cwd_label() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    match std::env::var_os("HOME").map(PathBuf::from) {
        Some(home) if cwd.starts_with(&home) => {
            format!("~/{}", cwd.strip_prefix(&home).unwrap_or(&cwd).display())
        }
        _ => cwd.display().to_string(),
    }
}
