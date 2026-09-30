//! The bevy-agent binary: the supervisor the user starts, and the agent
//! process it runs.

use std::path::PathBuf;
use std::process::ExitCode;

use bevy::prelude::*;
use bevy_agent::supervisor::{self, CHILD_ENV, GENERATION_ENV, STATE_ENV};
use bevy_agent::{DEFAULT_MODEL, State, plugins};
use rig_cassette::http::CassetteMode;

#[derive(Default)]
struct Args {
    model: Option<String>,
    state_dir: Option<PathBuf>,
    disabled: Vec<String>,
    brp_port: Option<u16>,
    resume: Option<plugins::durable::Resume>,
    compact_at: Option<usize>,
    cassette: Option<(CassetteMode, String)>,
    cassettes: Option<PathBuf>,
    headless: bool,
}

const USAGE: &str = "usage: bevy-agent [--model vendor:model] [--state-dir dir] [--brp-port port]
                  [--continue | --resume id] [--compact-at tokens]
                  [--record name | --replay name] [--cassettes dir] [--headless]
                  [--disable plugin,...]
plugins: brp, remote, durable";

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
            "--continue" => args.resume = Some(plugins::durable::Resume::Latest),
            "--resume" => args.resume = Some(plugins::durable::Resume::Id(value()?)),
            "--compact-at" => {
                let tokens = value()?;
                args.compact_at = Some(
                    tokens
                        .parse()
                        .map_err(|_| format!("bad number `{tokens}`"))?,
                );
            }
            "--record" => args.cassette = Some((CassetteMode::Record, value()?)),
            "--replay" => args.cassette = Some((CassetteMode::Replay, value()?)),
            "--cassettes" => args.cassettes = Some(value()?.into()),
            "--headless" => args.headless = true,
            "--disable" => args
                .disabled
                .extend(value()?.split(',').map(|name| name.trim().to_owned())),
            _ => return Err(format!("unknown argument `{arg}`\n{USAGE}")),
        }
    }
    Ok(args)
}

fn main() -> ExitCode {
    if std::env::args().nth(1).as_deref() == Some("eval") {
        return eval();
    }
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
    if std::env::var_os(CHILD_ENV).is_none() && !args.headless {
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
        tui: !args.headless,
        primary: true,
    };
    let mut app = bevy_agent::app(state, args.cassette.is_some());
    let enabled = |name: &str| !args.disabled.iter().any(|disabled| disabled == name);
    if let Some((mode, name)) = &args.cassette {
        let root = args
            .cassettes
            .clone()
            .unwrap_or_else(bevy_agent::cassette_root);
        app.add_plugins(plugins::cassette::CassettePlugin {
            mode: *mode,
            dir: root.join(name),
            session: true,
        });
    }
    if enabled("durable") {
        app.add_plugins(plugins::durable::DurablePlugin {
            resume: args.resume.clone(),
            compact_at: args.compact_at.unwrap_or(150_000),
        });
    }
    if enabled("brp") {
        app.add_plugins(plugins::brp::BrpPlugin {
            port: args.brp_port,
        });
        if enabled("remote") {
            app.add_plugins(plugins::remote::RemoteSessionsPlugin);
        }
    }
    let exit = app.run();
    if !args.headless {
        ratatui::restore();
    }
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

/// `bevy-agent eval <suite.json> [--record] [--cassettes dir]`: runs a suite
/// headless in a fresh copy of its workspace, replaying its cassettes unless
/// told to record them.
fn eval() -> ExitCode {
    let mut suite_path = None;
    let mut mode = CassetteMode::Replay;
    let mut root = bevy_agent::cassette_root();
    let mut iter = std::env::args().skip(2);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--record" => mode = CassetteMode::Record,
            "--cassettes" => root = iter.next().map(PathBuf::from).unwrap_or(root),
            _ => suite_path = Some(PathBuf::from(arg)),
        }
    }
    let Some(suite_path) = suite_path.and_then(|path| std::path::absolute(path).ok()) else {
        eprintln!("usage: bevy-agent eval <suite.json> [--record] [--cassettes dir]");
        return ExitCode::FAILURE;
    };
    let root = std::path::absolute(&root).unwrap_or(root);
    let suite = match plugins::eval::Suite::load(&suite_path) {
        Ok(suite) => suite,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let name = suite_path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "suite".into());
    // A fixed place relative to the source, so the relative paths in the
    // system prompt, and so the recorded requests, are the same everywhere.
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/eval")
        .join(&name);
    let _ = std::fs::remove_dir_all(&workspace);
    if let Err(error) = std::fs::create_dir_all(&workspace) {
        eprintln!("{}: {error}", workspace.display());
        return ExitCode::FAILURE;
    }
    if let Some(files) = &suite.workspace {
        let from = suite_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(files);
        if let Err(error) = copy_tree(&from, &workspace) {
            eprintln!("{}: {error}", from.display());
            return ExitCode::FAILURE;
        }
    }
    if let Err(error) = std::env::set_current_dir(&workspace) {
        eprintln!("{}: {error}", workspace.display());
        return ExitCode::FAILURE;
    }
    let state = State {
        dir: workspace.join(".bevy-agent"),
        generation: 0,
        cwd_label: name.clone(),
        model: DEFAULT_MODEL.into(),
        tui: false,
        primary: false,
    };
    let mut app = bevy_agent::app(state, true);
    app.add_plugins((
        plugins::cassette::CassettePlugin {
            mode,
            dir: root.join(format!("eval-{name}")),
            session: false,
        },
        plugins::eval::EvalPlugin {
            suite,
            timeout: std::time::Duration::from_secs(300),
        },
    ));
    match app.run() {
        AppExit::Success => ExitCode::SUCCESS,
        AppExit::Error(code) => ExitCode::from(code.get()),
    }
}

fn copy_tree(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}
