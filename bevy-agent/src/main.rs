//! bevy-agent: a minimal coding agent on Bevy and Rig that can rebuild and
//! restart itself. See DESIGN.md.

mod agent;
mod brp;
mod plugins;
mod reload;
mod session;
mod supervisor;
mod tools;
mod tui;

use std::process::ExitCode;
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;

use session::{PRESETS, Paths, Session, parse_model};

const USAGE: &str = "usage: bevy-agent [--model <number | name | vendor:model>] [--port <brp port>] [--continue]

  --model     model to start with (default 1):
                1. openai:gpt-6.1-sol   2. anthropic:claude-opus-5-5
                3. gcp.gemini:gemini-3.8-flash   4. deepseek:deepseek-flash
  --port      BRP port for external plugins (default: a free port)
  --continue  resume the last session in this state directory

State lives in $BEVY_AGENT_HOME (default ./.bevy-agent).";

struct Options {
    model: Option<String>,
    port: Option<u16>,
    resume: bool,
}

fn options(args: &[String]) -> Result<Options, String> {
    let mut options = Options { model: None, port: None, resume: false };
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => options.model = Some(args.next().ok_or("--model needs a value")?.clone()),
            "--port" => {
                let port = args.next().ok_or("--port needs a value")?;
                options.port = Some(port.parse().map_err(|_| format!("bad port `{port}`"))?);
            }
            "--continue" | "-c" => options.resume = true,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(options)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let paths = Paths::from_env();
    if std::env::var_os("BEVY_AGENT_CHILD").is_none() {
        if let Err(error) = options(&args) {
            eprintln!("{error}\n{USAGE}");
            return ExitCode::FAILURE;
        }
        return supervisor::run(&paths, &args);
    }
    match session(&paths, &args) {
        Ok(session) => run(paths, session),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn session(paths: &Paths, args: &[String]) -> Result<Session, String> {
    let options = options(args)?;
    let restarted = std::env::var("BEVY_AGENT_GENERATION").is_ok_and(|g| g != "0");
    let previous = Session::load(&paths.session());
    let mut session = match previous {
        Some(session) if restarted || options.resume => session,
        stale => {
            if stale.is_some() {
                let _ = std::fs::rename(paths.session(), paths.home.join(format!("session-{}.json", std::process::id())));
            }
            Session::new(parse_model(PRESETS[0])?, 0)
        }
    };
    if let Some(model) = options.model {
        session.model = parse_model(&model)?;
    }
    if let Some(port) = options.port {
        session.port = port;
    }
    if session.port == 0 {
        session.port = free_port().map_err(|e| format!("no free port: {e}"))?;
    }
    let _ = std::fs::write(paths.home.join("brp-port"), session.port.to_string());
    session.save(&paths.session());
    Ok(session)
}

fn free_port() -> std::io::Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

fn run(paths: Paths, session: Session) -> ExitCode {
    let port = session.port;
    if let Ok(log) = std::fs::OpenOptions::new().create(true).append(true).open(paths.home.join("agent.log")) {
        tracing_subscriber::fmt()
            .with_writer(std::sync::Mutex::new(log))
            .with_max_level(tracing_subscriber::filter::LevelFilter::WARN)
            .init();
    }
    let exit = App::new()
        .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_millis(20))))
        .insert_resource(paths)
        .insert_resource(session)
        .add_plugins((
            agent::AgentPlugin,
            reload::ReloadPlugin,
            plugins::NativePlugins,
            brp::BrpPlugin { port },
            tui::TuiPlugin,
        ))
        .run();
    ratatui::restore();
    match exit {
        AppExit::Success => ExitCode::SUCCESS,
        AppExit::Error(code) => ExitCode::from(code.get()),
    }
}
