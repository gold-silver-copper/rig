//! The supervisor: runs the agent as a child process and restarts it.
//!
//! Exit code [`RELOAD_EXIT`] means "restart into `<state>/next-binary`". A
//! child counts as started once it creates its ready file; one that exits
//! before that is replaced by the last binary that did start, and the agent
//! is told why through `RIG_PI_NOTE`.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bevy::prelude::*;

use crate::{Env, RELOAD_EXIT};

const USAGE: &str = "usage: rig-pi [--model vendor:model] [--port N] [--state DIR]

  --model   start on this model (default: the saved one, else openai:gpt-6.1-sol)
  --port    BRP HTTP port (default: the saved one, else a free port)
  --state   session, binaries and logs (default: ./.rig-pi)";

/// How long a new binary gets to come up before it counts as failed.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

pub fn run(args: &[String]) -> i32 {
    let mut model = None;
    let mut port = None;
    let mut state = PathBuf::from(".rig-pi");
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match (arg.as_str(), args.next()) {
            ("--model", Some(value)) => model = Some(value.clone()),
            ("--port", Some(value)) => port = value.parse().ok(),
            ("--state", Some(value)) => state = PathBuf::from(value),
            _ => {
                eprintln!("{USAGE}");
                return 2;
            }
        }
    }
    match supervise(model, port, &state) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("rig-pi: {error}");
            1
        }
    }
}

fn supervise(mut model: Option<String>, port: Option<u16>, state: &Path) -> Result<i32, String> {
    let bins = state.join("bin");
    std::fs::create_dir_all(&bins).map_err(|e| format!("{}: {e}", bins.display()))?;
    let state = state.canonicalize().map_err(|e| e.to_string())?;
    let port = choose_port(port, &state.join("port"))?;

    // Run a copy, so a `cargo build` overwriting target/ never touches a
    // binary we may have to fall back to.
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut good = state.join("bin").join(format!("rig-pi-{}", millis()));
    std::fs::copy(&exe, &good).map_err(|e| format!("{}: {e}", exe.display()))?;
    let mut next = good.clone();
    let mut note: Option<String> = None;
    let mut crashes = 0;
    let ready = state.join("ready");
    let log_path = state.join("agent.log");

    loop {
        let _ = std::fs::remove_file(&ready);
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| e.to_string())?;
        let log_start = log.metadata().map(|m| m.len()).unwrap_or(0);
        let mut command = Command::new(&next);
        command
            .arg("--child")
            .env("RIG_PI_STATE", &state)
            .env("RIG_PI_PORT", port.to_string())
            .env("RIG_PI_READY", &ready)
            .env("RIG_PI_NOTE", note.take().unwrap_or_default())
            .env("RIG_PI_MODEL", model.take().unwrap_or_default())
            .stderr(log);

        let started = Instant::now();
        let mut up = false;
        let outcome = match command.spawn() {
            Err(error) => Err(format!("could not be started: {error}")),
            Ok(mut child) => loop {
                if !up && ready.exists() {
                    up = true;
                    good = next.clone();
                }
                match child.try_wait() {
                    Ok(Some(status)) => break Ok(status),
                    Ok(None) => {}
                    Err(error) => break Err(error.to_string()),
                }
                if !up && started.elapsed() > STARTUP_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(format!(
                        "did not start within {}s",
                        STARTUP_TIMEOUT.as_secs()
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            },
        };

        if let Ok(status) = &outcome {
            match status.code() {
                Some(0) => return Ok(0),
                Some(RELOAD_EXIT) => {
                    good = next.clone();
                    crashes = 0;
                    let path = std::fs::read_to_string(state.join("next-binary"))
                        .map_err(|e| format!("reload without a next binary: {e}"))?;
                    next = PathBuf::from(path.trim());
                    note = Some(format!(
                        "Reload succeeded: rebuilt and restarted into {}. The conversation continues.",
                        next.display()
                    ));
                    continue;
                }
                _ => {}
            }
        }

        // The child crashed or never came up.
        crate::tui::restore();
        let what = match &outcome {
            Ok(status) => describe(status),
            Err(error) => error.clone(),
        };
        let tail = log_tail(&log_path, log_start);
        if !up && next == good {
            return Err(format!("the agent failed to start: {what}\n{tail}"));
        }
        if up {
            crashes += 1;
            if crashes > 3 {
                return Err(format!("the agent keeps crashing: {what}\n{tail}"));
            }
            note = Some(format!(
                "The agent crashed ({what}) and was restarted. Its stderr ended with:\n{tail}"
            ));
        } else {
            note = Some(format!(
                "Reload failed: the new binary {} {what} before it was up. Fell back to the last \
                 good binary {}. Its stderr ended with:\n{tail}",
                next.display(),
                good.display()
            ));
            next = good.clone();
        }
    }
}

fn describe(status: &ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("exited with code {code}"),
        None => format!("was killed ({status})"),
    }
}

/// The saved port if it is still free, else `requested`, else any free one.
fn choose_port(requested: Option<u16>, file: &Path) -> Result<u16, String> {
    let saved = std::fs::read_to_string(file)
        .ok()
        .and_then(|port| port.trim().parse::<u16>().ok());
    let port = requested
        .or(saved.filter(|port| TcpListener::bind(("127.0.0.1", *port)).is_ok()))
        .map(Ok)
        .unwrap_or_else(|| {
            TcpListener::bind(("127.0.0.1", 0))
                .and_then(|listener| listener.local_addr())
                .map(|addr| addr.port())
                .map_err(|e| e.to_string())
        })?;
    std::fs::write(file, port.to_string()).map_err(|e| e.to_string())?;
    Ok(port)
}

fn millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}

/// The last lines the failed child wrote to stderr.
fn log_tail(path: &Path, from: u64) -> String {
    let mut text = String::new();
    if let Ok(mut file) = std::fs::File::open(path) {
        let _ = file.seek(SeekFrom::Start(from));
        let _ = file.read_to_string(&mut text);
    }
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(30)..].join("\n")
}

/// Tells the supervisor the agent is up: a few frames in, so a panic in a
/// plugin's build or startup systems counts as a failed start.
pub struct ReadyPlugin;

impl Plugin for ReadyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Last, ready);
    }
}

fn ready(mut frames: Local<u32>, env: Res<Env>) {
    *frames += 1;
    if *frames == 10
        && let Some(path) = &env.ready_file
    {
        let _ = std::fs::write(path, std::process::id().to_string());
    }
}
