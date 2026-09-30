//! The process the user starts. It runs the agent as a child process from a
//! private copy of a binary, starts the new binary when the agent exits to
//! reload, and falls back to the last binary that started when a new one
//! crashes before it finishes starting. The files it shares with the agent
//! live in the state directory.

use std::fs;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use rig_core::message::ToolResultContent;

use crate::session::{Entry, Session, StateDir, Turn};
use crate::{CHILD_ENV, GENERATION_ENV};

/// The exit code with which the agent asks to be restarted into `next-binary`.
pub const RELOAD_CODE: u8 = 75;
pub const STATE_ENV: &str = "BEVY_AGENT_STATE";
const STARTED: &str = "started";
const NEXT: &str = "next-binary";
const NOTE: &str = "note.txt";
const BACKUP: &str = "session.before-reload.json";
const STDERR: &str = "agent.stderr.log";

pub fn run(state: &Path) -> ExitCode {
    match supervise(state) {
        Ok(code) => code,
        Err(error) => {
            ratatui::restore();
            eprintln!("bevy-agent: {error}");
            ExitCode::FAILURE
        }
    }
}

fn supervise(state: &Path) -> std::io::Result<ExitCode> {
    fs::create_dir_all(state.join("bin"))?;
    let mut good = install(&std::env::current_exe()?, state)?;
    let mut current = good.clone();
    let mut generation = 0u32;
    let mut quick_crashes = 0;
    let code = loop {
        let _ = fs::remove_file(state.join(STARTED));
        let _ = fs::remove_file(state.join(NEXT));
        let started_at = Instant::now();
        let status = Command::new(&current)
            .args(std::env::args().skip(1).filter(|_| generation == 0))
            .env(CHILD_ENV, "1")
            .env(GENERATION_ENV, generation.to_string())
            .env(STATE_ENV, state)
            .stderr(fs::File::create(state.join(STDERR))?)
            .status();
        let started = fs::read_to_string(state.join(STARTED)).is_ok();
        generation += 1;
        if let Ok(status) = &status {
            match status.code() {
                Some(0) => break ExitCode::SUCCESS,
                Some(code) if code == i32::from(RELOAD_CODE) => {
                    if started && good != current {
                        let _ = fs::remove_file(&good);
                        good = current.clone();
                    }
                    if let Ok(next) = fs::read_to_string(state.join(NEXT)) {
                        let _ = fs::copy(state.join("session.json"), state.join(BACKUP));
                        current = next.into();
                    }
                    continue;
                }
                _ => {}
            }
        }
        ratatui::restore();
        let what = match &status {
            Ok(status) => describe(status),
            Err(error) => format!("could not be started ({error})"),
        };
        let tail = tail(&state.join(STDERR));
        if !started && current != good {
            let _ = fs::copy(state.join(BACKUP), state.join("session.json"));
            fs::write(
                state.join(NOTE),
                format!(
                    "Reload failed: the new binary {what} before it finished starting, so the \
                     agent fell back to the previous binary, which is running now. Nothing \
                     changed. The new binary's stderr ended with:\n{tail}"
                ),
            )?;
            let _ = fs::remove_file(&current);
            current = good.clone();
            continue;
        }
        quick_crashes = if started_at.elapsed() < Duration::from_secs(30) {
            quick_crashes + 1
        } else {
            1
        };
        if !started || quick_crashes >= 3 {
            eprintln!("bevy-agent {what}. Its stderr ended with:\n{tail}");
            eprintln!("Resume the conversation with `bevy-agent --continue`.");
            break ExitCode::FAILURE;
        }
        fs::write(
            state.join(NOTE),
            format!("The agent {what} and was restarted. Its stderr ended with:\n{tail}"),
        )?;
    };
    let _ = fs::remove_file(&good);
    let _ = fs::remove_file(&current);
    Ok(code)
}

/// A private copy of `binary`, which later builds cannot overwrite.
pub fn install(binary: &Path, state: &Path) -> std::io::Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let copy = state.join("bin").join(format!("bevy-agent-{stamp}"));
    fs::copy(binary, &copy)?;
    Ok(copy)
}

fn describe(status: &ExitStatus) -> String {
    #[cfg(unix)]
    if let Some(signal) = std::os::unix::process::ExitStatusExt::signal(status) {
        return format!("was killed by signal {signal}");
    }
    match status.code() {
        Some(code) => format!("exited with code {code}"),
        None => "exited abnormally".into(),
    }
}

fn tail(path: &Path) -> String {
    let text = fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(30)..].join("\n")
}

/// Tells the supervisor this binary started, which makes it the fallback for
/// the next reload: it has run for a second and its BRP port answers. A
/// binary whose port never answers exits, so the supervisor falls back.
pub fn mark_started(
    time: Res<Time<Real>>,
    dir: Res<StateDir>,
    session: Res<Session>,
    mut done: Local<bool>,
    mut exit: MessageWriter<AppExit>,
) {
    if *done || time.elapsed() < Duration::from_secs(1) {
        return;
    }
    if TcpStream::connect(("127.0.0.1", session.brp_port)).is_ok() {
        *done = true;
        let _ = fs::write(dir.0.join(STARTED), std::process::id().to_string());
    } else if time.elapsed() > Duration::from_secs(10) {
        eprintln!("the BRP port {} is not listening", session.brp_port);
        exit.write(AppExit::error());
    }
}

/// Settles a restart: answers the tool call the previous process was
/// running, which is `reload` unless it crashed, and passes on what the
/// supervisor noted about the restart.
pub fn resume(session: &mut Session, state: &StateDir, generation: u32) {
    let note = fs::read_to_string(state.0.join(NOTE)).ok();
    let _ = fs::remove_file(state.0.join(NOTE));
    if let Turn::Tools { calls, results } = &mut session.turn
        && let Some(call) = calls.get(results.len())
    {
        let text = match (&note, call.function.name.as_str()) {
            (Some(note), _) => note.clone(),
            (None, "reload") => {
                "Reloaded: the new build is running now, with this conversation intact.".into()
            }
            (None, _) => "Interrupted: the agent restarted before this tool finished.".into(),
        };
        results.push(call.result(vec![ToolResultContent::text(text.clone())]));
        session.log(Entry::Output(text));
    } else if let Some(note) = note {
        session.log(Entry::Notice(note.clone()));
        session.notes.push(note);
    } else if generation > 0 {
        session.log(Entry::Notice("Reloaded into the new build.".into()));
    }
}
