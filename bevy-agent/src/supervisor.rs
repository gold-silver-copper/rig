//! The process the user starts. It runs the agent as a child and restarts it:
//! into `bin/candidate` after a successful reload, and back to the last binary
//! that started successfully (`bin/good`, then `bin/prev`) when one crashes.
//!
//! bin/candidate  built by a reload, not yet started
//! bin/good       the last binary that started (the child promotes itself)
//! bin/prev       the good binary before that

use std::fs;
use std::io::Write;
use std::os::unix::process::ExitStatusExt;
use std::process::{Command, ExitCode, ExitStatus};

use ratatui::crossterm::{cursor, execute, terminal};

use crate::reload::RELOAD_EXIT;
use crate::session::Paths;

pub fn run(paths: &Paths, args: &[String]) -> ExitCode {
    if let Err(error) = install(paths) {
        eprintln!("bevy-agent: cannot set up {}: {error}", paths.home.display());
        return ExitCode::FAILURE;
    }
    for generation in 0u64.. {
        let candidate = paths.bin("candidate");
        let trying = candidate.exists();
        let binary = if trying { candidate.clone() } else { paths.bin("good") };
        let mut command = Command::new(&binary);
        command
            .env("BEVY_AGENT_CHILD", "1")
            .env("BEVY_AGENT_GENERATION", generation.to_string())
            .env("BEVY_AGENT_HOME", &paths.home)
            .env("BEVY_AGENT_SRC", &paths.source)
            .env_remove("BEVY_AGENT_CANDIDATE");
        // Startup flags apply to the first start only; later starts resume the session.
        if generation == 0 {
            command.args(args);
        }
        if trying {
            command.env("BEVY_AGENT_CANDIDATE", "1");
        }
        if let Ok(log) = fs::File::create(paths.child_stderr()) {
            command.stderr(log);
        }
        let status = command.status();
        let why = match status {
            Ok(status) if status.success() => return ExitCode::SUCCESS,
            Ok(status) if status.code() == Some(RELOAD_EXIT.into()) => continue,
            Ok(status) => describe(status, paths),
            Err(error) => format!("could not start {}: {error}", binary.display()),
        };
        reset_terminal();
        let note = if trying && candidate.exists() {
            let _ = fs::remove_file(&candidate);
            format!("the new binary failed to start ({why}); fell back to the last good binary")
        } else if paths.bin("prev").exists() && fs::rename(paths.bin("prev"), paths.bin("good")).is_ok() {
            format!("the binary crashed ({why}); rolled back to the previous good binary")
        } else {
            eprintln!("bevy-agent: {why}");
            return ExitCode::FAILURE;
        };
        let _ = fs::write(paths.fallback_note(), note);
    }
    ExitCode::FAILURE
}

/// The launched binary is the first good one; stale state from an earlier run goes.
fn install(paths: &Paths) -> std::io::Result<()> {
    fs::create_dir_all(paths.home.join("bin"))?;
    for stale in ["candidate", "prev"] {
        let _ = fs::remove_file(paths.bin(stale));
    }
    let _ = fs::remove_file(paths.fallback_note());
    let tmp = paths.bin("good.tmp");
    fs::copy(std::env::current_exe()?, &tmp)?;
    fs::rename(tmp, paths.bin("good"))
}

fn describe(status: ExitStatus, paths: &Paths) -> String {
    let how = match (status.code(), status.signal()) {
        (Some(code), _) => format!("exit code {code}"),
        (None, Some(signal)) => format!("signal {signal}"),
        _ => status.to_string(),
    };
    let stderr = fs::read_to_string(paths.child_stderr()).unwrap_or_default();
    let tail: Vec<&str> = stderr.lines().rev().take(15).collect();
    let tail: Vec<&str> = tail.into_iter().rev().collect();
    match tail.is_empty() {
        true => how,
        false => format!("{how}; stderr:\n{}", tail.join("\n")),
    }
}

/// A child that died hard may leave the terminal raw and on the alternate screen.
fn reset_terminal() {
    let _ = terminal::disable_raw_mode();
    let mut stdout = std::io::stdout();
    let _ = execute!(stdout, terminal::LeaveAlternateScreen, cursor::Show);
    let _ = stdout.flush();
}
