//! The parent process: starts children, restarts them into rebuilt binaries,
//! and falls back to the last good binary when a new one does not start.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, ExitStatus, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::session::Paths;
use crate::{Options, RELOAD_EXIT_CODE, ReloadStatus};

/// How long a new binary may take to open its TUI and BRP server.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

/// `reload.json`: the binary a reloading child asks to be started next.
#[derive(Serialize, Deserialize)]
pub struct ReloadRequest {
    pub binary: PathBuf,
}

pub fn run(options: Options) -> ExitCode {
    match supervise(options) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("bevy-agent: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn supervise(options: Options) -> anyhow::Result<ExitCode> {
    fs::create_dir_all(options.state_dir().join("bin"))?;
    let paths = Paths::new(fs::canonicalize(options.state_dir())?);
    // Chosen once, so every child binds the same port.
    let port = match options.brp_port {
        Some(port) => port,
        None => std::net::TcpListener::bind("127.0.0.1:0")?
            .local_addr()?
            .port(),
    };
    // Run a copy: a rebuild replaces the file cargo wrote, never this copy.
    let mut last_good = stash_binary(&paths, &std::env::current_exe()?)?;
    let mut next = last_good.clone();
    let mut resume = options.resume;
    let mut reload: Option<ReloadStatus> = None;
    // Only the first child takes `--model`; later ones keep the session's,
    // which may have been switched from the TUI since.
    let mut model = options.model.clone();

    loop {
        prune_binaries(&paths, &[&last_good, &next]);
        let _ = fs::remove_file(paths.ready());
        let _ = fs::remove_file(paths.reload());
        let log_start = fs::metadata(paths.log()).map(|m| m.len()).unwrap_or(0);
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(paths.log())?;

        let mut command = Command::new(&next);
        command
            .arg("--child")
            .arg("--state-dir")
            .arg(&paths.dir)
            .arg("--brp-port")
            .arg(port.to_string())
            .stderr(Stdio::from(log));
        if let Some(model) = model.take() {
            command.arg("--model").arg(model);
        }
        if resume {
            command.arg("--resume");
        }
        match &reload {
            Some(ReloadStatus::Succeeded) => {
                command.arg("--reload-ok");
            }
            Some(ReloadStatus::Failed(reason)) => {
                command.arg("--reload-failed").arg(reason);
            }
            None => {}
        }

        let outcome = watch(command.spawn(), &paths)?;
        // Every later child continues this session.
        resume = true;
        let exited = match outcome {
            Outcome::Exited { status, ready } => {
                if ready {
                    last_good = next.clone();
                }
                if ready || status.success() {
                    Ok(status)
                } else {
                    Err(format!("exited ({status}) before it finished starting up"))
                }
            }
            Outcome::NotStarted(reason) => Err(reason),
        };
        let tail = || log_tail(&paths.log(), log_start);
        match exited {
            Err(reason) if next != last_good => {
                let reason = format!("{reason}. Its stderr ended with:\n{}", tail());
                next = last_good.clone();
                reload = Some(ReloadStatus::Failed(reason));
            }
            Ok(status) if status.success() => return Ok(ExitCode::SUCCESS),
            Ok(status) if status.code() == Some(i32::from(RELOAD_EXIT_CODE)) => {
                let request: ReloadRequest = serde_json::from_slice(&fs::read(paths.reload())?)?;
                next = request.binary;
                reload = Some(ReloadStatus::Succeeded);
            }
            Ok(status) => {
                eprintln!(
                    "bevy-agent: the agent exited ({status}). Its stderr ended with:\n{}",
                    tail()
                );
                return Ok(ExitCode::FAILURE);
            }
            Err(reason) => {
                eprintln!(
                    "bevy-agent: the agent {reason}. Its stderr ended with:\n{}",
                    tail()
                );
                return Ok(ExitCode::FAILURE);
            }
        }
    }
}

/// How a child ended.
enum Outcome {
    /// It ran and exited; `ready` if it had finished starting up.
    Exited { status: ExitStatus, ready: bool },
    /// It could not be started, or hung while starting and was stopped.
    NotStarted(String),
}

/// Wait for a child to exit, noting whether it became ready. One that does
/// not become ready within [`STARTUP_TIMEOUT`] is killed.
fn watch(spawned: std::io::Result<Child>, paths: &Paths) -> anyhow::Result<Outcome> {
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            return Ok(Outcome::NotStarted(format!(
                "could not be started: {error}"
            )));
        }
    };
    let started = Instant::now();
    let mut ready = false;
    loop {
        ready |= paths.ready().exists();
        if let Some(status) = child.try_wait()? {
            ready |= paths.ready().exists();
            return Ok(Outcome::Exited { status, ready });
        }
        if !ready && started.elapsed() > STARTUP_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(Outcome::NotStarted(format!(
                "did not finish starting up within {}s and was stopped",
                STARTUP_TIMEOUT.as_secs()
            )));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Copy `binary` into the state directory under a unique name.
pub fn stash_binary(paths: &Paths, binary: &Path) -> anyhow::Result<PathBuf> {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let target = paths.bin().join(format!("bevy-agent-{stamp}"));
    fs::create_dir_all(paths.bin())?;
    fs::copy(binary, &target)?;
    Ok(target)
}

/// Delete stashed binaries other than `keep`: each is a full executable.
fn prune_binaries(paths: &Paths, keep: &[&PathBuf]) {
    let Ok(entries) = fs::read_dir(paths.bin()) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !keep.contains(&&path) {
            let _ = fs::remove_file(path);
        }
    }
}

/// What a child wrote to stderr since `start`, trimmed to its last lines.
fn log_tail(path: &Path, start: u64) -> String {
    let mut text = String::new();
    if let Ok(mut file) = fs::File::open(path) {
        let _ = file.seek(SeekFrom::Start(start));
        let _ = file.read_to_string(&mut text);
    }
    let lines: Vec<&str> = text.lines().collect();
    let tail = lines[lines.len().saturating_sub(20)..].join("\n");
    if tail.trim().is_empty() {
        "(nothing)".to_string()
    } else {
        tail
    }
}
