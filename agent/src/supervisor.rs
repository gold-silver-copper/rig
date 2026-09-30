//! The parent process: starts children, restarts them into rebuilt binaries,
//! and falls back to the last good binary when a new one does not start.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::session::Paths;
use crate::{Options, RELOAD_EXIT_CODE, ReloadStatus};

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
        None => std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port(),
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

        let mut child = command.spawn()?;
        let mut ready = false;
        let status = loop {
            if !ready && paths.ready().exists() {
                ready = true;
            }
            if let Some(status) = child.try_wait()? {
                break status;
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        ready |= paths.ready().exists();
        if ready {
            last_good = next.clone();
        }

        // Every later child continues this session.
        resume = true;
        match status.code() {
            Some(0) => return Ok(ExitCode::SUCCESS),
            Some(code) if code == i32::from(RELOAD_EXIT_CODE) => {
                let request: ReloadRequest = serde_json::from_slice(&fs::read(paths.reload())?)?;
                next = request.binary;
                reload = Some(ReloadStatus::Succeeded);
            }
            _ if !ready && next != last_good => {
                let reason = format!(
                    "exited ({status}) before it finished starting up. Its stderr ended with:\n{}",
                    log_tail(&paths.log(), log_start)
                );
                next = last_good.clone();
                reload = Some(ReloadStatus::Failed(reason));
            }
            _ => {
                eprintln!(
                    "bevy-agent: the agent exited ({status}). Its stderr ended with:\n{}",
                    log_tail(&paths.log(), log_start)
                );
                return Ok(ExitCode::FAILURE);
            }
        }
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
