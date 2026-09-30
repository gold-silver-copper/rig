//! The supervisor: runs the agent as a child process and restarts it.
//!
//! Exit code [`RELOAD_EXIT`] means "restart into `<state>/next-binary`". A
//! child counts as started once it creates its ready file; one that exits
//! before that is replaced by the last binary that did start, and the agent
//! is told why through `RIG_PI_NOTE`. Once a new binary is up, the
//! supervisor `exec`s into it too (`--adopt`), so a reload also updates this
//! file's code; the running child stays its child across the `exec`.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom};
use std::net::TcpListener;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bevy::prelude::*;

use crate::{Env, Options, RELOAD_EXIT, USAGE};

/// How long a new binary gets to come up before it counts as failed.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

pub fn run(args: &[String]) -> i32 {
    let mut extra = Vec::new();
    let options = match Options::parse(args, &mut extra) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            return 2;
        }
    };
    let (mut port, mut adopt, mut state) = (None, None, PathBuf::from(".rig-pi"));
    for (flag, value) in extra {
        match flag.as_str() {
            "--port" => port = value.parse().ok(),
            "--state" => state = PathBuf::from(value),
            "--adopt" => adopt = value.parse().ok(),
            _ => {
                eprintln!("unknown option {flag}\n{USAGE}");
                return 2;
            }
        }
    }
    match supervise(&options, port, &state, adopt) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("rig-pi: {error}");
            1
        }
    }
}

/// The agent process: spawned by this supervisor, or by the one it `exec`ed
/// from.
enum Agent {
    Spawned(Child),
    Adopted(i32),
}

impl Agent {
    fn id(&self) -> i32 {
        match self {
            Agent::Spawned(child) => child.id() as i32,
            Agent::Adopted(pid) => *pid,
        }
    }

    fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        match self {
            Agent::Spawned(child) => child.try_wait(),
            Agent::Adopted(pid) => {
                let mut status = 0;
                // SAFETY: waitpid on our own child with a valid out-pointer.
                match unsafe { libc::waitpid(*pid, &mut status, libc::WNOHANG) } {
                    0 => Ok(None),
                    -1 => Err(std::io::Error::last_os_error()),
                    _ => Ok(Some(ExitStatus::from_raw(status))),
                }
            }
        }
    }

    fn kill(&mut self) {
        // SAFETY: signalling our own child by pid.
        unsafe { libc::kill(self.id(), libc::SIGKILL) };
        while let Ok(None) = self.try_wait() {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn supervise(
    options: &Options,
    port: Option<u16>,
    state: &Path,
    adopt: Option<i32>,
) -> Result<i32, String> {
    let mut launch = adopt.is_none();
    let bins = state.join("bin");
    std::fs::create_dir_all(&bins).map_err(|e| format!("{}: {e}", bins.display()))?;
    let state = state.canonicalize().map_err(|e| e.to_string())?;
    let bins = state.join("bin");
    let port = choose_port(port, &state.join("port"), adopt.is_some())?;
    let mut me = std::env::current_exe().map_err(|e| e.to_string())?;

    // Agents run from copies in <state>/bin, so a `cargo build` overwriting
    // target/ never touches a binary we may have to fall back to.
    let mut good = if adopt.is_some() {
        me.clone()
    } else {
        let copy = bins.join(format!("rig-pi-{}", millis()));
        std::fs::copy(&me, &copy).map_err(|e| format!("{}: {e}", me.display()))?;
        copy
    };
    let mut next = good.clone();
    let mut agent = adopt.map(Agent::Adopted);
    let mut note: Option<String> = None;
    let mut crashes = 0;
    let ready = state.join("ready");
    let log_path = state.join("agent.log");

    loop {
        let log_start = std::fs::metadata(&log_path).map(|m| m.len()).unwrap_or(0);
        let mut up = agent.is_some();
        let started = Instant::now();
        let spawned = match agent.take() {
            Some(adopted) => Ok(adopted),
            None => {
                let _ = std::fs::remove_file(&ready);
                let log = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_path)
                    .map_err(|e| e.to_string())?;
                let args = options.to_args(launch);
                let restarted = if launch { "" } else { "1" };
                launch = false;
                Command::new(&next)
                    .arg("--child")
                    .args(args)
                    .env("RIG_PI_STATE", &state)
                    .env("RIG_PI_PORT", port.to_string())
                    .env("RIG_PI_READY", &ready)
                    .env("RIG_PI_NOTE", note.take().unwrap_or_default())
                    .env("RIG_PI_RESTARTED", restarted)
                    .stderr(log)
                    .spawn()
                    .map(Agent::Spawned)
                    .map_err(|error| format!("could not be started: {error}"))
            }
        };
        let outcome = match spawned {
            Err(error) => Err(error),
            Ok(mut agent) => loop {
                if !up && ready.exists() {
                    up = true;
                    good = next.clone();
                    prune(&bins, &good);
                }
                if up && good != me {
                    // Only returns on failure; then keep supervising as we are.
                    let error = Command::new(&good)
                        .args(options.to_args(false))
                        .args(["--state", &state.to_string_lossy()])
                        .args(["--port", &port.to_string()])
                        .args(["--adopt", &agent.id().to_string()])
                        .exec();
                    eprintln!("rig-pi: could not hand over to {}: {error}", good.display());
                    me = good.clone();
                }
                match agent.try_wait() {
                    Ok(Some(status)) => break Ok(status),
                    Ok(None) => {}
                    Err(error) => break Err(error.to_string()),
                }
                if !up && started.elapsed() > STARTUP_TIMEOUT {
                    agent.kill();
                    break Err(format!("did not start within {}s", STARTUP_TIMEOUT.as_secs()));
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

        // The agent crashed or never came up.
        crate::tui::restore();
        let what = match &outcome {
            Ok(status) => describe(status),
            Err(error) => error.clone(),
        };
        let tail = match log_tail(&log_path, log_start) {
            tail if tail.trim().is_empty() => String::new(),
            tail => format!(" Its stderr ended with:\n{tail}"),
        };
        if !up && next == good {
            return Err(format!("the agent failed to start: {what}.{tail}"));
        }
        if up {
            crashes += 1;
            if crashes > 3 {
                return Err(format!("the agent keeps crashing: {what}.{tail}"));
            }
            note = Some(format!(
                "The agent crashed ({what}) and was restarted.{tail}"
            ));
        } else {
            note = Some(format!(
                "Reload failed: the new binary {} {what} before it was up. Fell back to the last \
                 good binary {}.{tail}",
                next.display(),
                good.display()
            ));
            next = good.clone();
            prune(&bins, &good);
        }
    }
}

fn describe(status: &ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("exited with code {code}"),
        None => format!("was killed ({status})"),
    }
}

/// `requested`, else the saved port if it is still free, else any free
/// one. An adopting supervisor keeps the port its agent is serving on.
fn choose_port(requested: Option<u16>, file: &Path, adopting: bool) -> Result<u16, String> {
    let saved = std::fs::read_to_string(file)
        .ok()
        .and_then(|port| port.trim().parse::<u16>().ok());
    let saved = saved.filter(|port| adopting || TcpListener::bind(("127.0.0.1", *port)).is_ok());
    let port = match requested.or(saved) {
        Some(port) => port,
        None => TcpListener::bind(("127.0.0.1", 0))
            .and_then(|listener| listener.local_addr())
            .map(|addr| addr.port())
            .map_err(|e| e.to_string())?,
    };
    std::fs::write(file, port.to_string()).map_err(|e| e.to_string())?;
    Ok(port)
}

/// Delete every binary but `keep`; each is a full debug build.
fn prune(bins: &Path, keep: &Path) {
    let Ok(entries) = std::fs::read_dir(bins) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.path() != keep {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}

/// The last lines the failed agent wrote to stderr.
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
