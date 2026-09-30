//! Reload = rebuild the agent's own source with cargo, then exit so the
//! supervisor restarts into the new binary. A failed build changes nothing.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use bevy::prelude::*;
use serde_json::{Value, json};

use crate::session::Paths;
use crate::tools::{Handler, Slot, Tools, definition, fill, slot};

/// Exit code that asks the supervisor to start `bin/candidate`.
pub const RELOAD_EXIT: u8 = 75;

pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Tools>();
        app.world_mut().resource_mut::<Tools>().insert(
            definition(
                "reload",
                "Rebuild your own source (cargo build) and restart into the new binary, keeping \
                 the conversation. On a compile error you keep running and get the errors back. \
                 Call it alone, after your edits.",
                json!({"type": "object", "properties": {}}),
            ),
            Handler::Reload,
        );
    }
}

/// Build in the background: `Ok(built executable)` or `Err(compiler errors)`.
pub fn spawn_build(source: PathBuf) -> Slot<Result<PathBuf, String>> {
    let out = slot();
    let sink = out.clone();
    std::thread::spawn(move || fill(&sink, build(&source)));
    out
}

fn build(source: &Path) -> Result<PathBuf, String> {
    let start = Instant::now();
    let mut child = Command::new("cargo")
        .args(["build", "--message-format=json-render-diagnostics"])
        .current_dir(source)
        // Let the source tree's rust-toolchain.toml pick the compiler.
        .env_remove("RUSTUP_TOOLCHAIN")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    let mut stderr = child.stderr.take();
    let diagnostics = std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(pipe) = stderr.as_mut() {
            let _ = pipe.read_to_string(&mut text);
        }
        text
    });
    let mut executable = None;
    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(message) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if message["reason"] == "compiler-artifact"
                && message["target"]["name"] == env!("CARGO_PKG_NAME")
                && let Some(path) = message["executable"].as_str()
            {
                executable = Some(PathBuf::from(path));
            }
        }
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    let diagnostics = diagnostics.join().unwrap_or_default();
    match (status.success(), executable) {
        (true, Some(executable)) => Ok(executable),
        (true, None) => Err("cargo succeeded but reported no bevy-agent executable".to_owned()),
        (false, _) => Err(format!(
            "cargo build failed after {:.0}s:\n{}",
            start.elapsed().as_secs_f32(),
            errors_only(&diagnostics)
        )),
    }
}

/// Keep the `error` diagnostics, drop progress lines and warnings.
fn errors_only(stderr: &str) -> String {
    let mut out = String::new();
    let mut keep = false;
    const PROGRESS: &[&str] = &["Compiling ", "Checking ", "Finished ", "Building ", "Blocking ", "Locking ", "Updating ", "Download", "Adding "];
    for line in stderr.lines() {
        if PROGRESS.iter().any(|p| line.trim_start().starts_with(p)) {
            keep = false;
            continue;
        }
        // A diagnostic runs until the next one; keep errors, drop warnings.
        if line.starts_with("error") {
            keep = true;
        } else if line.starts_with("warning") || line.starts_with("For more information") {
            keep = false;
        }
        if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    if out.is_empty() {
        out = stderr.to_owned();
    }
    crate::plugins::coding::truncate(out)
}

/// Stage the built binary where the supervisor will look for it.
pub fn install_candidate(paths: &Paths, built: &Path) -> std::io::Result<()> {
    let candidate = paths.bin("candidate");
    let tmp = paths.bin("candidate.tmp");
    std::fs::copy(built, &tmp)?;
    std::fs::rename(tmp, candidate)
}

/// Called once the new binary is up: it becomes the last known good one.
pub fn promote(paths: &Paths) {
    let candidate = paths.bin("candidate");
    if std::env::var_os("BEVY_AGENT_CANDIDATE").is_none() || !candidate.exists() {
        return;
    }
    let _ = std::fs::rename(paths.bin("good"), paths.bin("prev"));
    let _ = std::fs::rename(candidate, paths.bin("good"));
}

#[cfg(test)]
mod tests {
    #[test]
    fn errors_keep_their_source_lines_and_drop_warnings() {
        let stderr = "   Compiling bevy-agent v0.1.0\nwarning: unused variable: `x`\n --> src/a.rs:1:5\n  |\n1 | let x = 1;\n\nerror[E0308]: mismatched types\n --> src/b.rs:2:9\n  |\n2 | let y: u32 = \"s\";\n  |        ---   ^^^ expected `u32`, found `&str`\n\nerror: could not compile `bevy-agent`\n";
        let errors = super::errors_only(stderr);
        assert!(errors.contains("expected `u32`, found `&str`"), "{errors}");
        assert!(errors.contains("could not compile"));
        assert!(!errors.contains("unused variable") && !errors.contains("Compiling"));
    }
}
