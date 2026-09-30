use std::io::Read;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use rig_core::tool::{DynamicTool, ToolExecutionError, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{MAX_BYTES, MAX_LINES, arguments};
use crate::glue::AddTool;

pub struct BashPlugin;

impl Plugin for BashPlugin {
    fn build(&self, app: &mut App) {
        app.add_rig_tool(DynamicTool::new(
            "bash",
            format!(
                "Execute a bash command in the current working directory. Returns stdout and \
                 stderr. Output is truncated to last {MAX_LINES} lines or {}KB (whichever is hit \
                 first). If truncated, full output is saved to a temp file. Optionally provide a \
                 timeout in seconds.",
                MAX_BYTES / 1024
            ),
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}
                },
                "required": ["command"]
            }),
            |args: Value| {
                Box::pin(async move {
                    let args: Args = arguments(args)?;
                    tokio::task::spawn_blocking(move || run(&args.command, args.timeout))
                        .await
                        .map_err(ToolExecutionError::from_error)?
                })
            },
        ));
    }
}

#[derive(Deserialize)]
struct Args {
    command: String,
    timeout: Option<f64>,
}

fn run(command: &str, timeout: Option<f64>) -> Result<ToolOutput, ToolExecutionError> {
    // Both streams share one pipe so their output interleaves as it happened.
    let script = format!("exec 2>&1\n{command}");
    let mut child = Command::new("bash")
        .args(["-c", &script])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(ToolExecutionError::from_error)?;
    let output = Arc::new(Mutex::new(Vec::new()));
    let (done, eof) = std::sync::mpsc::channel::<()>();
    if let Some(mut stdout) = child.stdout.take() {
        let output = output.clone();
        std::thread::spawn(move || {
            let mut chunk = [0; 8192];
            while let Ok(read @ 1..) = stdout.read(&mut chunk) {
                if let Ok(mut output) = output.lock() {
                    output.extend_from_slice(&chunk[..read]);
                }
            }
            let _ = done.send(());
        });
    }
    let started = Instant::now();
    let deadline = timeout.map(Duration::from_secs_f64);
    let status = loop {
        match child.try_wait().map_err(ToolExecutionError::from_error)? {
            Some(status) => break Some(status),
            None if deadline.is_some_and(|deadline| started.elapsed() > deadline) => break None,
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    };
    if status.is_none() {
        // The whole process group, so children of the command stop too.
        let _ = Command::new("kill")
            .args(["-KILL", &format!("-{}", child.id())])
            .status();
        let _ = child.wait();
    }
    // Background processes can hold the pipe open; do not wait for them.
    let _ = eof.recv_timeout(Duration::from_millis(500));
    let bytes = output
        .lock()
        .map(|output| output.clone())
        .unwrap_or_default();
    let mut text = tail(&String::from_utf8_lossy(&bytes));
    let status = match status {
        None => Some(format!(
            "Command timed out after {} seconds",
            timeout.unwrap_or_default()
        )),
        Some(status) => match status.code() {
            Some(0) => None,
            Some(code) => Some(format!("Command exited with code {code}")),
            None => Some("Command terminated without an exit code".into()),
        },
    };
    if let Some(status) = status {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(&status);
    } else if text.is_empty() {
        text.push_str("(no output)");
    }
    Ok(ToolOutput::text(text))
}

/// The last lines of `output` within the limits, with the whole output
/// saved to a temp file when it is cut.
fn tail(output: &str) -> String {
    let lines: Vec<&str> = output.trim_end_matches('\n').split('\n').collect();
    let mut kept = 0;
    let mut bytes = 0;
    for line in lines.iter().rev() {
        if kept == MAX_LINES || bytes + line.len() + 1 > MAX_BYTES {
            break;
        }
        kept += 1;
        bytes += line.len() + 1;
    }
    if kept == lines.len() {
        return lines.join("\n");
    }
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let path = std::env::temp_dir().join(format!(
        "bevy-agent-bash-{}-{}.log",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::write(&path, output);
    let start = lines.len() - kept;
    format!(
        "{}\n\n[Showing lines {}-{} of {}. Full output: {}]",
        lines[start..].join("\n"),
        start + 1,
        lines.len(),
        lines.len(),
        path.display()
    )
}
