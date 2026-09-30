use std::io::Read;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use super::truncate;
use crate::agent::{AddTool, ToolReply, arguments};

pub struct BashPlugin;

impl Plugin for BashPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(
            "bash",
            "Run a bash command in the working directory. Returns stdout and stderr \
             combined, and the exit code. `timeout` is in seconds (default 120).",
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"}
                },
                "required": ["command"]
            }),
            bash,
        );
    }
}

#[derive(Deserialize)]
struct Args {
    command: String,
    timeout: Option<u64>,
}

fn bash(In(args): In<Value>) -> ToolReply {
    let args: Args = match arguments(args) {
        Ok(args) => args,
        Err(reply) => return reply,
    };
    ToolReply::spawn(move || run(&args.command, Duration::from_secs(args.timeout.unwrap_or(120))))
}

fn run(command: &str, timeout: Duration) -> String {
    // Both streams share one pipe so their output interleaves as it happened.
    let script = format!("exec 2>&1\n{command}");
    let mut child = match Command::new("bash")
        .args(["-c", &script])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(child) => child,
        Err(error) => return format!("error: cannot run bash: {error}"),
    };
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
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() > timeout => break None,
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(error) => return format!("error: {error}"),
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
    let output = output.lock().map(|output| output.clone()).unwrap_or_default();
    let mut text = truncate(&String::from_utf8_lossy(&output), 50_000);
    match status {
        None => text.push_str(&format!("\n[timed out after {}s]", timeout.as_secs())),
        Some(status) => match status.code() {
            Some(0) => {}
            Some(code) => text.push_str(&format!("\n[exit code {code}]")),
            None => text.push_str("\n[killed by a signal]"),
        },
    }
    text
}
