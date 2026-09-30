use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::{clip, str_arg};
use crate::tools::Native;

pub fn tool() -> Native {
    Native {
        name: "bash",
        description: "Run a shell command in the working directory. Returns stdout, stderr, and the exit code.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" },
                "timeout": { "type": "integer", "description": "Seconds (default 120)" }
            },
            "required": ["command"]
        }),
        run,
    }
}

fn run(args: Value) -> Result<String, String> {
    let command = str_arg(&args, "command")?;
    let timeout = Duration::from_secs(args.get("timeout").and_then(Value::as_u64).unwrap_or(120));
    let mut child = Command::new("bash")
        .args(["-c", command])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| error.to_string())?;
    let mut stdout = child.stdout.take().ok_or("no stdout")?;
    let mut stderr = child.stderr.take().ok_or("no stderr")?;
    let out = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stdout.read_to_string(&mut text);
        text
    });
    let err = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            break Some(status);
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut text = out.join().unwrap_or_default();
    let stderr = err.join().unwrap_or_default();
    if !stderr.is_empty() {
        text.push_str(&format!("\n[stderr]\n{stderr}"));
    }
    match status {
        Some(status) => text.push_str(&format!("\n[exit {}]", status.code().unwrap_or(-1))),
        None => text.push_str(&format!("\n[killed after {}s]", timeout.as_secs())),
    }
    Ok(clip(text, 30_000))
}
