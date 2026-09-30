//! The four coding tools, as in pi: read, write, edit, bash.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use serde_json::{Value, json};

use crate::tools::AgentAppExt;

const MAX_OUTPUT: usize = 30_000;

pub struct CodingToolsPlugin;

impl Plugin for CodingToolsPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(
            "read",
            "Read a text file. Returns numbered lines; use offset/limit for large files.",
            json!({"type": "object", "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "description": "1-based first line"},
                "limit": {"type": "integer", "description": "maximum number of lines"}
            }, "required": ["path"]}),
            read,
        )
        .add_tool(
            "write",
            "Create or overwrite a file with the given content, creating parent directories.",
            json!({"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}
            }, "required": ["path", "content"]}),
            write,
        )
        .add_tool(
            "edit",
            "Replace exactly one occurrence of old_text with new_text in a file.",
            json!({"type": "object", "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"}
            }, "required": ["path", "old_text", "new_text"]}),
            edit,
        )
        .add_tool(
            "bash",
            "Run a shell command with `sh -c` in the working directory. Returns exit code, stdout and stderr.",
            json!({"type": "object", "properties": {
                "command": {"type": "string"},
                "timeout_secs": {"type": "integer", "description": "default 120"}
            }, "required": ["command"]}),
            bash,
        );
    }
}

fn arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string argument `{key}`"))
}

pub fn truncate(text: String) -> String {
    if text.len() <= MAX_OUTPUT {
        return text;
    }
    let mut cut = text.len() - MAX_OUTPUT;
    while !text.is_char_boundary(cut) {
        cut += 1;
    }
    format!("[... {cut} bytes truncated ...]\n{}", &text[cut..])
}

fn read(args: Value) -> Result<String, String> {
    let path = arg(&args, "path")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(1).max(1) as usize;
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(2000) as usize;
    let lines: Vec<String> = text
        .lines()
        .enumerate()
        .skip(offset - 1)
        .take(limit)
        .map(|(n, line)| format!("{:>5}\t{line}", n + 1))
        .collect();
    Ok(truncate(lines.join("\n")))
}

fn write(args: Value) -> Result<String, String> {
    let path = arg(&args, "path")?;
    let content = arg(&args, "content")?;
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, content).map_err(|e| format!("{path}: {e}"))?;
    Ok(format!("wrote {} bytes to {path}", content.len()))
}

fn edit(args: Value) -> Result<String, String> {
    let path = arg(&args, "path")?;
    let (old, new) = (arg(&args, "old_text")?, arg(&args, "new_text")?);
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    match text.matches(old).count() {
        1 => {
            std::fs::write(path, text.replacen(old, new, 1)).map_err(|e| e.to_string())?;
            Ok(format!("edited {path}"))
        }
        0 => Err(format!("old_text not found in {path}")),
        n => Err(format!("old_text occurs {n} times in {path}; make it unique")),
    }
}

fn bash(args: Value) -> Result<String, String> {
    let command = arg(&args, "command")?;
    let timeout = Duration::from_secs(args.get("timeout_secs").and_then(Value::as_u64).unwrap_or(120));
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    // Drain both pipes on threads so a chatty command cannot block on a full
    // pipe; a background process that keeps a pipe open is waited on only briefly.
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut out = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut out);
            }
            let _ = tx.send(String::from_utf8_lossy(&out).into_owned());
        });
        rx
    };
    let stdout = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let stderr = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break status.code().map_or("killed by signal".to_owned(), |c| c.to_string());
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            break format!("timed out after {}s", timeout.as_secs());
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let collect = |rx: std::sync::mpsc::Receiver<String>| {
        rx.recv_timeout(Duration::from_secs(2))
            .unwrap_or_else(|_| "[output still open by a background process]".to_owned())
    };
    let (stdout, stderr) = (collect(stdout), collect(stderr));
    Ok(truncate(format!("exit: {status}\nstdout:\n{stdout}\nstderr:\n{stderr}")))
}
