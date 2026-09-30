//! The coding tools: read, write, edit and bash, as in pi.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use bevy::prelude::*;
use serde_json::{Value, json};

use super::truncate;
use crate::agent::{AgentAppExt, CallOutput, ToolInvocation, ToolTasks};

const MAX_OUTPUT: usize = 30_000;

pub struct CodingToolsPlugin;

impl Plugin for CodingToolsPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(
            "read",
            "Read a text file. Optional `offset` (1-based first line) and `limit` (line count).",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "offset": { "type": "integer" },
                    "limit": { "type": "integer" }
                },
                "required": ["path"]
            }),
            |In(call): In<ToolInvocation>, mut commands: Commands| {
                let output = read(&call.call.function.arguments).unwrap_or_else(error);
                commands.entity(call.entity).insert(CallOutput(output));
            },
        )
        .add_tool(
            "write",
            "Create or overwrite a file with `content`, creating parent directories.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
            |In(call): In<ToolInvocation>, mut commands: Commands| {
                let output = write(&call.call.function.arguments).unwrap_or_else(error);
                commands.entity(call.entity).insert(CallOutput(output));
            },
        )
        .add_tool(
            "edit",
            "Replace `old_text`, which must occur exactly once in the file, with `new_text`.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_text": { "type": "string" },
                    "new_text": { "type": "string" }
                },
                "required": ["path", "old_text", "new_text"]
            }),
            |In(call): In<ToolInvocation>, mut commands: Commands| {
                let output = edit(&call.call.function.arguments).unwrap_or_else(error);
                commands.entity(call.entity).insert(CallOutput(output));
            },
        )
        .add_tool(
            "bash",
            "Run a shell command in the working directory; returns stdout, stderr and the exit \
             code. Optional `timeout` in seconds (default 120).",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "timeout": { "type": "integer" }
                },
                "required": ["command"]
            }),
            |In(call): In<ToolInvocation>, tasks: Res<ToolTasks>| {
                let arguments = call.call.function.arguments;
                tasks.spawn(call.entity, async move {
                    bash(&arguments).await.unwrap_or_else(error)
                });
            },
        );
    }
}

fn error(error: anyhow::Error) -> String {
    format!("error: {error:#}")
}

fn string<'a>(arguments: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    arguments[key]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing string argument `{key}`"))
}

fn read(arguments: &Value) -> anyhow::Result<String> {
    let path = string(arguments, "path")?;
    let text = std::fs::read_to_string(path)?;
    let offset = arguments["offset"].as_u64().unwrap_or(1).max(1) as usize;
    let limit = arguments["limit"]
        .as_u64()
        .map_or(usize::MAX, |l| l as usize);
    let lines: Vec<&str> = text.lines().skip(offset - 1).take(limit).collect();
    let mut output = lines.join("\n");
    if output.len() > MAX_OUTPUT {
        let mut end = MAX_OUTPUT;
        while !output.is_char_boundary(end) {
            end -= 1;
        }
        output.truncate(end);
        output.push_str("\n[… truncated; read on with `offset` …]");
    }
    Ok(output)
}

fn write(arguments: &Value) -> anyhow::Result<String> {
    let path = PathBuf::from(string(arguments, "path")?);
    let content = string(arguments, "content")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, content)?;
    Ok(format!(
        "wrote {} bytes to {}",
        content.len(),
        path.display()
    ))
}

fn edit(arguments: &Value) -> anyhow::Result<String> {
    let path = string(arguments, "path")?;
    let old = string(arguments, "old_text")?;
    let new = string(arguments, "new_text")?;
    let text = std::fs::read_to_string(path)?;
    match text.matches(old).count() {
        1 => {
            std::fs::write(path, text.replacen(old, new, 1))?;
            Ok(format!("edited {path}"))
        }
        0 => anyhow::bail!("`old_text` does not occur in {path}"),
        n => anyhow::bail!("`old_text` occurs {n} times in {path}; include more context"),
    }
}

async fn bash(arguments: &Value) -> anyhow::Result<String> {
    let command = string(arguments, "command")?;
    let timeout = Duration::from_secs(arguments["timeout"].as_u64().unwrap_or(120));
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .output();
    let output = tokio::time::timeout(timeout, child)
        .await
        .map_err(|_| anyhow::anyhow!("timed out after {}s", timeout.as_secs()))??;
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        text.push_str(&stderr);
    }
    text.push_str(&format!("\n[exit: {}]", output.status));
    Ok(truncate(&text, MAX_OUTPUT))
}
