//! The built-in tools, as a hot-patchable native plugin. Every function here
//! can be edited while the agent runs.

use std::process::Command;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::AsyncComputeTaskPool;
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use serde_json::json;

use crate::agent::{Dispatched, Tool, ToolCallRequest, ToolDone, ToolTask};

pub struct NativeToolsPlugin;

impl Plugin for NativeToolsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, register)
            .add_systems(Update, run_tools);
    }
}

/// The system prompt.
pub fn preamble() -> String {
    format!(
        "You are a minimal coding agent working in {}. Use the tools to read, \
         search and change files and to run commands. Be brief.",
        std::env::current_dir()
            .map(|d| d.display().to_string())
            .unwrap_or_default()
    )
}

/// A word for the status bar; edit it to see a patch land.
pub fn status_hint() -> &'static str {
    "native v1"
}

fn register(mut commands: Commands) {
    let path = json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] });
    let tools = [
        ("read_file", "Read a text file.", path.clone()),
        ("list_dir", "List a directory.", path),
        (
            "write_file",
            "Write a whole text file, creating it if needed.",
            json!({ "type": "object", "properties": { "path": { "type": "string" }, "content": { "type": "string" } }, "required": ["path", "content"] }),
        ),
        (
            "bash",
            "Run a shell command and return its output.",
            json!({ "type": "object", "properties": { "command": { "type": "string" } }, "required": ["command"] }),
        ),
    ];
    for (name, description, parameters) in tools {
        if let Ok(name) = ToolName::new(name) {
            commands.spawn(Tool {
                definition: ToolDefinition::new(name, description, parameters),
                remote: false,
            });
        }
    }
}

fn run_tools(
    mut commands: Commands,
    calls: Query<
        (Entity, &ToolCallRequest),
        (Without<ToolDone>, Without<ToolTask>, Without<Dispatched>),
    >,
    tools: Query<&Tool>,
) {
    for (entity, call) in &calls {
        let name = call.0.function.name.as_str();
        if !tools
            .iter()
            .any(|tool| !tool.remote && tool.definition.name == name)
        {
            continue;
        }
        let args = &call.0.function.arguments;
        let text = |key: &str| args[key].as_str().unwrap_or_default().to_owned();
        match name {
            "read_file" => {
                commands
                    .entity(entity)
                    .insert(ToolDone(read_file(&text("path"))));
            }
            "list_dir" => {
                commands
                    .entity(entity)
                    .insert(ToolDone(list_dir(&text("path"))));
            }
            "write_file" => {
                let result = std::fs::write(text("path"), text("content"))
                    .map_or_else(|e| format!("error: {e}"), |()| "ok".to_owned());
                commands.entity(entity).insert(ToolDone(result));
            }
            "bash" => {
                let command = text("command");
                let task = AsyncComputeTaskPool::get().spawn(async move { bash(&command) });
                commands.entity(entity).insert(ToolTask(task));
            }
            _ => {
                commands
                    .entity(entity)
                    .insert(ToolDone(format!("error: `{name}` has no native handler")));
            }
        }
    }
}

fn read_file(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| format!("error: {e}"))
}

fn list_dir(path: &str) -> String {
    match std::fs::read_dir(path) {
        Ok(entries) => {
            let mut names: Vec<String> = entries
                .flatten()
                .map(|e| {
                    let mut name = e.file_name().to_string_lossy().into_owned();
                    if e.path().is_dir() {
                        name.push('/');
                    }
                    name
                })
                .collect();
            names.sort();
            names.join("\n")
        }
        Err(e) => format!("error: {e}"),
    }
}

fn bash(command: &str) -> String {
    match Command::new("bash").arg("-lc").arg(command).output() {
        Ok(output) => {
            let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            if !output.status.success() {
                text.push_str(&format!("\n[exit {}]", output.status));
            }
            truncate(text, 16_000)
        }
        Err(e) => format!("error: {e}"),
    }
}

fn truncate(mut text: String, max: usize) -> String {
    if text.len() > max {
        let cut = (0..=max)
            .rev()
            .find(|i| text.is_char_boundary(*i))
            .unwrap_or(0);
        text.truncate(cut);
        text.push_str("\n[truncated]");
    }
    text
}
