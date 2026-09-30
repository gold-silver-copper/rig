//! The tool registry and the built-in tools: `read`, `write`, `edit`, `bash`.
//!
//! A native plugin adds a tool with [`AddTool::add_tool`]; a BRP plugin adds
//! one over the wire (see `brp.rs`).

use std::collections::BTreeMap;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use serde_json::{Value, json};

pub type ToolFn = Arc<dyn Fn(Value) -> Result<String, String> + Send + Sync>;

/// How a tool runs.
#[derive(Clone)]
pub enum Run {
    /// A function run off the main thread.
    Local(ToolFn),
    /// Forwarded to the BRP plugin with this name.
    Remote(String),
    /// Rebuild and restart the agent.
    Reload,
}

#[derive(Clone)]
pub struct Tool {
    pub def: ToolDefinition,
    pub run: Run,
}

#[derive(Resource, Default)]
pub struct Tools(pub BTreeMap<String, Tool>);

impl Tools {
    pub fn insert(&mut self, def: ToolDefinition, run: Run) {
        self.0.insert(def.name.clone(), Tool { def, run });
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.0.values().map(|tool| tool.def.clone()).collect()
    }
}

pub trait AddTool {
    /// Offer the model a tool named `name` taking JSON-schema `parameters`.
    fn add_tool(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        run: impl Fn(Value) -> Result<String, String> + Send + Sync + 'static,
    ) -> &mut Self;
}

impl AddTool for App {
    fn add_tool(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        run: impl Fn(Value) -> Result<String, String> + Send + Sync + 'static,
    ) -> &mut Self {
        let name = ToolName::new(name).unwrap_or_else(|e| panic!("tool name: {e}"));
        let def = ToolDefinition::new(name, description, parameters);
        self.world_mut()
            .get_resource_or_init::<Tools>()
            .insert(def, Run::Local(Arc::new(run)));
        self
    }
}

pub struct ToolsPlugin;

impl Plugin for ToolsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Tools>()
            .add_tool(
                "read",
                "Read a text file. Optional 1-based `offset` line and `limit` line count.",
                json!({"type": "object", "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"}
                }, "required": ["path"]}),
                read,
            )
            .add_tool(
                "write",
                "Create or overwrite a file with `content`, creating parent directories.",
                json!({"type": "object", "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                }, "required": ["path", "content"]}),
                write,
            )
            .add_tool(
                "edit",
                "Replace `old_text`, which must occur exactly once in the file, with `new_text`.",
                json!({"type": "object", "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                }, "required": ["path", "old_text", "new_text"]}),
                edit,
            )
            .add_tool(
                "bash",
                "Run a bash command in the working directory; returns stdout and stderr. Default timeout 120s.",
                json!({"type": "object", "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "seconds"}
                }, "required": ["command"]}),
                bash,
            );
    }
}

fn arg<'a>(args: &'a Value, name: &str) -> Result<&'a str, String> {
    args.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string argument `{name}`"))
}

/// Keep at most `max` bytes, cut on a char boundary, from the head or tail.
pub fn clip(text: &str, max: usize, tail: bool) -> String {
    if text.len() <= max {
        return text.to_owned();
    }
    let omitted = text.len() - max;
    if tail {
        let start = (omitted..=text.len())
            .find(|i| text.is_char_boundary(*i))
            .unwrap_or(text.len());
        format!("[{omitted} bytes omitted]\n{}", &text[start..])
    } else {
        let end = (0..=max)
            .rev()
            .find(|i| text.is_char_boundary(*i))
            .unwrap_or(0);
        format!("{}\n[{omitted} bytes omitted]", &text[..end])
    }
}

fn read(args: Value) -> Result<String, String> {
    let path = arg(&args, "path")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let offset = args
        .get("offset")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1) as usize;
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(2000) as usize;
    let lines: Vec<&str> = text.lines().skip(offset - 1).take(limit).collect();
    Ok(clip(&lines.join("\n"), 50_000, false))
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
    let old = arg(&args, "old_text")?;
    let new = arg(&args, "new_text")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    match text.matches(old).count() {
        1 => {
            std::fs::write(path, text.replacen(old, new, 1)).map_err(|e| e.to_string())?;
            Ok(format!("edited {path}"))
        }
        0 => Err(format!("`old_text` not found in {path}")),
        n => Err(format!(
            "`old_text` occurs {n} times in {path}; add context"
        )),
    }
}

fn bash(args: Value) -> Result<String, String> {
    let command = arg(&args, "command")?;
    let timeout = Duration::from_secs(args.get("timeout").and_then(Value::as_u64).unwrap_or(120));
    let mut child = Command::new("bash")
        .arg("-c")
        .arg(format!("{{ {command}\n}} 2>&1"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;
    // A reader thread, so a command that outlives the timeout (or leaves a
    // grandchild holding the pipe) cannot block the tool.
    let output = Arc::new(Mutex::new(Vec::new()));
    let reader = child.stdout.take().map(|mut stdout| {
        let output = output.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            while let Ok(n @ 1..) = stdout.read(&mut buf) {
                output.lock().unwrap().extend_from_slice(&buf[..n]);
            }
        })
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(status);
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let drained = Instant::now();
    while reader.as_ref().is_some_and(|r| !r.is_finished())
        && drained.elapsed() < Duration::from_secs(1)
    {
        std::thread::sleep(Duration::from_millis(10));
    }
    let text = String::from_utf8_lossy(&output.lock().unwrap()).into_owned();
    let text = clip(&text, 30_000, true);
    match status {
        Some(status) if status.success() => Ok(text),
        Some(status) => Err(format!("{text}\n[{status}]")),
        None => Err(format!("{text}\n[timed out after {}s]", timeout.as_secs())),
    }
}
