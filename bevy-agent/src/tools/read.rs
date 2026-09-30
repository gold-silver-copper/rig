use bevy::prelude::*;
use rig_core::tool::{DynamicTool, ToolExecutionError, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{MAX_BYTES, MAX_LINES, arguments};
use crate::glue::AddTool;

pub struct ReadPlugin;

impl Plugin for ReadPlugin {
    fn build(&self, app: &mut App) {
        app.add_rig_tool(DynamicTool::new(
            "read",
            format!(
                "Read the contents of a file. Output is truncated to {MAX_LINES} lines or {}KB \
                 (whichever is hit first). Use offset/limit for large files. When you need the \
                 full file, continue with offset until complete.",
                MAX_BYTES / 1024
            ),
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                    "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                    "limit": {"type": "number", "description": "Maximum number of lines to read"}
                },
                "required": ["path"]
            }),
            |args: Value| Box::pin(async move { read(arguments(args)?) }),
        ));
    }
}

#[derive(Deserialize)]
struct Args {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

fn read(args: Args) -> Result<ToolOutput, ToolExecutionError> {
    let text = std::fs::read_to_string(&args.path)
        .map_err(|error| ToolExecutionError::not_found(format!("{}: {error}", args.path)))?;
    let lines: Vec<&str> = text.split('\n').collect();
    let start = args.offset.unwrap_or(1).max(1) - 1;
    if start >= lines.len() {
        return Err(ToolExecutionError::invalid_args(format!(
            "Offset {} is beyond end of file ({} lines total)",
            start + 1,
            lines.len()
        )));
    }
    let end = args
        .limit
        .map_or(lines.len(), |limit| (start + limit).min(lines.len()));
    let mut shown = Vec::new();
    let mut bytes = 0;
    for line in &lines[start..end] {
        if shown.len() == MAX_LINES || bytes + line.len() + 1 > MAX_BYTES {
            break;
        }
        bytes += line.len() + 1;
        shown.push(*line);
    }
    let last = start + shown.len();
    let mut output = shown.join("\n");
    if last < end {
        output.push_str(&format!(
            "\n\n[Showing lines {}-{last} of {}. Use offset={} to continue.]",
            start + 1,
            lines.len(),
            last + 1
        ));
    } else if end < lines.len() {
        output.push_str(&format!(
            "\n\n[{} more lines in file. Use offset={} to continue.]",
            lines.len() - end,
            end + 1
        ));
    }
    Ok(ToolOutput::text(output))
}
