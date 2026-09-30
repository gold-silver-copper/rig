use bevy::prelude::*;
use rig_core::tool::{DynamicTool, ToolExecutionError, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

use super::arguments;
use crate::glue::AddTool;

pub struct WritePlugin;

impl Plugin for WritePlugin {
    fn build(&self, app: &mut App) {
        app.add_rig_tool(DynamicTool::new(
            "write",
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it \
             does. Automatically creates parent directories.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                    "content": {"type": "string", "description": "Content to write to the file"}
                },
                "required": ["path", "content"]
            }),
            |args: Value| Box::pin(async move { write(arguments(args)?) }),
        ));
    }
}

#[derive(Deserialize)]
struct Args {
    path: String,
    content: String,
}

fn write(args: Args) -> Result<ToolOutput, ToolExecutionError> {
    let path = std::path::Path::new(&args.path);
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(ToolExecutionError::from_error)?;
    }
    std::fs::write(path, &args.content).map_err(ToolExecutionError::from_error)?;
    Ok(ToolOutput::text(format!("Successfully wrote to {}", args.path)))
}
