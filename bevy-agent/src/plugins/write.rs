use bevy::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::agent::{AddTool, ToolReply, arguments};

pub struct WritePlugin;

impl Plugin for WritePlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(
            "write",
            "Create or overwrite a file with `content`, creating parent directories.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }),
            write,
        );
    }
}

#[derive(Deserialize)]
struct Args {
    path: String,
    content: String,
}

fn write(In(args): In<Value>) -> ToolReply {
    let args: Args = match arguments(args) {
        Ok(args) => args,
        Err(reply) => return reply,
    };
    let path = std::path::Path::new(&args.path);
    let result = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(path, &args.content));
    ToolReply::Done(match result {
        Ok(()) => format!("wrote {} bytes to {}", args.content.len(), args.path),
        Err(error) => format!("error: {}: {error}", args.path),
    })
}
