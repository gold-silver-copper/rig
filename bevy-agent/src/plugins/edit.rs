use bevy::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::agent::{AddTool, ToolReply, arguments};

pub struct EditPlugin;

impl Plugin for EditPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(
            "edit",
            "Replace `old_text`, which must occur exactly once in the file, with `new_text`.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text", "new_text"]
            }),
            edit,
        );
    }
}

#[derive(Deserialize)]
struct Args {
    path: String,
    old_text: String,
    new_text: String,
}

fn edit(In(args): In<Value>) -> ToolReply {
    let args: Args = match arguments(args) {
        Ok(args) => args,
        Err(reply) => return reply,
    };
    let text = match std::fs::read_to_string(&args.path) {
        Ok(text) => text,
        Err(error) => return ToolReply::Done(format!("error: {}: {error}", args.path)),
    };
    let count = text.matches(&args.old_text).count();
    if count != 1 || args.old_text.is_empty() {
        return ToolReply::Done(format!(
            "error: old_text occurs {count} times in {}; it must occur exactly once",
            args.path
        ));
    }
    let edited = text.replacen(&args.old_text, &args.new_text, 1);
    ToolReply::Done(match std::fs::write(&args.path, edited) {
        Ok(()) => format!("edited {}", args.path),
        Err(error) => format!("error: {}: {error}", args.path),
    })
}
