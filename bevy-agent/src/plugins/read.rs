use bevy::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::agent::{AddTool, ToolReply, arguments};

pub struct ReadPlugin;

impl Plugin for ReadPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(
            "read",
            "Read a text file. `offset` is the first line to read (1-based), `limit` the \
             number of lines; output is capped at 2000 lines.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"}
                },
                "required": ["path"]
            }),
            read,
        );
    }
}

#[derive(Deserialize)]
struct Args {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

fn read(In(args): In<Value>) -> ToolReply {
    let args: Args = match arguments(args) {
        Ok(args) => args,
        Err(reply) => return reply,
    };
    let text = match std::fs::read_to_string(&args.path) {
        Ok(text) => text,
        Err(error) => return ToolReply::Done(format!("error: {}: {error}", args.path)),
    };
    let skip = args.offset.unwrap_or(1).saturating_sub(1);
    let take = args.limit.unwrap_or(2000).min(2000);
    let lines: Vec<&str> = text.lines().skip(skip).take(take).collect();
    let total = text.lines().count();
    let mut output = lines.join("\n");
    if skip + lines.len() < total {
        output.push_str(&format!(
            "\n[{} more lines; continue with offset {}]",
            total - skip - lines.len(),
            skip + lines.len() + 1
        ));
    }
    ToolReply::Done(output)
}
