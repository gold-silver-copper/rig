//! An example native plugin: an ordinary Bevy plugin that adds one tool.
//! Edit it, then `reload`, and the change is live.

use bevy::prelude::*;
use serde_json::{Value, json};

use crate::tools::AgentAppExt;

pub struct GreeterPlugin;

impl Plugin for GreeterPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(
            "greet",
            "Greet someone by name.",
            json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
            |args: Value| {
                let name = args.get("name").and_then(Value::as_str).unwrap_or("world");
                Ok(format!("Hello, {name}!"))
            },
        );
    }
}
