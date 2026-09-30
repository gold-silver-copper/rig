//! A tiny example tool.

use bevy::prelude::*;
use serde_json::json;

use crate::tools::AddTool;

pub struct GreetPlugin;

impl Plugin for GreetPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(
            "greet",
            "Greet someone by name.",
            json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
            |args| {
                let name = args["name"].as_str().unwrap_or("stranger");
                Ok(format!("Hello, {name}!"))
            },
        );
    }
}
