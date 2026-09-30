//! A sample native plugin: the smallest possible tool.

use bevy::prelude::*;
use serde_json::json;

use crate::agent::{AgentAppExt, CallOutput, ToolInvocation};

pub struct GreetingPlugin;

impl Plugin for GreetingPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(
            "greeting",
            "Greet someone by name.",
            json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
            greet,
        );
    }
}

fn greet(In(invocation): In<ToolInvocation>, mut commands: Commands) {
    let name = invocation.call.function.arguments["name"]
        .as_str()
        .unwrap_or("stranger");
    commands
        .entity(invocation.entity)
        .insert(CallOutput(format!("Hello, {name}!")));
}
