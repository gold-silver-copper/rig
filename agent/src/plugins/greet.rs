//! A tiny example tool.

use bevy::prelude::*;
use serde_json::json;

use crate::glue::{Tool, Tools};
use crate::tools::blocking;

pub struct GreetPlugin;

impl Plugin for GreetPlugin {
    fn build(&self, app: &mut App) {
        let greet = blocking(
            "greet",
            "Greet someone by name.",
            json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
            |args| {
                let name = args["name"].as_str().unwrap_or("stranger");
                Ok(format!("Hello, {name}!"))
            },
        );
        app.world_mut().get_resource_or_init::<Tools>().add(Tool::Task(greet));
    }
}
