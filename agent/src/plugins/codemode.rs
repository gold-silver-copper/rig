//! Code mode: a `codemode` tool that runs model-written JavaScript whose only
//! capability is calling the agent's other tools (rig-codemode, QuickJS).
//! The model can chain, loop over and filter tool calls in one step; nested
//! calls and their results stay out of the conversation.

use bevy::prelude::*;
use rig_codemode::CodeMode;

use crate::glue::{Tool, Tools};
use crate::prompt::Prompt;

pub const TOOL: &str = "codemode";

pub struct CodeModePlugin;

impl Plugin for CodeModePlugin {
    fn build(&self, app: &mut App) {
        app.world_mut().get_resource_or_init::<Prompt>().tool(
            TOOL,
            "Run JavaScript that calls other tools (chains, loops, filtering large results)",
            &["Use codemode to batch or chain several tool calls, or to filter large tool output down to \
               what you need, instead of issuing many individual tool calls."],
        );
        app.add_systems(PreUpdate, rebuild.run_if(resource_changed::<Tools>));
    }
}

/// Offer every task tool but code mode itself inside the sandbox. World
/// tools (reload, BRP tools) need the ECS and are not callable from it.
fn rebuild(mut tools: ResMut<Tools>) {
    let nested: Vec<_> = tools
        .0
        .iter()
        .filter(|(name, _)| name.as_str() != TOOL)
        .filter_map(|(_, tool)| match tool {
            Tool::Task(tool) => Some(tool.clone()),
            Tool::World(_) => None,
        })
        .collect();
    let tool = CodeMode::new(nested).tool();
    let unchanged = matches!(
        tools.0.get(TOOL),
        Some(Tool::Task(current)) if current.definition() == tool.definition()
    );
    if !unchanged {
        tools.add(Tool::Task(tool));
    }
}
