//! Native plugins: ordinary Bevy plugins compiled into the agent. Each offers
//! tools with [`AgentAppExt::add_tool`](crate::agent::AgentAppExt::add_tool).
//! Changing one takes effect through a reload.

mod coding;
mod greeting;

use bevy::prelude::*;

/// Every native plugin the agent is built with.
pub struct NativePlugins;

impl Plugin for NativePlugins {
    fn build(&self, app: &mut App) {
        app.add_plugins((coding::CodingToolsPlugin, greeting::GreetingPlugin));
    }
}

/// At most `max` bytes of `text`, keeping the end, where errors and the
/// latest output are.
pub fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut start = text.len() - max;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("[… {start} bytes cut …]\n{}", &text[start..])
}
