//! Native plugins: ordinary Bevy plugins compiled into the agent. Each one
//! registers its tools with [`AddTool`](crate::agent::AddTool). To add one,
//! write a module here, add it to [`NativePlugins`], and reload.

mod bash;
mod edit;
mod read;
mod write;

use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;

pub struct NativePlugins;

impl PluginGroup for NativePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(read::ReadPlugin)
            .add(write::WritePlugin)
            .add(edit::EditPlugin)
            .add(bash::BashPlugin)
    }
}

/// Keeps the end of long output: at most `limit` bytes.
pub fn truncate(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_owned();
    }
    let mut start = text.len() - limit;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("[{start} earlier bytes cut]\n{}", &text[start..])
}
