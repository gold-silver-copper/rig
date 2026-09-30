//! Native plugins: compiled into the agent, hot-patched when their source
//! changes. A tool plugin is a module with a `tool()` returning a
//! [`Native`]; list it in [`native_tools`] and it goes live on the next
//! patch. Plugins that need the ECS are ordinary Bevy plugins, added in
//! [`NativePlugins`].

use bevy::app::{PluginGroup, PluginGroupBuilder};

use crate::tools::Native;

mod bash;
mod edit;
mod read;
mod reload;
mod status;
mod write;

/// Every native tool. Re-read after each hot patch.
pub fn native_tools() -> Vec<Native> {
    vec![read::tool(), write::tool(), edit::tool(), bash::tool()]
}

/// Native plugins that work on the ECS directly.
pub struct NativePlugins;

impl PluginGroup for NativePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(reload::ReloadPlugin)
            .add(status::StatusPlugin)
    }
}

/// Clip tool output so one call cannot flood the context.
fn clip(mut text: String, max: usize) -> String {
    if text.len() > max {
        let mut cut = max;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        let dropped = text.len() - cut;
        text.truncate(cut);
        text.push_str(&format!("\n[... {dropped} more bytes]"));
    }
    text
}

fn str_arg<'a>(args: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| format!("missing string argument `{key}`"))
}
