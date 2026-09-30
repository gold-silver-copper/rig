//! The native tools, pi's four: `read`, `write`, `edit` and `bash`. Each is
//! an ordinary Bevy plugin that registers a Rig `DynamicTool`, which the
//! glue runs. To add a tool, write a plugin here, add it to
//! [`ToolPlugins`], and reload.

mod bash;
mod edit;
mod read;
mod write;

use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;
use rig_core::tool::ToolExecutionError;

use crate::glue::ToolOutput;

pub struct ToolPlugins;

impl PluginGroup for ToolPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(read::ReadPlugin)
            .add(write::WritePlugin)
            .add(edit::EditPlugin)
            .add(bash::BashPlugin)
            .add(RedactPlugin)
    }
}

/// Output is capped at this many lines or bytes, whichever comes first.
pub const MAX_LINES: usize = 2000;
pub const MAX_BYTES: usize = 50 * 1024;

/// Parses tool arguments.
pub fn arguments<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
) -> Result<T, ToolExecutionError> {
    serde_json::from_value(value)
        .map_err(|error| ToolExecutionError::invalid_args(format!("invalid arguments: {error}")))
}

/// Replaces the values of credential-like environment variables in every
/// tool output, so a command such as `env` cannot hand API keys to the
/// model, its provider, or a saved session.
struct RedactPlugin;

impl Plugin for RedactPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Secrets::from_env()).add_observer(redact);
    }
}

#[derive(Resource)]
pub struct Secrets(Vec<(String, String)>);

impl Secrets {
    fn from_env() -> Self {
        let mut secrets: Vec<(String, String)> = std::env::vars()
            .filter(|(name, value)| {
                let name = name.to_uppercase();
                value.len() >= 8
                    && ["KEY", "TOKEN", "SECRET", "PASSWORD"]
                        .iter()
                        .any(|word| name.contains(word))
            })
            .collect();
        // Longest first, so a secret containing another is replaced whole.
        secrets.sort_by_key(|(_, value)| std::cmp::Reverse(value.len()));
        Self(secrets)
    }

    pub fn redact(&self, text: &str) -> Option<String> {
        let mut redacted: Option<String> = None;
        for (name, value) in &self.0 {
            let current = redacted.as_deref().unwrap_or(text);
            if current.contains(value.as_str()) {
                redacted = Some(current.replace(value.as_str(), &format!("[redacted {name}]")));
            }
        }
        redacted
    }
}

fn redact(insert: On<Insert<ToolOutput>>, mut outputs: Query<&mut ToolOutput>, secrets: Res<Secrets>) {
    if let Ok(mut output) = outputs.get_mut(insert.entity)
        && let Some(redacted) = secrets.redact(&output.0)
    {
        output.0 = redacted;
    }
}
