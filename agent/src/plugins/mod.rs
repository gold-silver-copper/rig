//! Feature plugins: ordinary Bevy plugins compiled into the agent, each of
//! which `--disable <name>` turns off. Add one here, then `reload`.

pub mod brp_tools;
pub mod cassette;
pub mod codemode;
pub mod durable;
pub mod evals;
pub mod greet;
pub mod mcp;
pub mod remote;
pub mod runtime;
pub mod telemetry;

use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;

use crate::Options;

/// Every feature plugin the options leave on.
pub fn features(options: &Options) -> PluginGroupBuilder {
    let mut group = PluginGroupBuilder::start::<Features>();
    let on = |name: &str| options.enabled(name);
    if on("greet") {
        group = group.add(greet::GreetPlugin);
    }
    if on("brp-tools") {
        group = group.add(brp_tools::BrpToolsPlugin);
    }
    if on("remote") {
        group = group.add(remote::RemoteSessionsPlugin);
    }
    if on("mcp") {
        group = group.add(mcp::McpPlugin {
            config: options.mcp.clone(),
        });
    }
    if on("durable") {
        group = group.add(durable::DurablePlugin);
    }
    if on("telemetry") {
        group = group.add(telemetry::TelemetryPlugin);
    }
    if on("cassette") {
        group = group.add(cassette::CassettePlugin {
            record: options.record.clone(),
            replay: options.replay.clone(),
        });
    }
    if on("evals") {
        group = group.add(evals::EvalsPlugin {
            record: options.record.clone(),
            replay: options.replay.clone(),
        });
    }
    if on("codemode") {
        group = group.add(codemode::CodeModePlugin);
    }
    group
}

/// The feature plugin group, as [`features`] builds it.
struct Features;

impl PluginGroup for Features {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
    }
}
