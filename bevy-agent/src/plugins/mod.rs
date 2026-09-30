//! Native plugins: ordinary Bevy plugins compiled into the agent. Add one
//! here, register it in `NativePlugins`, and `reload`.

use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;

pub mod coding;
pub mod greeter;

pub struct NativePlugins;

impl PluginGroup for NativePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(coding::CodingToolsPlugin)
            .add(greeter::GreeterPlugin)
    }
}
