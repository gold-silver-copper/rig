//! Native plugins: ordinary Bevy plugins compiled into the agent. Add one
//! here, then `reload`.

mod greet;

use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;

pub struct NativePlugins;

impl PluginGroup for NativePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>().add(greet::GreetPlugin)
    }
}
