//! Native plugins. They live in this crate so the hot patcher can rebuild
//! them: edit a system here and the running agent picks it up.

pub mod tools;

use bevy_app::prelude::*;

pub struct NativePlugins;

impl Plugin for NativePlugins {
    fn build(&self, app: &mut App) {
        app.add_plugins(tools::NativeToolsPlugin);
    }
}
