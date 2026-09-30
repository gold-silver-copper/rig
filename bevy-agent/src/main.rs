//! A minimal coding agent: Rig for the model, Bevy for the runtime, ratatui for
//! the screen. Native plugins are modules of this crate and hot-patch in place;
//! external processes extend it over the Bevy Remote Protocol.

// Bevy queries are long by nature; `ProviderError` is rig's size to choose.
#![allow(clippy::type_complexity, clippy::result_large_err)]

mod agent;
mod hot;
mod model;
mod plugins;
mod remote;
mod tui;

use std::time::Duration;

use bevy_app::{ScheduleRunnerPlugin, prelude::*};

fn main() -> anyhow::Result<()> {
    let (llm, model_name) = model::from_env()?;
    let port = remote::pick_port()?;
    let terminal = ratatui::init();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        hook(info);
    }));

    App::new()
        .add_plugins((
            TaskPoolPlugin::default(),
            ScheduleRunnerPlugin::run_loop(Duration::from_millis(33)),
            agent::AgentPlugin { llm, model_name },
            tui::TuiPlugin::new(terminal),
            remote::RemotePlugin { port },
            hot::HotPatchPlugin,
            plugins::NativePlugins,
        ))
        .run();

    ratatui::restore();
    Ok(())
}
