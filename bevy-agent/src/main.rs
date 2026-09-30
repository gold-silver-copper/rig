mod hot;

use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;

fn main() -> AppExit {
    if let Some(code) = rigpi_hotpatch::intercept() {
        std::process::exit(code);
    }
    let fat = hot::bootstrap();
    App::new()
        .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_millis(500))))
        .add_plugins(hot::HotReloadPlugin(fat))
        .add_systems(Update, tick)
        .run()
}

fn tick(hot: Res<hot::HotReload>) {
    println!("tick v3 again {}", 40 + 2); print!(" {:?}", hot.status);
}
