//! The `reload` tool: recompile the agent and hot-patch it in place.

use bevy::prelude::*;

use crate::hot::{HotReload, PatchStatus};
use crate::tools::{Claimed, ToolCall, ToolOutput, ToolSpec};

pub struct ReloadPlugin;

impl Plugin for ReloadPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_spec)
            .add_systems(Update, (start_reload, finish_reload));
    }
}

/// The patch generation a reload call is waiting to see pass.
#[derive(Component)]
struct AwaitingPatch(u64);

fn spawn_spec(mut commands: Commands) {
    commands.spawn(ToolSpec {
        name: "reload".into(),
        description: "Recompile your own source and hot-patch the running agent, keeping this \
                      session. Call after editing files under your src/. Returns compiler errors \
                      if the build fails."
            .into(),
        parameters: r#"{"type":"object","properties":{}}"#.into(),
    });
}

fn start_reload(
    mut commands: Commands,
    mut hot: ResMut<HotReload>,
    calls: Query<(Entity, &ToolCall), (Without<ToolOutput>, Without<Claimed>)>,
) {
    for (entity, call) in &calls {
        if call.name != "reload" {
            continue;
        }
        let mut entity = commands.entity(entity);
        let before = hot.generation;
        if hot.request() {
            entity.insert((Claimed, AwaitingPatch(before)));
        } else {
            entity.insert(ToolOutput::error("hot-patching is unavailable in this build"));
        }
    }
}

fn finish_reload(
    mut commands: Commands,
    hot: Res<HotReload>,
    waiting: Query<(Entity, &AwaitingPatch)>,
) {
    for (entity, AwaitingPatch(before)) in &waiting {
        if hot.generation == *before || matches!(hot.status, PatchStatus::Building) {
            continue;
        }
        let output = match &hot.status {
            PatchStatus::Applied { count, elapsed } => ToolOutput::ok(format!(
                "hot-patched (patch #{count}, {:.2}s); the new code is live",
                elapsed.as_secs_f32()
            )),
            PatchStatus::Failed(error) => ToolOutput::error(error.clone()),
            other => ToolOutput::error(format!("unexpected patch state: {other:?}")),
        };
        commands.entity(entity).remove::<AwaitingPatch>().insert(output);
    }
}
