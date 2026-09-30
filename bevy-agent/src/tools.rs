//! Tools as ECS data. A tool is an entity with a [`ToolSpec`]; a call the
//! model makes is an entity with a [`ToolCall`] that some plugin answers by
//! inserting a [`ToolOutput`]. Native tools answer from Rust functions,
//! remote tools from another process over BRP.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_ecs::HotPatched;
use crossbeam_channel::{Receiver, TryRecvError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::plugins;

/// A tool the model may call. `parameters` is a JSON Schema, as text so the
/// component stays reflectable (and spawnable over BRP).
#[derive(Component, Reflect, Clone, Debug, Serialize, Deserialize)]
#[reflect(Component, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: String,
}

/// Marks a tool answered by an external process over BRP.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct RemoteTool;

/// Marks a tool answered by a native Rust function from [`plugins`].
#[derive(Component)]
pub struct NativeTool;

/// One call from the model, waiting for a [`ToolOutput`].
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component)]
pub struct ToolCall {
    pub name: String,
    /// JSON arguments, as text.
    pub arguments: String,
}

/// The answer to a [`ToolCall`].
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// A call someone has taken and is working on.
#[derive(Component)]
pub struct Claimed;

/// Calls nobody has answered or taken yet.
pub type OpenCalls<'w, 's> =
    Query<'w, 's, (Entity, &'static ToolCall), (Without<ToolOutput>, Without<Claimed>)>;

/// A native tool's function: JSON arguments in, text or an error out.
pub type Runner = fn(Value) -> Result<String, String>;

/// A native tool call running on its own thread.
#[derive(Component)]
struct Running(Receiver<Result<String, String>>);

/// A native tool: a description for the model and a blocking function that
/// runs on its own thread. Plugins return these from their `tool()`.
pub struct Native {
    pub name: &'static str,
    pub description: String,
    pub parameters: Value,
    pub run: Runner,
}

/// Native tool functions by name, rebuilt from [`plugins::native_tools`]
/// at startup and after every hot patch, so the pointers are always the
/// newest code and tools added to that list go live without a restart.
#[derive(Resource, Default)]
struct NativeRunners(HashMap<String, Runner>);

pub struct ToolsPlugin;

impl Plugin for ToolsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ToolSpec>()
            .register_type::<RemoteTool>()
            .register_type::<ToolCall>()
            .register_type::<ToolOutput>()
            .init_resource::<NativeRunners>()
            .add_systems(Startup, sync_native_tools)
            .add_systems(
                Update,
                (
                    sync_native_tools.run_if(on_message::<HotPatched>),
                    run_native_calls,
                    finish_native_calls,
                )
                    .chain(),
            );
    }
}

fn sync_native_tools(
    mut commands: Commands,
    mut runners: ResMut<NativeRunners>,
    mut specs: Query<(Entity, &mut ToolSpec), With<NativeTool>>,
) {
    runners.0.clear();
    let mut fresh: HashMap<String, ToolSpec> = HashMap::new();
    for tool in plugins::native_tools() {
        runners.0.insert(tool.name.to_owned(), tool.run);
        fresh.insert(
            tool.name.to_owned(),
            ToolSpec {
                name: tool.name.to_owned(),
                description: tool.description,
                parameters: tool.parameters.to_string(),
            },
        );
    }
    for (entity, mut spec) in &mut specs {
        match fresh.remove(&spec.name) {
            Some(new) => *spec = new,
            None => commands.entity(entity).despawn(),
        }
    }
    for spec in fresh.into_values() {
        commands.spawn((spec, NativeTool));
    }
}

fn run_native_calls(mut commands: Commands, runners: Res<NativeRunners>, calls: OpenCalls) {
    for (entity, call) in &calls {
        let Some(&run) = runners.0.get(&call.name) else {
            continue;
        };
        let args = serde_json::from_str(&call.arguments).unwrap_or(Value::Null);
        let (tx, rx) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            let _ = tx.send(run(args));
        });
        commands.entity(entity).insert((Claimed, Running(rx)));
    }
}

fn finish_native_calls(mut commands: Commands, running: Query<(Entity, &Running)>) {
    for (entity, Running(rx)) in &running {
        let output = match rx.try_recv() {
            Ok(Ok(text)) => ToolOutput::ok(text),
            Ok(Err(text)) => ToolOutput::error(text),
            Err(TryRecvError::Empty) => continue,
            Err(TryRecvError::Disconnected) => ToolOutput::error("the tool panicked"),
        };
        commands.entity(entity).remove::<Running>().insert(output);
    }
}
