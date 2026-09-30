//! External plugins over the Bevy Remote Protocol.
//!
//! The agent serves BRP on a port fixed for the whole session, so it stays the
//! same across reloads. Besides Bevy's built-in `world.*` methods, it adds:
//!
//! - `agent.register_tool` `{plugin, name, description, parameters}`: offer a
//!   tool to the model. Registering a name again replaces it, so a plugin can
//!   re-register after every reconnect.
//! - `agent.unregister_tool` `{name}`.
//! - `agent.tool_calls+watch` `{plugin}`: a stream of the calls to the
//!   plugin's tools, each `{call_id, name, arguments}`.
//! - `agent.tool_result` `{call_id, output}`: answer a call.
//! - `agent.prompt` `{text}`: queue a prompt, as if typed.
//! - `agent.status`: the model, turn state, tools and transcript.
//!
//! Registrations are part of the session, so a plugin's tools stay offered
//! across a reload. A call nobody watches waits for the plugin to reconnect;
//! re-registering redelivers calls that were not answered.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_remote::http::RemoteHttpPlugin;
use bevy_remote::{BrpError, BrpResult, RemotePlugin, error_codes};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::agent::{
    ActiveModel, CallOutput, Config, PromptQueue, ToolInvocation, ToolRegistry, ToolTasks,
    Transcript, Turn, call_key,
};
use crate::session::ExternalTool;

/// How long a call to an external tool may wait for its answer.
const CALL_TIMEOUT: Duration = Duration::from_secs(180);

pub struct BrpPlugin;

/// Tools registered by external plugins, by name.
#[derive(Resource, Default)]
pub struct ExternalTools(pub BTreeMap<String, ExternalTool>);

/// Calls to external tools that have not been answered.
#[derive(Resource, Default)]
struct ExternalCalls(BTreeMap<String, ExternalCall>);

struct ExternalCall {
    entity: Entity,
    plugin: String,
    name: String,
    arguments: Value,
    delivered: bool,
    since: Instant,
}

/// The one handler behind every external tool.
#[derive(Resource)]
struct ExternalHandler(bevy::ecs::system::SystemId<In<ToolInvocation>>);

impl Plugin for BrpPlugin {
    fn build(&self, app: &mut App) {
        let port = app.world().resource::<Config>().brp_port;
        let remote = RemotePlugin::default()
            .with_method_main("agent.register_tool", register_tool)
            .with_method_main("agent.unregister_tool", unregister_tool)
            .with_watching_method_main("agent.tool_calls+watch", watch_tool_calls)
            .with_method_main("agent.tool_result", tool_result)
            .with_method_main("agent.prompt", prompt)
            .with_method_main("agent.status", status);
        let handler = app.world_mut().register_system(external_tool);
        app.add_plugins((remote, RemoteHttpPlugin::default().with_port(port)))
            .insert_resource(ExternalHandler(handler))
            .init_resource::<ExternalTools>()
            .init_resource::<ExternalCalls>()
            .add_systems(Startup, probe_server)
            .add_systems(Update, (time_out_calls, announce_ready));
    }

    /// Offer the tools registered before the reload again.
    fn finish(&self, app: &mut App) {
        let saved = app
            .world_mut()
            .remove_resource::<SavedExternalTools>()
            .map(|saved| saved.0)
            .unwrap_or_default();
        for tool in saved {
            add_external_tool(app.world_mut(), tool);
        }
    }
}

/// The registrations restored from the session, until `finish` adds them.
#[derive(Resource)]
pub struct SavedExternalTools(pub Vec<ExternalTool>);

/// Whether the BRP server came up, once known.
#[derive(Resource)]
struct Probe(crossbeam_channel::Receiver<bool>);

fn probe_server(config: Res<Config>, tasks: Res<ToolTasks>, mut commands: Commands) {
    let (sender, receiver) = crossbeam_channel::bounded(1);
    let port = config.brp_port;
    tasks.handle().spawn(async move {
        for _ in 0..200 {
            if tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
                let _ = sender.send(true);
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let _ = sender.send(false);
    });
    commands.insert_resource(Probe(receiver));
}

/// Tell the supervisor this binary started, once its BRP server accepts
/// connections; a binary whose server never comes up has failed to start.
fn announce_ready(
    probe: Option<Res<Probe>>,
    config: Res<Config>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
) {
    let Some(probe) = probe else { return };
    let Ok(up) = probe.0.try_recv() else { return };
    commands.remove_resource::<Probe>();
    if up {
        let _ = std::fs::write(config.paths.ready(), std::process::id().to_string());
    } else {
        eprintln!("the BRP server did not start on port {}", config.brp_port);
        exit.write(AppExit::error());
    }
}

fn add_external_tool(world: &mut World, tool: ExternalTool) {
    let handler = world.resource::<ExternalHandler>().0;
    world.resource_mut::<ToolRegistry>().0.insert(
        tool.name.clone(),
        crate::agent::RegisteredTool {
            definition: rig_core::completion::ToolDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            },
            handler,
        },
    );
    world
        .resource_mut::<ExternalTools>()
        .0
        .insert(tool.name.clone(), tool);
}

fn external_tool(
    In(invocation): In<ToolInvocation>,
    tools: Res<ExternalTools>,
    mut calls: ResMut<ExternalCalls>,
    mut commands: Commands,
) {
    let name = invocation.call.function.name.to_string();
    let Some(tool) = tools.0.get(&name) else {
        commands
            .entity(invocation.entity)
            .insert(CallOutput(format!("error: `{name}` is no longer registered")));
        return;
    };
    calls.0.insert(
        call_key(&invocation.call),
        ExternalCall {
            entity: invocation.entity,
            plugin: tool.plugin.clone(),
            name,
            arguments: invocation.call.function.arguments.clone(),
            delivered: false,
            since: Instant::now(),
        },
    );
}

fn time_out_calls(mut calls: ResMut<ExternalCalls>, mut commands: Commands) {
    calls.0.retain(|_, call| {
        if call.since.elapsed() < CALL_TIMEOUT {
            return true;
        }
        commands.entity(call.entity).insert(CallOutput(format!(
            "error: plugin `{}` did not answer within {}s",
            call.plugin,
            CALL_TIMEOUT.as_secs()
        )));
        false
    });
}

fn params<T: for<'de> Deserialize<'de>>(params: Option<Value>) -> Result<T, BrpError> {
    serde_json::from_value(params.unwrap_or(Value::Null)).map_err(|error| BrpError {
        code: error_codes::INVALID_PARAMS,
        message: error.to_string(),
        data: None,
    })
}

fn register_tool(In(input): In<Option<Value>>, world: &mut World) -> BrpResult {
    #[derive(Deserialize)]
    struct Params {
        plugin: String,
        name: String,
        #[serde(default)]
        description: String,
        #[serde(default = "empty_schema")]
        parameters: Value,
    }
    fn empty_schema() -> Value {
        json!({ "type": "object", "properties": {} })
    }
    let Params {
        plugin,
        name,
        description,
        parameters,
    } = params(input)?;
    if world.resource::<ToolRegistry>().0.contains_key(&name)
        && !world.resource::<ExternalTools>().0.contains_key(&name)
    {
        return Err(BrpError {
            code: error_codes::INVALID_PARAMS,
            message: format!("`{name}` is a native tool and cannot be replaced"),
            data: None,
        });
    }
    // A plugin registering again has reconnected: whatever it was sent
    // before and did not answer is sent again.
    for call in world.resource_mut::<ExternalCalls>().0.values_mut() {
        if call.plugin == plugin {
            call.delivered = false;
        }
    }
    add_external_tool(
        world,
        ExternalTool {
            plugin,
            name: name.clone(),
            description,
            parameters,
        },
    );
    Ok(json!({ "registered": name }))
}

fn unregister_tool(In(input): In<Option<Value>>, world: &mut World) -> BrpResult {
    #[derive(Deserialize)]
    struct Params {
        name: String,
    }
    let Params { name } = params(input)?;
    if world.resource_mut::<ExternalTools>().0.remove(&name).is_some() {
        world.resource_mut::<ToolRegistry>().0.remove(&name);
    }
    Ok(json!({ "unregistered": name }))
}

fn watch_tool_calls(In(input): In<Option<Value>>, mut calls: ResMut<ExternalCalls>) -> BrpResult<Option<Value>> {
    #[derive(Deserialize)]
    struct Params {
        plugin: String,
    }
    let Params { plugin } = params(input)?;
    let mut batch = Vec::new();
    for (call_id, call) in &mut calls.0 {
        if call.plugin == plugin && !call.delivered {
            call.delivered = true;
            batch.push(json!({ "call_id": call_id, "name": call.name, "arguments": call.arguments }));
        }
    }
    Ok((!batch.is_empty()).then(|| Value::Array(batch)))
}

fn tool_result(
    In(input): In<Option<Value>>,
    mut calls: ResMut<ExternalCalls>,
    mut commands: Commands,
) -> BrpResult {
    #[derive(Deserialize)]
    struct Params {
        call_id: String,
        output: String,
    }
    let Params { call_id, output } = params(input)?;
    let Some(call) = calls.0.remove(&call_id) else {
        return Err(BrpError {
            code: error_codes::INVALID_PARAMS,
            message: format!("no call `{call_id}` is waiting for a result"),
            data: None,
        });
    };
    commands.entity(call.entity).insert(CallOutput(output));
    Ok(json!({ "accepted": call_id }))
}

fn prompt(In(input): In<Option<Value>>, mut queue: ResMut<PromptQueue>) -> BrpResult {
    #[derive(Deserialize)]
    struct Params {
        text: String,
    }
    let Params { text } = params(input)?;
    queue.0.push_back(text);
    Ok(json!({ "queued": queue.0.len() }))
}

fn status(
    In(_): In<Option<Value>>,
    active: Res<ActiveModel>,
    turn: Res<Turn>,
    queue: Res<PromptQueue>,
    registry: Res<ToolRegistry>,
    transcript: Res<Transcript>,
) -> BrpResult {
    Ok(json!({
        "pid": std::process::id(),
        "model": active.label(),
        "busy": !turn.is_idle(),
        "queued": queue.0.len(),
        "tools": registry.0.keys().collect::<Vec<_>>(),
        "transcript": transcript.0,
    }))
}
