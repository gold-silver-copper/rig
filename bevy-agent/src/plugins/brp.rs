//! The Bevy Remote Protocol server, and external tools over it. An external
//! process is a plain BRP client:
//!
//! - `agent.serve+watch` with `{"tools": [{"name", "description", "parameters"}]}`
//!   registers those tools and streams their calls as
//!   `{"call_id", "tool", "arguments"}` events. Sending it again, for example
//!   after a reload closed the stream, registers the same tools again.
//! - `agent.tool_result` with `{"call_id", "output"}` answers a call.
//!
//! The port is chosen once and kept across reloads, and so are the tool
//! definitions, so the model keeps seeing them while a plugin reconnects.

use std::collections::HashMap;
use std::net::TcpListener;
use std::time::{Duration, Instant};

use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use bevy_remote::http::RemoteHttpPlugin;
use bevy_remote::{
    BrpError, BrpResult, RemoteMethodSystemId, RemoteMethods, RemotePlugin, error_codes,
};
use rig_core::completion::ToolDefinition;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::glue::{Tool, ToolCall, ToolOutput, Tools};
use crate::{BrpPort, ProcessState};

/// How long a call waits for its plugin to (re)connect.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Serves BRP on `port`, or on the port the process that reloaded used, or on a free one.
pub struct BrpPlugin {
    pub port: Option<u16>,
}

impl Plugin for BrpPlugin {
    fn build(&self, app: &mut App) {
        let saved = app.world().resource::<ProcessState>().0.clone();
        let port = self
            .port
            .or_else(|| {
                saved
                    .get("brp_port")
                    .and_then(Value::as_u64)
                    .and_then(|port| u16::try_from(port).ok())
            })
            .unwrap_or_else(free_port);
        app.world_mut()
            .resource_mut::<ProcessState>()
            .0
            .insert("brp_port".into(), json!(port));
        let definitions: Vec<ToolDefinition> = saved
            .get("external_tools")
            .and_then(|tools| serde_json::from_value(tools.clone()).ok())
            .unwrap_or_default();
        let mut external = External::default();
        for definition in definitions {
            external.register(app.world_mut(), definition);
        }
        app.insert_resource(BrpPort(port))
            .insert_resource(external)
            .add_plugins((
                RemotePlugin::default()
                    .with_watching_method_main("agent.serve+watch", serve)
                    .with_method_main("agent.tool_result", tool_result),
                RemoteHttpPlugin::default().with_port(port),
            ))
            .add_systems(Update, (queue_calls, expire));
    }
}

/// Adds a BRP method to the server the [`BrpPlugin`] runs.
pub fn add_method<M>(
    app: &mut App,
    name: &str,
    handler: impl IntoSystem<In<Option<Value>>, BrpResult, M> + 'static,
) {
    let id: SystemId<In<Option<Value>>, BrpResult> = app.world_mut().register_system(handler);
    if let Some(mut methods) = app.world_mut().get_resource_mut::<RemoteMethods>() {
        methods.insert(name, RemoteMethodSystemId::Instant(id));
    }
}

/// Parses BRP parameters, reporting a mismatch as invalid params.
pub fn params<T: serde::de::DeserializeOwned>(params: Option<Value>) -> BrpResult<T> {
    serde_json::from_value(params.unwrap_or(Value::Null)).map_err(|error| BrpError {
        code: error_codes::INVALID_PARAMS,
        message: error.to_string(),
        data: None,
    })
}

/// A free local port, for the first start.
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|address| address.port())
        .unwrap_or(0)
}

/// External tools, and calls to them that have not been answered.
#[derive(Resource, Default)]
struct External {
    tools: HashMap<String, ToolDefinition>,
    calls: Vec<Pending>,
    /// When a plugin last polled for each tool.
    seen: HashMap<String, Instant>,
}

struct Pending {
    entity: Entity,
    tool: String,
    arguments: Value,
    delivered: bool,
    queued: Instant,
}

impl External {
    /// Adds or replaces a tool; returns whether anything changed.
    fn register(&mut self, world: &mut World, definition: ToolDefinition) -> bool {
        if self.tools.get(&definition.name) == Some(&definition) {
            return false;
        }
        let mut tools = world.resource_mut::<Tools>();
        if tools
            .0
            .get(&definition.name)
            .is_some_and(|tool| tool.run.is_some())
        {
            // A native tool of the same name wins.
            return false;
        }
        tools.0.insert(
            definition.name.clone(),
            Tool {
                definition: definition.clone(),
                run: None,
            },
        );
        self.tools.insert(definition.name.clone(), definition);
        let saved: Vec<&ToolDefinition> = self.tools.values().collect();
        let saved = json!(saved);
        world
            .resource_mut::<ProcessState>()
            .0
            .insert("external_tools".into(), saved);
        true
    }
}

fn queue_calls(calls: Query<(Entity, &ToolCall), Added<ToolCall>>, mut external: ResMut<External>) {
    for (entity, ToolCall(call)) in &calls {
        let name = call.function.name.as_str();
        if external.tools.contains_key(name) {
            external.calls.push(Pending {
                entity,
                tool: name.to_owned(),
                arguments: call.function.arguments.clone(),
                delivered: false,
                queued: Instant::now(),
            });
        }
    }
}

#[derive(Deserialize)]
struct ServeParams {
    tools: Vec<ToolDefinition>,
}

/// Runs every frame while a plugin's stream is open.
fn serve(In(raw): In<Option<Value>>, world: &mut World) -> BrpResult<Option<Value>> {
    let ServeParams { tools: served } = params(raw)?;
    world.resource_scope(|world, mut external: Mut<External>| {
        for definition in &served {
            external
                .seen
                .insert(definition.name.clone(), Instant::now());
            external.register(world, definition.clone());
        }
        let Some(call) = external
            .calls
            .iter_mut()
            .find(|call| !call.delivered && served.iter().any(|tool| tool.name == call.tool))
        else {
            return Ok(None);
        };
        call.delivered = true;
        Ok(Some(json!({
            "call_id": call.entity.to_bits().to_string(),
            "tool": call.tool,
            "arguments": call.arguments,
        })))
    })
}

#[derive(Deserialize)]
struct ResultParams {
    call_id: String,
    output: String,
}

fn tool_result(
    In(raw): In<Option<Value>>,
    mut external: ResMut<External>,
    mut commands: Commands,
) -> BrpResult {
    let ResultParams { call_id, output } = params(raw)?;
    let Some(index) = external
        .calls
        .iter()
        .position(|call| call.entity.to_bits().to_string() == call_id)
    else {
        return Err(BrpError {
            code: error_codes::INVALID_PARAMS,
            message: format!("no pending call `{call_id}`"),
            data: None,
        });
    };
    let call = external.calls.swap_remove(index);
    if let Ok(mut entity) = commands.get_entity(call.entity) {
        entity.insert(ToolOutput(output));
    }
    Ok(Value::Null)
}

/// Fails calls whose plugin does not connect in time, and forgets calls
/// that were cancelled.
fn expire(
    mut external: ResMut<External>,
    calls: Query<(), With<ToolCall>>,
    mut commands: Commands,
) {
    let External {
        calls: pending,
        seen,
        ..
    } = &mut *external;
    pending.retain(|call| {
        if !calls.contains(call.entity) {
            return false;
        }
        let connected = seen
            .get(&call.tool)
            .is_some_and(|at| at.elapsed() < Duration::from_secs(1));
        if !call.delivered && !connected && call.queued.elapsed() > CONNECT_TIMEOUT {
            commands.entity(call.entity).insert(ToolOutput(format!(
                "error: no external plugin serving `{}` is connected",
                call.tool
            )));
            return false;
        }
        true
    });
}
