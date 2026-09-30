//! Out-of-process plugins over the Bevy Remote Protocol.
//!
//! The stock BRP methods already let another process inspect and edit the
//! agent's world, including spawning [`ToolSpec`] + [`RemoteTool`] entities
//! and inserting [`ToolOutput`]s. The `rigpi/*` methods wrap that into a
//! small plugin API: register a tool, take the calls the model makes to it,
//! answer them, submit prompts, and read the transcript.

use std::net::{Ipv4Addr, TcpListener};

use bevy::prelude::*;
use bevy_remote::http::RemoteHttpPlugin;
use bevy_remote::{BrpError, BrpResult, RemotePlugin, error_codes};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::agent::{Agent, Cancel, Submit, Transcript, Turn};
use crate::tools::{Claimed, RemoteTool, ToolCall, ToolOutput, ToolSpec};

/// The port BRP listens on (127.0.0.1 only).
#[derive(Resource)]
pub struct BrpPort(pub u16);

/// `requested`, or a free port from the OS.
pub fn pick_port(requested: Option<u16>) -> u16 {
    requested
        .or_else(|| {
            let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).ok()?;
            Some(listener.local_addr().ok()?.port())
        })
        .unwrap_or(0)
}

pub struct BrpPlugin(pub u16);

impl Plugin for BrpPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(BrpPort(self.0))
            .add_plugins(
                RemotePlugin::default()
                    .with_method_main("rigpi/prompt", prompt)
                    .with_method_main("rigpi/cancel", cancel)
                    .with_method_main("rigpi/register_tool", register_tool)
                    .with_method_main("rigpi/unregister_tool", unregister_tool)
                    .with_method_main("rigpi/take_calls", take_calls)
                    .with_method_main("rigpi/complete_call", complete_call)
                    .with_method_main("rigpi/transcript", transcript)
                    .with_method_main("rigpi/state", state),
            )
            .add_plugins(
                RemoteHttpPlugin::default()
                    .with_address(Ipv4Addr::LOCALHOST)
                    .with_port(self.0),
            );
    }
}

fn parse<T: for<'de> Deserialize<'de>>(params: Option<Value>) -> Result<T, BrpError> {
    serde_json::from_value(params.unwrap_or(Value::Null)).map_err(|error| BrpError {
        code: error_codes::INVALID_PARAMS,
        message: error.to_string(),
        data: None,
    })
}

/// `{ "text": "..." }`: submit a prompt as if typed.
fn prompt(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    #[derive(Deserialize)]
    struct Params {
        text: String,
    }
    let Params { text } = parse(params)?;
    world.write_message(Submit(text));
    Ok(Value::Null)
}

/// Stop the current turn, as Esc does.
fn cancel(In(_): In<Option<Value>>, world: &mut World) -> BrpResult {
    world.write_message(Cancel);
    Ok(Value::Null)
}

/// `{ "name", "description", "parameters": <JSON Schema> }`: add a tool
/// this process will answer, replacing any remote tool of the same name.
fn register_tool(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    #[derive(Deserialize)]
    struct Params {
        name: String,
        description: String,
        #[serde(default = "empty_object")]
        parameters: Value,
    }
    let Params {
        name,
        description,
        parameters,
    } = parse(params)?;
    despawn_remote(world, &name);
    let entity = world
        .spawn((
            ToolSpec {
                name,
                description,
                parameters: parameters.to_string(),
            },
            RemoteTool,
        ))
        .id();
    Ok(json!({ "entity": entity.to_bits() }))
}

fn empty_object() -> Value {
    json!({ "type": "object", "properties": {} })
}

/// `{ "name" }`: remove a remote tool.
fn unregister_tool(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    #[derive(Deserialize)]
    struct Params {
        name: String,
    }
    let Params { name } = parse(params)?;
    Ok(json!({ "removed": despawn_remote(world, &name) }))
}

fn despawn_remote(world: &mut World, name: &str) -> usize {
    let mut query = world.query_filtered::<(Entity, &ToolSpec), With<RemoteTool>>();
    let stale: Vec<Entity> = query
        .iter(world)
        .filter(|(_, spec)| spec.name == name)
        .map(|(entity, _)| entity)
        .collect();
    for &entity in &stale {
        world.despawn(entity);
    }
    stale.len()
}

/// `{ "tools": [names]? }`: claim the unanswered calls to remote tools
/// (optionally only these), returning `[{ call, name, arguments }]`.
fn take_calls(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    #[derive(Deserialize, Default)]
    struct Params {
        tools: Option<Vec<String>>,
    }
    let Params { tools } = if params.is_some() {
        parse(params)?
    } else {
        Params::default()
    };
    let mut remote = world.query_filtered::<&ToolSpec, With<RemoteTool>>();
    let remote: Vec<String> = remote
        .iter(world)
        .map(|spec| spec.name.clone())
        .filter(|name| tools.as_ref().is_none_or(|tools| tools.contains(name)))
        .collect();
    let mut calls =
        world.query_filtered::<(Entity, &ToolCall), (Without<ToolOutput>, Without<Claimed>)>();
    let taken: Vec<(Entity, ToolCall)> = calls
        .iter(world)
        .filter(|(_, call)| remote.contains(&call.name))
        .map(|(entity, call)| (entity, call.clone()))
        .collect();
    let mut out = Vec::new();
    for (entity, call) in taken {
        world.entity_mut(entity).insert(Claimed);
        out.push(json!({
            "call": entity.to_bits(),
            "name": call.name,
            "arguments": serde_json::from_str::<Value>(&call.arguments).unwrap_or(Value::Null),
        }));
    }
    Ok(Value::Array(out))
}

/// `{ "call", "output", "is_error"? }`: answer a call.
fn complete_call(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    #[derive(Deserialize)]
    struct Params {
        call: u64,
        output: String,
        #[serde(default)]
        is_error: bool,
    }
    let Params {
        call,
        output,
        is_error,
    } = parse(params)?;
    let entity = Entity::try_from_bits(call).ok_or_else(|| BrpError {
        code: error_codes::INVALID_PARAMS,
        message: format!("{call} is not an entity"),
        data: None,
    })?;
    let mut entity_mut = world
        .get_entity_mut(entity)
        .map_err(|_| BrpError::entity_not_found(entity))?;
    if !entity_mut.contains::<ToolCall>() {
        return Err(BrpError::component_not_present("ToolCall", entity));
    }
    entity_mut.insert(ToolOutput {
        content: output,
        is_error,
    });
    Ok(Value::Null)
}

/// The transcript so far.
fn transcript(In(_): In<Option<Value>>, world: &mut World) -> BrpResult {
    serde_json::to_value(&world.resource::<Transcript>().entries).map_err(|error| BrpError {
        code: error_codes::INTERNAL_ERROR,
        message: error.to_string(),
        data: None,
    })
}

/// `{ "model", "turn", "tools", "usage" }`.
fn state(In(_): In<Option<Value>>, world: &mut World) -> BrpResult {
    let tools: Vec<String> = world
        .query::<&ToolSpec>()
        .iter(world)
        .map(|spec| spec.name.clone())
        .collect();
    let agent = world.resource::<Agent>();
    let turn = match agent.turn {
        Turn::Idle => "idle",
        Turn::Thinking { .. } => "thinking",
        Turn::Tools { .. } => "tools",
    };
    Ok(json!({ "model": agent.model_name, "turn": turn, "tools": tools, "usage": agent.usage }))
}
