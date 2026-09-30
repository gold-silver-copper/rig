//! External plugins over the Bevy Remote Protocol. Besides Bevy's built-in
//! methods, the agent serves two:
//!
//! - `agent.serve+watch` with `{"tools": [{"name", "description", "parameters"}]}`
//!   registers those tools and streams their calls to the plugin as
//!   `{"call_id", "tool", "arguments"}` events. Sending it again, for example
//!   after a reload closed the stream, re-registers the same tools.
//! - `agent.tool_result` with `{"call_id", "output"}` answers a call.
//! - `agent.prompt` with `{"text"}` queues a prompt, as typing it would.
//! - `agent.transcript` with `{"since"}` returns the transcript entries from
//!   that index on, and whether the agent is busy.
//!
//! The port is chosen once and kept across reloads.

use std::collections::HashMap;
use std::net::TcpListener;
use std::time::{Duration, Instant};

use async_channel::Sender;
use bevy::prelude::*;
use bevy_remote::http::RemoteHttpPlugin;
use bevy_remote::{BrpError, BrpResult, RemotePlugin, error_codes};
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolCall;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::agent::{ToolReply, Tools};
use crate::session::Session;

/// How long a call waits for its plugin to (re)connect.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

pub struct BrpPlugin {
    pub port: u16,
}

impl Plugin for BrpPlugin {
    fn build(&self, app: &mut App) {
        let definitions = app.world().resource::<Session>().external_tools.clone();
        let mut tools = app.world_mut().get_resource_or_init::<Tools>();
        for definition in &definitions {
            tools.register_external(definition);
        }
        app.init_resource::<External>()
            .add_plugins((
                RemotePlugin::default()
                    .with_watching_method_main("agent.serve+watch", serve)
                    .with_method_main("agent.tool_result", tool_result)
                    .with_method_main("agent.prompt", prompt)
                    .with_method_main("agent.transcript", transcript),
                RemoteHttpPlugin::default().with_port(self.port),
            ))
            .add_systems(Update, expire);
    }
}

/// A free local port, for the first start.
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|address| address.port())
        .unwrap_or(0)
}

/// Calls to external tools that have not been answered yet.
#[derive(Resource, Default)]
pub struct External {
    calls: Vec<Pending>,
    /// When a plugin last polled for each tool.
    seen: HashMap<String, Instant>,
    next_id: u64,
}

struct Pending {
    id: String,
    tool: String,
    arguments: Value,
    delivered: bool,
    queued: Instant,
    reply: Sender<String>,
}

impl External {
    pub fn enqueue(&mut self, call: &ToolCall) -> ToolReply {
        let (reply, receiver) = async_channel::bounded(1);
        self.next_id += 1;
        self.calls.push(Pending {
            id: format!("call-{}-{}", std::process::id(), self.next_id),
            tool: call.function.name.to_string(),
            arguments: call.function.arguments.clone(),
            delivered: false,
            queued: Instant::now(),
            reply,
        });
        ToolReply::Pending(receiver)
    }
}

#[derive(Deserialize)]
struct ServeParams {
    tools: Vec<ToolDefinition>,
}

fn parse<T: serde::de::DeserializeOwned>(params: Option<Value>) -> BrpResult<T> {
    serde_json::from_value(params.unwrap_or(Value::Null)).map_err(|error| BrpError {
        code: error_codes::INVALID_PARAMS,
        message: error.to_string(),
        data: None,
    })
}

/// Runs every frame while a plugin's stream is open.
fn serve(
    In(params): In<Option<Value>>,
    mut external: ResMut<External>,
    mut tools: ResMut<Tools>,
    mut session: ResMut<Session>,
) -> BrpResult<Option<Value>> {
    let ServeParams { tools: served } = parse(params)?;
    for definition in &served {
        external
            .seen
            .insert(definition.name.clone(), Instant::now());
        if tools.register_external(definition) {
            session
                .external_tools
                .retain(|known| known.name != definition.name);
            session.external_tools.push(definition.clone());
        }
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
        "call_id": call.id,
        "tool": call.tool,
        "arguments": call.arguments,
    })))
}

#[derive(Deserialize)]
struct ResultParams {
    call_id: String,
    output: String,
}

fn tool_result(In(params): In<Option<Value>>, mut external: ResMut<External>) -> BrpResult {
    let ResultParams { call_id, output } = parse(params)?;
    let Some(index) = external.calls.iter().position(|call| call.id == call_id) else {
        return Err(BrpError {
            code: error_codes::INVALID_PARAMS,
            message: format!("no pending call `{call_id}`"),
            data: None,
        });
    };
    let call = external.calls.swap_remove(index);
    let _ = call.reply.try_send(output);
    Ok(Value::Null)
}

#[derive(Deserialize)]
struct PromptParams {
    text: String,
}

fn prompt(In(params): In<Option<Value>>, mut session: ResMut<Session>) -> BrpResult {
    let PromptParams { text } = parse(params)?;
    session.queue.push_back(text);
    Ok(json!({"queued": session.queue.len()}))
}

#[derive(Deserialize, Default)]
struct TranscriptParams {
    #[serde(default)]
    since: usize,
}

fn transcript(In(params): In<Option<Value>>, session: Res<Session>) -> BrpResult {
    let TranscriptParams { since } = params
        .map(|params| parse(Some(params)))
        .transpose()?
        .unwrap_or_default();
    let entries = session.transcript.get(since..).unwrap_or_default();
    Ok(json!({
        "entries": entries,
        "next": session.transcript.len(),
        "busy": session.busy() || !session.queue.is_empty(),
        "model": session.model,
    }))
}

/// Fails calls whose plugin does not connect in time, and forgets cancelled ones.
fn expire(mut external: ResMut<External>) {
    let External { calls, seen, .. } = &mut *external;
    calls.retain(|call| {
        let connected = seen
            .get(&call.tool)
            .is_some_and(|at| at.elapsed() < Duration::from_secs(1));
        if !call.delivered && !connected && call.queued.elapsed() > CONNECT_TIMEOUT {
            let _ = call.reply.try_send(format!(
                "error: no external plugin serving `{}` is connected",
                call.tool
            ));
            return false;
        }
        !call.reply.is_closed()
    });
}
