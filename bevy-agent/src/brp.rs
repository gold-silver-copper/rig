//! External plugins over the Bevy Remote Protocol.
//!
//! A plugin process registers tools with `agent.register_tools`, receives
//! calls on the `agent.tool_calls+watch` stream, and answers each with
//! `agent.tool_result`. `agent.prompt` queues a prompt and `agent.status`
//! reports state; Bevy's built-in `world.*` methods work as usual.
//!
//! Across a reload the port stays the same and registered tools stay listed.
//! When its stream ends, a plugin reconnects and registers again; calls it
//! was sent but did not answer are sent again.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_remote::http::RemoteHttpPlugin;
use bevy_remote::{BrpError, BrpResult, RemotePlugin, error_codes};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::session::{Paths, RemoteTool, Session};
use crate::tools::{Handler, Slot, Tools, definition, fill};

/// How long a call waits for its plugin before it fails.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(120);

pub struct BrpPlugin {
    pub port: u16,
}

impl Plugin for BrpPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RemoteCalls>()
            .add_plugins(
                RemotePlugin::default()
                    .with_method_main("agent.register_tools", register_tools)
                    .with_watching_method_main("agent.tool_calls+watch", tool_calls)
                    .with_method_main("agent.tool_result", tool_result)
                    .with_method_main("agent.prompt", prompt)
                    .with_method_main("agent.status", status),
            )
            .add_plugins(RemoteHttpPlugin::default().with_port(self.port))
            .add_systems(Update, expire_calls);
    }
}

pub struct RemoteCall {
    pub id: String,
    pub plugin: String,
    pub tool: String,
    pub arguments: Value,
    pub sent: bool,
    pub since: Instant,
    pub slot: Slot<String>,
}

#[derive(Resource, Default)]
pub struct RemoteCalls(pub Vec<RemoteCall>);

fn parse<T: DeserializeOwned>(params: Option<Value>) -> Result<T, BrpError> {
    serde_json::from_value(params.unwrap_or(Value::Null)).map_err(|e| BrpError {
        code: error_codes::INVALID_PARAMS,
        message: e.to_string(),
        data: None,
    })
}

#[derive(Deserialize)]
struct Registration {
    plugin: String,
    tools: Vec<ToolSpec>,
}

#[derive(Deserialize)]
struct ToolSpec {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default = "empty_schema")]
    parameters: Value,
}

fn empty_schema() -> Value {
    json!({"type": "object", "properties": {}})
}

/// Idempotent: registering again replaces the plugin's tools.
fn register_tools(
    In(params): In<Option<Value>>,
    mut tools: ResMut<Tools>,
    mut session: ResMut<Session>,
    mut calls: ResMut<RemoteCalls>,
    paths: Res<Paths>,
) -> BrpResult {
    let registration: Registration = parse(params)?;
    let plugin = registration.plugin;
    for spec in &registration.tools {
        let taken = tools.0.get(&spec.name).is_some_and(
            |tool| !matches!(&tool.handler, Handler::Remote { plugin: owner } if *owner == plugin),
        );
        if spec.name.is_empty() || taken {
            return Err(BrpError {
                code: error_codes::INVALID_PARAMS,
                message: format!("tool name `{}` is empty or taken", spec.name),
                data: None,
            });
        }
    }
    let registered = |session: &Session| -> Vec<_> {
        session.remote_tools.iter().filter(|t| t.plugin == plugin).map(|t| t.def.clone()).collect()
    };
    let before = registered(&session);
    tools
        .0
        .retain(|_, tool| !matches!(&tool.handler, Handler::Remote { plugin: owner } if *owner == plugin));
    session.remote_tools.retain(|tool| tool.plugin != plugin);
    for spec in registration.tools {
        let def = definition(&spec.name, &spec.description, spec.parameters);
        tools.insert(def.clone(), Handler::Remote { plugin: plugin.clone() });
        session.remote_tools.push(RemoteTool { plugin: plugin.clone(), def });
    }
    // A plugin that registers again has lost its connection: resend its open calls.
    for call in calls.0.iter_mut().filter(|call| call.plugin == plugin) {
        call.sent = false;
    }
    let after = registered(&session);
    let names: Vec<_> = after.iter().map(|def| def.name.clone()).collect();
    if before != after {
        session.info(format!("plugin `{plugin}` registered: {}", names.join(", ")));
    }
    session.save(&paths.session());
    Ok(json!({ "registered": names }))
}

#[derive(Deserialize)]
struct PluginName {
    plugin: String,
}

/// Watching method: each frame, stream the plugin's calls it has not been sent.
fn tool_calls(In(params): In<Option<Value>>, mut calls: ResMut<RemoteCalls>) -> BrpResult<Option<Value>> {
    let PluginName { plugin } = parse(params)?;
    let fresh: Vec<Value> = calls
        .0
        .iter_mut()
        .filter(|call| call.plugin == plugin && !call.sent)
        .map(|call| {
            call.sent = true;
            json!({ "id": call.id, "tool": call.tool, "arguments": call.arguments })
        })
        .collect();
    Ok((!fresh.is_empty()).then(|| json!(fresh)))
}

#[derive(Deserialize)]
struct Answer {
    id: String,
    output: String,
    #[serde(default)]
    is_error: bool,
}

fn tool_result(In(params): In<Option<Value>>, mut calls: ResMut<RemoteCalls>) -> BrpResult {
    let answer: Answer = parse(params)?;
    let Some(index) = calls.0.iter().position(|call| call.id == answer.id) else {
        return Err(BrpError {
            code: error_codes::INVALID_PARAMS,
            message: format!("no open call `{}`", answer.id),
            data: None,
        });
    };
    let call = calls.0.swap_remove(index);
    let output = match answer.is_error {
        true => format!("error: {}", answer.output),
        false => answer.output,
    };
    fill(&call.slot, output);
    Ok(Value::Null)
}

#[derive(Deserialize)]
struct Prompt {
    text: String,
}

fn prompt(In(params): In<Option<Value>>, mut session: ResMut<Session>, paths: Res<Paths>) -> BrpResult {
    let Prompt { text } = parse(params)?;
    session.queue.push_back(text);
    session.save(&paths.session());
    Ok(json!({ "queued": session.queue.len() }))
}

fn status(In(_): In<Option<Value>>, session: Res<Session>, tools: Res<Tools>) -> BrpResult {
    Ok(json!({
        "pid": std::process::id(),
        "model": session.model.to_string(),
        "busy": session.turn.is_some(),
        "queued": session.queue.len(),
        "messages": session.history.len(),
        "tools": tools.0.keys().collect::<Vec<_>>(),
    }))
}

fn expire_calls(mut calls: ResMut<RemoteCalls>) {
    calls.0.retain(|call| {
        let expired = call.since.elapsed() > ANSWER_TIMEOUT;
        if expired {
            fill(
                &call.slot,
                format!("error: plugin `{}` did not answer within {}s", call.plugin, ANSWER_TIMEOUT.as_secs()),
            );
        }
        !expired
    });
}
