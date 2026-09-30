//! BRP plugins: other processes extend the agent over the Bevy Remote
//! Protocol, on a port that stays the same across reloads.
//!
//! - `rig_pi.serve+watch` `{plugin, session, tools: [{name, description, parameters}]}`
//!   registers the plugin's tools, then streams the calls the model makes to
//!   them as `{call_id, name, arguments}`. `session` is any string unique to
//!   the connection; reconnecting with a new one re-registers cleanly.
//! - `rig_pi.tool_result` `{call_id, output}` or `{call_id, error}` answers a call.
//! - `rig_pi.prompt` `{text}` queues a prompt; `rig_pi.status` describes the agent.
//!
//! The built-in `world.*` methods work too.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_remote::http::RemoteHttpPlugin;
use bevy_remote::{BrpError, BrpResult, RemotePlugin, error_codes};
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::Env;
use crate::session::{Kind, Phase, RemoteTool, Session};
use crate::tools::{Run, Tools};

pub struct BrpPlugin {
    pub port: u16,
}

impl Plugin for BrpPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Remote>()
            .add_plugins((
                RemotePlugin::default()
                    .with_method_main("rig_pi.prompt", prompt)
                    .with_method_main("rig_pi.status", status)
                    .with_method_main("rig_pi.tool_result", tool_result)
                    .with_watching_method_main("rig_pi.serve+watch", serve),
                RemoteHttpPlugin::default().with_port(self.port),
            ))
            .add_systems(Startup, restore_tools);
    }
}

/// Connected plugins, the calls waiting for them, and their answers.
#[derive(Resource, Default)]
pub struct Remote {
    plugins: HashMap<String, Connection>,
    calls: HashMap<String, VecDeque<Value>>,
    results: HashMap<String, Result<String, String>>,
}

#[derive(Default)]
struct Connection {
    session: String,
    /// Sessions a reconnect replaced; their streams stay silent.
    retired: HashSet<String>,
    last_seen: Option<Instant>,
}

impl Remote {
    pub fn enqueue(&mut self, plugin: &str, id: &str, name: &str, arguments: Value) {
        self.calls
            .entry(plugin.to_owned())
            .or_default()
            .push_back(json!({"call_id": id, "name": name, "arguments": arguments}));
    }

    pub fn take_result(&mut self, id: &str) -> Option<Result<String, String>> {
        self.results.remove(id)
    }

    pub fn connected(&self, plugin: &str) -> bool {
        self.plugins
            .get(plugin)
            .and_then(|c| c.last_seen)
            .is_some_and(|seen| seen.elapsed() < Duration::from_secs(1))
    }
}

/// Offer the model the tools BRP plugins registered before a reload; calls
/// wait for the plugin to reconnect.
fn restore_tools(session: Res<Session>, mut tools: ResMut<Tools>) {
    for tool in &session.remote_tools {
        tools.insert(tool.def.clone(), Run::Remote(tool.plugin.clone()));
    }
}

fn params<T: DeserializeOwned>(params: Option<Value>) -> BrpResult<T> {
    serde_json::from_value(params.unwrap_or(Value::Null)).map_err(|error| BrpError {
        code: error_codes::INVALID_PARAMS,
        message: error.to_string(),
        data: None,
    })
}

#[derive(Deserialize)]
struct Serve {
    plugin: String,
    #[serde(default)]
    session: String,
    #[serde(default)]
    tools: Vec<ToolSpec>,
}

#[derive(Deserialize)]
struct ToolSpec {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default = "empty_object")]
    parameters: Value,
}

fn empty_object() -> Value {
    json!({"type": "object", "properties": {}})
}

/// Runs every frame for every open `rig_pi.serve+watch` stream.
fn serve(
    In(raw): In<Option<Value>>,
    mut remote: ResMut<Remote>,
    mut session: ResMut<Session>,
    mut tools: ResMut<Tools>,
) -> BrpResult<Option<Value>> {
    let serve: Serve = params(raw)?;
    let connection = remote.plugins.entry(serve.plugin.clone()).or_default();
    if connection.retired.contains(&serve.session) {
        return Ok(None);
    }
    connection.last_seen = Some(Instant::now());
    if connection.session != serve.session {
        let old = std::mem::replace(&mut connection.session, serve.session.clone());
        connection.retired.insert(old);
        let names = register(&serve, &mut session, &mut tools)?;
        return Ok(Some(json!({"registered": names})));
    }
    Ok(remote
        .calls
        .get_mut(&serve.plugin)
        .and_then(VecDeque::pop_front))
}

/// Replace the plugin's tools with the ones it just sent.
fn register(serve: &Serve, session: &mut Session, tools: &mut Tools) -> BrpResult<Vec<String>> {
    let plugin = &serve.plugin;
    if let Some(spec) = serve.tools.iter().find(|spec| {
        tools
            .0
            .get(&spec.name)
            .is_some_and(|tool| !matches!(&tool.run, Run::Remote(owner) if owner == plugin))
    }) {
        return Err(BrpError {
            code: error_codes::INVALID_PARAMS,
            message: format!(
                "tool `{}` is already provided by the agent or another plugin",
                spec.name
            ),
            data: None,
        });
    }
    tools
        .0
        .retain(|_, tool| !matches!(&tool.run, Run::Remote(owner) if owner == plugin));
    let before: Vec<String> = session
        .remote_tools
        .iter()
        .filter(|tool| &tool.plugin == plugin)
        .map(|tool| tool.def.name.clone())
        .collect();
    session.remote_tools.retain(|tool| &tool.plugin != plugin);
    let mut names = Vec::new();
    for spec in &serve.tools {
        let name = ToolName::new(spec.name.as_str()).map_err(|error| BrpError {
            code: error_codes::INVALID_PARAMS,
            message: error.to_string(),
            data: None,
        })?;
        let def = ToolDefinition::new(name, &spec.description, spec.parameters.clone());
        tools.insert(def.clone(), Run::Remote(plugin.clone()));
        session.remote_tools.push(RemoteTool {
            plugin: plugin.clone(),
            def,
        });
        names.push(spec.name.clone());
    }
    let verb = if before == names {
        "reconnected"
    } else {
        "registered"
    };
    session.log(
        Kind::Info,
        format!("BRP plugin `{plugin}` {verb}: {}", names.join(", ")),
    );
    Ok(names)
}

#[derive(Deserialize)]
struct ToolAnswer {
    call_id: String,
    output: Option<String>,
    error: Option<String>,
}

fn tool_result(In(raw): In<Option<Value>>, mut remote: ResMut<Remote>) -> BrpResult {
    let answer: ToolAnswer = params(raw)?;
    let result = match answer.error {
        Some(error) => Err(error),
        None => Ok(answer.output.unwrap_or_default()),
    };
    remote.results.insert(answer.call_id, result);
    Ok(json!({"ok": true}))
}

#[derive(Deserialize)]
struct Prompt {
    text: String,
}

fn prompt(In(raw): In<Option<Value>>, mut session: ResMut<Session>) -> BrpResult {
    let prompt: Prompt = params(raw)?;
    session.queue.push_back(prompt.text);
    Ok(json!({"queued": session.queue.len()}))
}

fn status(
    In(_): In<Option<Value>>,
    session: Res<Session>,
    tools: Res<Tools>,
    remote: Res<Remote>,
    env: Res<Env>,
) -> BrpResult {
    let phase = match &session.phase {
        Phase::Idle => "idle",
        Phase::Model => "model",
        Phase::Tools { .. } => "tools",
    };
    let plugins: Vec<&String> = remote
        .plugins
        .keys()
        .filter(|plugin| remote.connected(plugin))
        .collect();
    let recent = session.transcript.len().saturating_sub(30);
    Ok(json!({
        "pid": std::process::id(),
        "port": env.port,
        "model": session.model.to_string(),
        "phase": phase,
        "queue": session.queue,
        "messages": session.messages.len(),
        "tools": tools.0.keys().collect::<Vec<_>>(),
        "plugins": plugins,
        "transcript": session.transcript[recent..]
            .iter()
            .map(|entry| json!({"kind": format!("{:?}", entry.kind), "text": entry.text}))
            .collect::<Vec<_>>(),
    }))
}
