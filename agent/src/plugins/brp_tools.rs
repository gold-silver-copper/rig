//! BRP tools: other processes give the model tools over the Bevy Remote
//! Protocol.
//!
//! - `rig_pi.serve+watch` `{plugin, session, tools: [{name, description, parameters}]}`
//!   registers the plugin's tools, then streams the calls the model makes to
//!   them as `{call_id, name, arguments}`. `session` is any string unique to
//!   the connection; reconnecting with a new one re-registers cleanly.
//! - `rig_pi.tool_result` `{call_id, output}` or `{call_id, error}` answers.
//!
//! Registered tools are saved, so after a reload the model can call them
//! while their plugin reconnects; a call waits for it.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_remote::BrpResult;
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use rig_core::tool::ToolOutput;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Env;
use crate::brp::{add_method, add_watching_method, invalid, params};
use crate::glue::{AgentSet, Call, Done, Ready, Tool, Tools};
use crate::session::{EntryKind, Focused, Transcript};

pub struct BrpToolsPlugin;

impl Plugin for BrpToolsPlugin {
    fn build(&self, app: &mut App) {
        let env = app.world().resource::<Env>().clone();
        let saved: Vec<RemoteTool> = std::fs::read(env.state.join("brp-tools.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        let mut tools = app.world_mut().get_resource_or_init::<Tools>();
        for tool in &saved {
            tools.add(Tool::World(tool.def.clone()));
        }
        app.insert_resource(Remote {
            tools: saved,
            ..default()
        });
        add_watching_method(app, "rig_pi.serve+watch", serve);
        add_method(app, "rig_pi.tool_result", tool_result);
        app.add_systems(Update, forward_calls.after(AgentSet::Run));
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct RemoteTool {
    plugin: String,
    def: ToolDefinition,
}

/// Connected plugins, their tools, the calls waiting for them, and answers.
#[derive(Resource, Default)]
struct Remote {
    tools: Vec<RemoteTool>,
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
    fn owner(&self, tool: &str) -> Option<&str> {
        self.tools.iter().find(|t| t.def.name == tool).map(|t| t.plugin.as_str())
    }

    fn connected(&self, plugin: &str) -> bool {
        self.plugins
            .get(plugin)
            .and_then(|c| c.last_seen)
            .is_some_and(|seen| seen.elapsed() < Duration::from_secs(1))
    }
}

/// A call handed to a plugin, waiting for its answer.
#[derive(Component)]
struct Forwarded {
    id: String,
    plugin: String,
    since: Instant,
}

fn forward_calls(
    ready: Query<(Entity, &Call, Option<&Forwarded>), (With<Ready>, Without<Done>)>,
    mut remote: ResMut<Remote>,
    mut commands: Commands,
) {
    for (entity, call, forwarded) in &ready {
        let name = call.call.function.name.as_str();
        let Some(forwarded) = forwarded else {
            let Some(plugin) = remote.owner(name).map(str::to_owned) else {
                continue;
            };
            let id = call.call.id.to_string();
            remote.calls.entry(plugin.clone()).or_default().push_back(json!({
                "call_id": id, "name": name, "arguments": call.call.function.arguments,
            }));
            commands.entity(entity).insert(Forwarded {
                id,
                plugin,
                since: Instant::now(),
            });
            continue;
        };
        let waited = forwarded.since.elapsed();
        let outcome = match remote.results.remove(&forwarded.id) {
            Some(result) => Some(result.map(ToolOutput::text)),
            None if !remote.connected(&forwarded.plugin) && waited > Duration::from_secs(30) => {
                Some(Err(format!("BRP plugin `{}` is not connected", forwarded.plugin)))
            }
            None if waited > Duration::from_secs(300) => {
                Some(Err(format!("BRP plugin `{}` did not answer within 300s", forwarded.plugin)))
            }
            None => None,
        };
        if let Some(outcome) = outcome {
            commands.entity(entity).remove::<Forwarded>().try_insert(Done(outcome));
        }
    }
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
    mut tools: ResMut<Tools>,
    mut focused: Query<&mut Transcript, With<Focused>>,
    env: Res<Env>,
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
        let (names, verb) = register(&serve, &mut remote, &mut tools)?;
        let _ = std::fs::write(
            env.state.join("brp-tools.json"),
            serde_json::to_vec(&remote.tools).unwrap_or_default(),
        );
        if let Ok(mut transcript) = focused.single_mut() {
            transcript.log(
                EntryKind::Info,
                format!("BRP plugin `{}` {verb}: {}", serve.plugin, names.join(", ")),
            );
        }
        return Ok(Some(json!({"registered": names})));
    }
    Ok(remote.calls.get_mut(&serve.plugin).and_then(VecDeque::pop_front))
}

/// Replace the plugin's tools with the ones it just sent.
fn register(
    serve: &Serve,
    remote: &mut Remote,
    tools: &mut Tools,
) -> BrpResult<(Vec<String>, &'static str)> {
    let plugin = &serve.plugin;
    for spec in &serve.tools {
        ToolName::new(spec.name.as_str()).map_err(|error| invalid(error.to_string()))?;
        let owned_elsewhere = tools.0.contains_key(&spec.name)
            && remote.owner(&spec.name).is_none_or(|owner| owner != plugin);
        if owned_elsewhere {
            return Err(invalid(format!(
                "tool `{}` is already provided by the agent or another plugin",
                spec.name
            )));
        }
    }
    let before: Vec<String> = remote
        .tools
        .iter()
        .filter(|t| &t.plugin == plugin)
        .map(|t| t.def.name.clone())
        .collect();
    for name in &before {
        tools.0.remove(name);
    }
    remote.tools.retain(|t| &t.plugin != plugin);
    let mut names = Vec::new();
    for spec in &serve.tools {
        let def = ToolDefinition {
            name: spec.name.clone(),
            description: spec.description.clone(),
            parameters: spec.parameters.clone(),
        };
        tools.add(Tool::World(def.clone()));
        remote.tools.push(RemoteTool {
            plugin: plugin.clone(),
            def,
        });
        names.push(spec.name.clone());
    }
    let verb = if before == names { "reconnected" } else { "registered" };
    Ok((names, verb))
}

#[derive(Deserialize)]
struct Answer {
    call_id: String,
    output: Option<String>,
    error: Option<String>,
}

fn tool_result(In(raw): In<Option<Value>>, mut remote: ResMut<Remote>) -> BrpResult {
    let answer: Answer = params(raw)?;
    let result = match answer.error {
        Some(error) => Err(error),
        None => Ok(answer.output.unwrap_or_default()),
    };
    remote.results.insert(answer.call_id, result);
    Ok(json!({"ok": true}))
}
