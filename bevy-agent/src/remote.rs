//! Extension over the Bevy Remote Protocol. An external process registers a
//! tool, watches for its calls and posts results; it can also prompt the agent,
//! read the transcript and trigger a hot patch.

use std::net::TcpListener;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_remote::http::RemoteHttpPlugin;
use bevy_remote::{BrpError, BrpResult};
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use serde_json::{Value, json};

use crate::agent::{
    Conversation, Dispatched, Entry, InFlight, Tool, ToolCallRequest, ToolDone, Transcript,
};
use crate::hot::Hot;

#[derive(Resource)]
pub struct BrpPort(pub u16);

pub struct RemotePlugin {
    pub port: u16,
}

impl Plugin for RemotePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(BrpPort(self.port))
            .add_plugins(
                bevy_remote::RemotePlugin::default()
                    .with_method_main("agent.register_tool", register_tool)
                    .with_watching_method_main("agent.tool_calls+watch", watch_tool_calls)
                    .with_method_main("agent.tool_result", tool_result)
                    .with_method_main("agent.prompt", prompt)
                    .with_method_main("agent.transcript", transcript)
                    .with_method_main("agent.patch", patch),
            )
            .add_plugins(RemoteHttpPlugin::default().with_port(self.port));
    }
}

/// `AGENT_BRP_PORT`, or a port the OS reports free.
pub fn pick_port() -> anyhow::Result<u16> {
    if let Ok(port) = std::env::var("AGENT_BRP_PORT") {
        return Ok(port.parse()?);
    }
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

fn field<'a>(params: &'a Option<Value>, key: &str) -> BrpResult<&'a Value> {
    params
        .as_ref()
        .and_then(|p| p.get(key))
        .ok_or_else(|| BrpError::internal(format!("missing param `{key}`")))
}

fn text(params: &Option<Value>, key: &str) -> BrpResult<String> {
    field(params, key)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| BrpError::internal(format!("param `{key}` must be a string")))
}

/// `{name, description, parameters}`: a remote tool, replacing any of that name.
fn register_tool(
    In(params): In<Option<Value>>,
    mut commands: Commands,
    tools: Query<(Entity, &Tool)>,
) -> BrpResult {
    let name = text(&params, "name")?;
    let description = text(&params, "description").unwrap_or_default();
    let parameters = field(&params, "parameters")
        .cloned()
        .unwrap_or_else(|_| json!({ "type": "object" }));
    for (entity, tool) in &tools {
        if tool.definition.name == name {
            commands.entity(entity).despawn();
        }
    }
    let tool_name = ToolName::new(&name).map_err(BrpError::internal)?;
    let entity = commands
        .spawn(Tool {
            definition: ToolDefinition::new(tool_name, description, parameters),
            remote: true,
        })
        .id();
    Ok(json!({ "entity": entity.to_bits(), "name": name }))
}

/// `{name?}`: streams each new call of remote tools (of `name`, when given).
fn watch_tool_calls(
    In(params): In<Option<Value>>,
    mut commands: Commands,
    calls: Query<(Entity, &ToolCallRequest), (Without<Dispatched>, Without<ToolDone>)>,
    tools: Query<&Tool>,
) -> BrpResult<Option<Value>> {
    let filter = text(&params, "name").ok();
    for (entity, call) in &calls {
        let name = call.0.function.name.as_str();
        let remote = tools
            .iter()
            .any(|tool| tool.remote && tool.definition.name == name);
        if !remote || filter.as_deref().is_some_and(|f| f != name) {
            continue;
        }
        commands.entity(entity).insert(Dispatched);
        return Ok(Some(json!({
            "call_id": call.0.id.to_string(),
            "name": name,
            "arguments": call.0.function.arguments,
        })));
    }
    Ok(None)
}

/// `{call_id, content}`: answer a dispatched call.
fn tool_result(
    In(params): In<Option<Value>>,
    mut commands: Commands,
    calls: Query<(Entity, &ToolCallRequest)>,
) -> BrpResult {
    let call_id = text(&params, "call_id")?;
    let content = match field(&params, "content")? {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    let (entity, _) = calls
        .iter()
        .find(|(_, call)| call.0.id.to_string() == call_id)
        .ok_or_else(|| BrpError::internal(format!("no pending call `{call_id}`")))?;
    commands.entity(entity).insert(ToolDone(content));
    Ok(json!({ "ok": true }))
}

/// `{text}`: prompt the agent as the user would.
fn prompt(
    In(params): In<Option<Value>>,
    mut convo: ResMut<Conversation>,
    mut transcript: ResMut<Transcript>,
    in_flight: Option<Res<InFlight>>,
) -> BrpResult {
    if convo.busy(in_flight.is_some()) {
        return Err(BrpError::internal("busy"));
    }
    convo.prompt(text(&params, "text")?, &mut transcript);
    Ok(json!({ "ok": true }))
}

/// The transcript so far and whether a turn is in progress.
fn transcript(
    In(_): In<Option<Value>>,
    transcript: Res<Transcript>,
    convo: Res<Conversation>,
    in_flight: Option<Res<InFlight>>,
) -> BrpResult {
    let entries: Vec<Value> = transcript
        .entries
        .iter()
        .map(|entry| match entry {
            Entry::User(text) => json!({ "role": "user", "text": text }),
            Entry::Assistant(text) => json!({ "role": "assistant", "text": text }),
            Entry::Tool { name, args, result } => {
                json!({ "role": "tool", "name": name, "args": args, "result": result })
            }
            Entry::Note(text) => json!({ "role": "note", "text": text }),
        })
        .collect();
    Ok(json!({ "busy": convo.busy(in_flight.is_some()), "entries": entries }))
}

/// Rebuild and hot-patch the agent's own native plugins.
fn patch(In(_): In<Option<Value>>, mut hot: ResMut<Hot>) -> BrpResult {
    hot.request();
    Ok(json!({ "status": hot.status() }))
}
