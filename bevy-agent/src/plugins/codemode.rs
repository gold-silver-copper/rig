//! Code mode, after pi's: a `codemode` tool whose argument is JavaScript
//! that calls the other tools, run in rig-codemode's sandbox where calling
//! tools is the only capability. Each call the script makes is a nested
//! tool-call entity of the same agent, run by the same systems as the
//! model's own calls; only the script's output reaches the model.

use async_channel::{Receiver, Sender};
use bevy::prelude::*;
use rig_codemode::{CodeMode, Execution, TOOL_NAME};
use rig_core::completion::ToolDefinition;
use rig_core::message::{self, CallId, LocalCallId, ToolFunction, ToolName};
use serde_json::Value;

use crate::glue::{CallOf, CallSpan, Nested, ToolCall, ToolOutput, Tools};

pub struct CodeModePlugin;

type Request = (String, Value, Sender<Result<Value, String>>);

/// Nested calls scripts ask for, from every running script.
#[derive(Resource)]
struct Requests {
    sender: Sender<(Entity, Request)>,
    receiver: Receiver<(Entity, Request)>,
}

/// A running script, on its `codemode` call entity.
#[derive(Component)]
struct Script(Receiver<Execution>);

/// Where a nested call's result goes, on the nested call entity.
#[derive(Component)]
struct Reply(Sender<Result<Value, String>>);

impl Plugin for CodeModePlugin {
    fn build(&self, app: &mut App) {
        let (sender, receiver) = async_channel::unbounded();
        let definition = sandbox(app.world().resource::<Tools>()).definition();
        app.world_mut().resource_mut::<Tools>().0.insert(
            TOOL_NAME.into(),
            crate::glue::Tool {
                definition,
                run: None,
            },
        );
        app.insert_resource(Requests { sender, receiver })
            .add_systems(
                Update,
                (
                    refresh_definition,
                    start_scripts,
                    nested_calls,
                    nested_results,
                    finish_scripts,
                ),
            );
    }
}

/// A sandbox over every tool but `codemode` itself.
fn sandbox(tools: &Tools) -> CodeMode {
    CodeMode::new(
        tools
            .definitions()
            .into_iter()
            .filter(|definition| definition.name != TOOL_NAME),
    )
}

/// Keeps the tool's declarations in step with the registry, which grows
/// when, say, an MCP server connects.
fn refresh_definition(mut tools: ResMut<Tools>) {
    if !tools.is_changed() {
        return;
    }
    let definition: ToolDefinition = sandbox(&tools).definition();
    let tools = tools.bypass_change_detection();
    if let Some(tool) = tools.0.get_mut(TOOL_NAME)
        && tool.definition != definition
    {
        tool.definition = definition;
    }
}

fn start_scripts(
    mut commands: Commands,
    calls: Query<(Entity, &ToolCall), Added<ToolCall>>,
    tools: Res<Tools>,
    requests: Res<Requests>,
) {
    for (entity, ToolCall(call)) in &calls {
        if call.function.name != TOOL_NAME {
            continue;
        }
        let Some(code) = call.function.arguments.get("code").and_then(Value::as_str) else {
            commands
                .entity(entity)
                .insert(ToolOutput("error: `code` must be a string".into()));
            continue;
        };
        let code = code.to_owned();
        let sandbox = sandbox(&tools);
        let requests = requests.sender.clone();
        let (done, finished) = async_channel::bounded(1);
        std::thread::spawn(move || {
            let execution = sandbox.execute(&code, move |name, arguments| {
                let (reply, answer) = async_channel::bounded(1);
                requests
                    .send_blocking((entity, (name.to_owned(), arguments, reply)))
                    .map_err(|_| "the agent stopped".to_owned())?;
                answer
                    .recv_blocking()
                    .unwrap_or_else(|_| Err("the call was cancelled".into()))
            });
            let _ = done.send_blocking(execution);
        });
        commands.entity(entity).insert(Script(finished));
    }
}

/// Spawns each call a script asks for as a nested call of the same agent.
fn nested_calls(
    mut commands: Commands,
    requests: Res<Requests>,
    parents: Query<(&CallOf, &CallSpan)>,
    tools: Res<Tools>,
) {
    while let Ok((parent, (name, arguments, reply))) = requests.receiver.try_recv() {
        let Ok((CallOf(agent), CallSpan(span))) = parents.get(parent) else {
            let _ = reply.try_send(Err("the script's call is gone".into()));
            continue;
        };
        let known = tools.0.contains_key(&name) && name != TOOL_NAME;
        let Ok(tool_name) = ToolName::new(name.clone())
            .map_err(|_| ())
            .and_then(|tool| if known { Ok(tool) } else { Err(()) })
        else {
            let _ = reply.try_send(Err(format!("there is no tool named `{name}`")));
            continue;
        };
        let call = message::ToolCall::new(
            CallId::from(LocalCallId::new()),
            ToolFunction::new(tool_name, arguments),
        );
        let span = tracing::info_span!(
            parent: span,
            "execute_tool",
            gen_ai.operation.name = "execute_tool",
            gen_ai.tool.name = name.as_str(),
        );
        commands.spawn((
            ToolCall(call),
            CallOf(*agent),
            Nested,
            CallSpan(span),
            Reply(reply),
        ));
    }
}

/// Hands finished nested calls back to their scripts.
fn nested_results(
    mut commands: Commands,
    calls: Query<(Entity, &ToolOutput, &Reply), With<Nested>>,
) {
    for (entity, ToolOutput(output), Reply(reply)) in &calls {
        let result = match output.strip_prefix("error: ") {
            Some(error) => Err(error.to_owned()),
            None => Ok(Value::String(output.clone())),
        };
        let _ = reply.try_send(result);
        commands.entity(entity).despawn();
    }
}

fn finish_scripts(mut commands: Commands, scripts: Query<(Entity, &Script)>) {
    for (entity, Script(finished)) in &scripts {
        if let Ok(execution) = finished.try_recv() {
            commands
                .entity(entity)
                .remove::<Script>()
                .insert(ToolOutput(execution.render()));
        }
    }
}
