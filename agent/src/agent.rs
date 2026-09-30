//! The turn loop: send the conversation to the model, run the tool calls it
//! makes one at a time, send the results back, until it answers without
//! calling a tool.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::tasks::futures::check_ready;
use bevy::tasks::{AsyncComputeTaskPool, IoTaskPool, Task};
use rig_core::ProviderError;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message};
use rig_core::message::{AssistantContent, ToolCall, ToolResultContent, UserContent};

use crate::Env;
use crate::brp::Remote;
use crate::reload;
use crate::session::{Kind, Phase, Session};
use crate::tools::{Run, Tools, clip};

/// Work in flight for the current phase. Not persisted: a restart starts
/// the phase's work again.
#[derive(Resource, Default)]
pub struct InFlight {
    model: Option<Task<Result<CompletionResponse, ProviderError>>>,
    tool: Option<Running>,
}

impl InFlight {
    pub fn tool_running(&self) -> bool {
        self.tool.is_some()
    }
}

enum Running {
    Local(Task<Result<String, String>>),
    Remote { id: String, since: Instant },
    Reload(Task<Result<std::path::PathBuf, String>>),
}

pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InFlight>()
            .add_systems(Startup, resume)
            .add_systems(Update, drive);
    }
}

fn preamble(env: &Env) -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    format!(
        "You are rig-pi, a minimal coding agent in a terminal. Be concise.\n\
         Working directory: {cwd}\n\
         Your own source code is the Rust crate at {src} (Bevy app, Rig for models, ratatui TUI). \
         Native plugins are ordinary Bevy plugins in {src}/src/plugins. The rig crates you use are in {rig}.\n\
         You may edit any of it, including Cargo manifests. Call the `reload` tool to rebuild and restart \
         into the new version: the conversation continues after the restart, so keep working in the same turn. \
         If the build fails you get the compiler errors and keep running the old version.",
        cwd = cwd.display(),
        src = env.source.display(),
        rig = env.source.join("../crates").display(),
    )
}

/// Append user text, merging into a trailing user message so the
/// conversation never has two user turns in a row.
pub fn push_user(session: &mut Session, text: String) {
    session.log(Kind::User, text.clone());
    if let Some(Message::User { content }) = session.messages.last_mut() {
        content.push(UserContent::text(text));
    } else {
        session.messages.push(Message::user(text));
    }
}

/// Stop the current turn. Unanswered tool calls get a "cancelled" result so
/// the conversation stays valid for the next request.
pub fn cancel(session: &mut Session, inflight: &mut InFlight) {
    inflight.model = None;
    inflight.tool = None;
    if let Phase::Tools { calls, results, .. } = std::mem::take(&mut session.phase) {
        let mut results = results;
        for call in &calls[results.len()..] {
            results.push(call.result(vec![ToolResultContent::text("cancelled by the user")]));
        }
        session.messages.push(Message::tool_results(results));
    }
    session.log(Kind::Info, "cancelled");
}

/// On startup, a turn interrupted by a reload continues: the `reload` call
/// gets the supervisor's account of the restart as its result.
fn resume(mut session: ResMut<Session>, env: Res<Env>) {
    let note = env.note.clone();
    let reloading = match &session.phase {
        Phase::Tools { calls, next, .. } => calls
            .get(*next)
            .is_some_and(|call| call.function.name == reload::TOOL),
        _ => false,
    };
    if reloading {
        let text = note.unwrap_or_else(|| "The agent was restarted.".into());
        finish_tool(&mut session, Ok(text));
    } else if let Some(note) = note {
        session.log(Kind::Info, note);
    }
}

pub fn drive(
    mut session: ResMut<Session>,
    mut inflight: ResMut<InFlight>,
    tools: Res<Tools>,
    env: Res<Env>,
    mut remote: ResMut<Remote>,
    mut exit: MessageWriter<AppExit>,
) {
    if !session.busy() {
        let Some(prompt) = session.queue.front().cloned() else {
            return;
        };
        session.queue.pop_front();
        push_user(&mut session, prompt);
        session.phase = Phase::Model;
    }
    match &session.phase {
        Phase::Idle => {}
        Phase::Model => drive_model(&mut session, &mut inflight, &tools, &env),
        Phase::Tools { calls, next, .. } => {
            let Some(call) = calls.get(*next).cloned() else {
                let Phase::Tools { results, .. } = std::mem::take(&mut session.phase) else {
                    return;
                };
                session.messages.push(Message::tool_results(results));
                session.phase = Phase::Model;
                return;
            };
            drive_tool(
                &mut session,
                &mut inflight,
                &tools,
                &env,
                &mut remote,
                &mut exit,
                call,
            );
        }
    }
}

fn drive_model(session: &mut ResMut<Session>, inflight: &mut InFlight, tools: &Tools, env: &Env) {
    let Some(task) = &mut inflight.model else {
        let request = CompletionRequest::from(session.messages.clone())
            .preamble(preamble(env))
            .tools(tools.definitions());
        match session.model.completion_model() {
            Ok(model) => inflight.model = Some(IoTaskPool::get().spawn(model.call(request))),
            Err(error) => {
                let text = format!("{}: {error}", session.model);
                session.log(Kind::Error, text);
                session.phase = Phase::Idle;
            }
        }
        return;
    };
    let Some(result) = check_ready(task) else {
        return;
    };
    inflight.model = None;
    let response = match result {
        Ok(response) => response,
        Err(error) => {
            let text = format!("{}: {error}", session.model);
            session.log(Kind::Error, text);
            session.phase = Phase::Idle;
            return;
        }
    };
    for part in &response.choice {
        match part {
            AssistantContent::Text(text) if !text.text.trim().is_empty() => {
                session.log(Kind::Assistant, text.text.trim())
            }
            AssistantContent::ToolCall(call) => session.log(
                Kind::Call,
                format!("{} {}", call.function.name, call.function.arguments),
            ),
            _ => {}
        }
    }
    let calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
    session.messages.extend(response.message());
    session.phase = if calls.is_empty() {
        Phase::Idle
    } else {
        Phase::Tools {
            calls,
            next: 0,
            results: Vec::new(),
        }
    };
}

fn drive_tool(
    session: &mut ResMut<Session>,
    inflight: &mut InFlight,
    tools: &Tools,
    env: &Env,
    remote: &mut Remote,
    exit: &mut MessageWriter<AppExit>,
    call: ToolCall,
) {
    let Some(running) = &mut inflight.tool else {
        let args = call.function.arguments.clone();
        inflight.tool = match tools
            .0
            .get(call.function.name.as_str())
            .map(|t| t.run.clone())
        {
            None => {
                finish_tool(
                    session,
                    Err(format!("unknown tool `{}`", call.function.name)),
                );
                None
            }
            Some(Run::Local(run)) => Some(Running::Local(
                AsyncComputeTaskPool::get().spawn(async move { run(args) }),
            )),
            Some(Run::Remote(plugin)) => {
                let id = call.id.to_string();
                remote.enqueue(&plugin, &id, call.function.name.as_str(), args);
                Some(Running::Remote {
                    id,
                    since: Instant::now(),
                })
            }
            Some(Run::Reload) => {
                session.log(Kind::Info, "building…");
                Some(Running::Reload(reload::spawn_build(env)))
            }
        };
        return;
    };
    let outcome = match running {
        Running::Local(task) => check_ready(task),
        Running::Remote { id, since } => remote.take_result(id).or_else(|| {
            (since.elapsed() > Duration::from_secs(300))
                .then(|| Err("the plugin did not answer within 300s".into()))
        }),
        Running::Reload(task) => match check_ready(task) {
            Some(Ok(binary)) => match reload::restart(session, env, &binary, exit) {
                // Exiting: leave the call open for the new process to answer.
                Ok(()) => return,
                Err(error) => Some(Err(error)),
            },
            Some(Err(errors)) => Some(Err(errors)),
            None => None,
        },
    };
    if let Some(outcome) = outcome {
        inflight.tool = None;
        finish_tool(session, outcome);
    }
}

/// Record the result of the current tool call and move to the next one.
fn finish_tool(session: &mut Session, outcome: Result<String, String>) {
    let text = match outcome {
        Ok(text) if text.trim().is_empty() => "(no output)".to_owned(),
        Ok(text) => text,
        Err(error) => format!("Error: {error}"),
    };
    session.log(Kind::Result, clip(&text, 2_000, false));
    if let Phase::Tools {
        calls,
        next,
        results,
    } = &mut session.phase
        && let Some(call) = calls.get(*next)
    {
        results.push(call.result(vec![ToolResultContent::text(text)]));
        *next += 1;
    }
}
