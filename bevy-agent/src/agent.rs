//! The agent loop as one Bevy system: prompt -> model -> tools -> model ...
//! Every step is written to the session file, so a restart resumes the turn.

use std::path::PathBuf;
use std::time::Instant;

use bevy::prelude::*;
use bevy::tasks::{IoTaskPool, Task, futures::check_ready};
use rig_core::ProviderError;
use rig_core::completion::{AssistantContent, CompletionRequest, CompletionResponse, Message};
use rig_core::message::{ToolResultContent, UserContent};

use crate::brp::{RemoteCall, RemoteCalls};
use crate::reload::{self, RELOAD_EXIT};
use crate::session::{Entry, Paths, Session};
use crate::tools::{Handler, Slot, Tools, slot, spawn_native, take};

pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Runtime>()
            .init_resource::<Tools>()
            .add_systems(Startup, resume)
            .add_systems(Update, (drive, save_on_change).chain());
    }
}

/// What is in flight right now; none of it survives a restart.
#[derive(Resource, Default)]
pub struct Runtime {
    model: Option<Task<Result<CompletionResponse, ProviderError>>>,
    /// Running tool calls: index into the turn's calls, and the output slot.
    running: Vec<(usize, Slot<String>)>,
    /// A build: for the `reload` call at this index, or for `/reload` (`None`).
    build: Option<(Option<usize>, Slot<Result<PathBuf, String>>, Instant)>,
    pub user_reload: bool,
    exiting: bool,
    frames: u32,
    next_call: u64,
}

impl Runtime {
    fn idle(&self) -> bool {
        self.model.is_none() && self.running.is_empty() && self.build.is_none() && !self.exiting
    }

    pub fn activity(&self) -> String {
        if let Some((_, _, since)) = &self.build {
            format!("building ({}s)", since.elapsed().as_secs())
        } else if self.model.is_some() {
            "thinking".to_owned()
        } else if !self.running.is_empty() {
            format!("running {} tool(s)", self.running.len())
        } else if self.exiting {
            "restarting".to_owned()
        } else {
            "idle".to_owned()
        }
    }
}

fn preamble(paths: &Paths) -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let source = paths.source.display();
    format!(
        "You are bevy-agent, a minimal coding agent in a terminal UI. Working directory: {}.\n\
         Your own source is the Cargo package at {source} (built on Bevy and Rig). The Rig \
         crates you depend on are at {source}/../crates. Native plugins are ordinary Bevy \
         plugins in {source}/src/plugins/, listed in src/plugins/mod.rs; each adds tools with \
         `app.add_tool(...)`. External processes can add tools over the Bevy Remote Protocol.\n\
         To change your own behavior, edit your source, then call `reload` by itself: it \
         rebuilds you and restarts into the new binary with this conversation intact, and you \
         continue this turn. If the build fails you get the compiler errors and keep running \
         the old binary. Use tools rather than guessing; be concise.",
        cwd.display()
    )
}

/// Answer every call the previous process left open, so the turn can go on.
fn resume(mut session: ResMut<Session>, mut tools: ResMut<Tools>, paths: Res<Paths>) {
    for remote in session.remote_tools.clone() {
        if !tools.0.contains_key(&remote.def.name) {
            tools.insert(remote.def, Handler::Remote { plugin: remote.plugin });
        }
    }
    let note = std::fs::read_to_string(paths.fallback_note()).ok();
    let _ = std::fs::remove_file(paths.fallback_note());
    let restarted = std::env::var("BEVY_AGENT_GENERATION").is_ok_and(|g| g != "0");
    match &note {
        Some(note) => session.error(note.clone()),
        None if restarted => session.info(format!("restarted into the new binary (pid {})", std::process::id())),
        None => {}
    }
    let mut answered = Vec::new();
    for pending in session.turn.iter_mut().flatten().filter(|p| p.result.is_none()) {
        let result = match (pending.call.function.name.as_str(), &note) {
            ("reload", None) => "Reload succeeded: rebuilt and restarted into the new binary. \
                                 Your changes are live and the conversation is intact."
                .to_owned(),
            ("reload", Some(note)) => format!("Reload failed: {note}"),
            (_, note) => format!(
                "error: the agent restarted before this tool finished.{}",
                note.as_ref().map(|n| format!(" {n}")).unwrap_or_default()
            ),
        };
        answered.push(Entry::ToolResult(preview(&result)));
        pending.result = Some(result);
    }
    session.transcript.extend(answered);
    session.save(&paths.session());
}

fn preview(text: &str) -> String {
    const LINES: usize = 6;
    let lines: Vec<&str> = text.lines().collect();
    match lines.len() > LINES {
        true => format!("{}\n… ({} more lines)", lines[..LINES].join("\n"), lines.len() - LINES),
        false => text.to_owned(),
    }
}

fn describe_call(name: &str, arguments: &serde_json::Value) -> String {
    let mut args = arguments.to_string();
    if args.len() > 200 {
        let mut cut = 200;
        while !args.is_char_boundary(cut) {
            cut -= 1;
        }
        args.truncate(cut);
        args.push('…');
    }
    format!("{name} {args}")
}

fn start_model(session: &Session, tools: &Tools, paths: &Paths) -> Result<Task<Result<CompletionResponse, ProviderError>>, String> {
    let model = session.model.completion_model().map_err(|e| e.to_string())?;
    let mut chat_history = vec![Message::system(preamble(paths))];
    chat_history.extend(session.history.iter().cloned());
    let request = CompletionRequest {
        model: None,
        chat_history,
        documents: Vec::new(),
        tools: tools.definitions(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };
    Ok(IoTaskPool::get().spawn(model.call(request)))
}

fn save_on_change(session: Res<Session>, paths: Res<Paths>) {
    if session.is_changed() && !session.is_added() {
        session.save(&paths.session());
    }
}

fn drive(
    mut session: ResMut<Session>,
    mut rt: ResMut<Runtime>,
    tools: Res<Tools>,
    mut remote: ResMut<RemoteCalls>,
    paths: Res<Paths>,
    mut exit: MessageWriter<AppExit>,
) {
    rt.frames += 1;
    if rt.frames == 10 {
        reload::promote(&paths);
    }

    // The model answered.
    if let Some(task) = rt.model.as_mut()
        && let Some(reply) = check_ready(task)
    {
        rt.model = None;
        match reply {
            Ok(response) => {
                for part in &response.choice {
                    match part {
                        AssistantContent::Text(text) if !text.text.trim().is_empty() => {
                            session.transcript.push(Entry::Assistant(text.text.trim().to_owned()));
                        }
                        AssistantContent::ToolCall(call) => session.transcript.push(Entry::ToolCall(
                            describe_call(call.function.name.as_str(), &call.function.arguments),
                        )),
                        _ => {}
                    }
                }
                let calls: Vec<_> = response
                    .tool_calls()
                    .cloned()
                    .map(|call| crate::session::PendingCall { call, result: None })
                    .collect();
                match response.message() {
                    Some(message) => session.history.push(message),
                    None => session.info("(the model returned an empty response)"),
                }
                session.turn = (!calls.is_empty()).then_some(calls);
            }
            Err(error) => {
                session.error(format!("model error: {error}"));
                session.turn = None;
            }
        }
    }

    // Tools finished.
    let mut finished = Vec::new();
    rt.running.retain(|(index, slot)| match take(slot) {
        Some(output) => {
            finished.push((*index, output));
            false
        }
        None => true,
    });
    for (index, output) in finished {
        session.transcript.push(Entry::ToolResult(preview(&output)));
        if let Some(pending) = session.turn.as_mut().and_then(|calls| calls.get_mut(index)) {
            pending.result = Some(output);
        }
    }

    // A build finished.
    if let Some((call, build, _)) = &rt.build
        && let Some(outcome) = take(build)
    {
        let call = *call;
        rt.build = None;
        let failure = match outcome.map(|built| reload::install_candidate(&paths, &built)) {
            Ok(Ok(())) => {
                session.info("build succeeded; restarting into the new binary");
                session.save(&paths.session());
                rt.exiting = true;
                exit.write(AppExit::from_code(RELOAD_EXIT));
                return;
            }
            Ok(Err(error)) => format!("could not stage the new binary: {error}"),
            Err(errors) => errors,
        };
        let result = format!("Reload aborted, the old binary keeps running. {failure}");
        match call.and_then(|index| session.turn.as_mut()?.get_mut(index)) {
            Some(pending) => pending.result = Some(result.clone()),
            None => {
                session.error(result);
                return;
            }
        }
        session.transcript.push(Entry::ToolResult(preview(&result)));
    }

    if !rt.idle() {
        return;
    }

    // Nothing in flight: take the next step. Read before writing, so an idle
    // frame does not mark the session changed.
    if session.turn.is_none() {
        if rt.user_reload {
            rt.user_reload = false;
            session.info("building…");
            rt.build = Some((None, reload::spawn_build(paths.source.clone()), Instant::now()));
        } else if let Some(prompt) = session.queue.front().cloned() {
            session.queue.pop_front();
            session.transcript.push(Entry::User(prompt.clone()));
            session.push_user(vec![UserContent::text(prompt)]);
            session.turn = Some(Vec::new());
        }
        return;
    }
    let Some(mut calls) = session.turn.take() else {
        return;
    };

    if calls.iter().all(|pending| pending.result.is_some()) {
        if !calls.is_empty() {
            let results = calls
                .iter()
                .map(|p| {
                    let output = p.result.clone().unwrap_or_default();
                    UserContent::ToolResult(p.call.result(vec![ToolResultContent::text(output)]))
                })
                .collect();
            session.push_user(results);
        }
        match start_model(&session, &tools, &paths) {
            Ok(task) => {
                rt.model = Some(task);
                session.turn = Some(Vec::new());
            }
            Err(error) => session.error(format!("cannot call the model: {error}")),
        }
        return;
    }

    // Run every open call except `reload`, which goes last, alone.
    let mut reload_call = None;
    for (index, pending) in calls.iter_mut().enumerate().filter(|(_, p)| p.result.is_none()) {
        let name = pending.call.function.name.as_str();
        let arguments = pending.call.function.arguments.clone();
        match tools.0.get(name).map(|tool| &tool.handler) {
            Some(Handler::Native(run)) => rt.running.push((index, spawn_native(run.clone(), arguments))),
            Some(Handler::Remote { plugin }) => {
                rt.next_call += 1;
                let output = slot();
                remote.0.push(RemoteCall {
                    id: format!("{}-{}", std::process::id(), rt.next_call),
                    plugin: plugin.clone(),
                    tool: name.to_owned(),
                    arguments,
                    sent: false,
                    since: Instant::now(),
                    slot: output.clone(),
                });
                rt.running.push((index, output));
            }
            Some(Handler::Reload) => {
                reload_call.get_or_insert(index);
            }
            None => pending.result = Some(format!("error: no tool named `{name}`")),
        }
    }
    if rt.running.is_empty()
        && let Some(index) = reload_call
    {
        rt.build = Some((Some(index), reload::spawn_build(paths.source.clone()), Instant::now()));
    }
    session.turn = Some(calls);
}
