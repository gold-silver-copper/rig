//! Evals: scripted sessions checked against their cassettes, offline.
//!
//! A script (`evals/<name>.json`) is a list of steps: switch model, or send a
//! prompt and expect these tool calls and this final answer. `--record NAME`
//! writes the script of a live session next to its cassette, so a bug seen
//! live replays offline; `--replay NAME` runs it in the TUI; `rig-pi eval`
//! runs every script headless, each as its own agent entity in one world,
//! and exits non-zero if any step differs.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use rig_cassette::http::CassetteMode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::glue::{AgentEvent, AgentSet, EventKind, Inbox, Model, Turn};
use crate::plugins::cassette::Cassette;
use crate::session::{EntryKind, Focused, Kind, Transcript, new_session_id, spawn_agent};
use crate::tui::{SlashCommand, SlashCommands};
use crate::parse_model;
use crate::{Env, Options};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Script {
    /// The cassette the script replays against; defaults to the file name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cassette: Option<String>,
    pub steps: Vec<Step>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum Step {
    Model {
        model: String,
    },
    Prompt {
        prompt: String,
        /// The tool calls the turn must make, in order.
        #[serde(default)]
        calls: Vec<ExpectedCall>,
        /// The final answer, exactly.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        answer: Option<String>,
        /// Or just a part of it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        answer_contains: Option<String>,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ExpectedCall {
    pub name: String,
    pub arguments: Value,
}

pub fn evals_dir(env: &Env) -> PathBuf {
    env.source.join("evals")
}

pub fn load(dir: &Path, name: &str) -> Result<Script, String> {
    let path = dir.join(format!("{name}.json"));
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

pub struct EvalsPlugin {
    pub record: Option<String>,
    pub replay: Option<String>,
}

impl Plugin for EvalsPlugin {
    fn build(&self, app: &mut App) {
        app.world_mut().get_resource_or_init::<SlashCommands>().0.insert(
            "eval".into(),
            "replay evals offline next to this session: /eval [NAME...]".into(),
        );
        app.add_systems(
            Update,
            (
                observe.after(AgentSet::Run),
                drive.in_set(AgentSet::Input),
                start_evals.in_set(AgentSet::Input),
                report_evals.after(AgentSet::Run),
            ),
        )
        .add_systems(Last, write_recordings);
        if let Some(name) = self.record.clone() {
            app.add_systems(PostStartup, move |mut commands: Commands, agents: Query<(Entity, &Model), With<Focused>>| {
                for (agent, model) in &agents {
                    commands.entity(agent).insert(Recorder {
                        name: name.clone(),
                        script: Script {
                            cassette: None,
                            steps: vec![Step::Model { model: model.0.to_string() }],
                        },
                        model: model.0.to_string(),
                        turn: Observed::default(),
                    });
                }
            });
        }
        if let Some(name) = self.replay.clone() {
            app.add_systems(
                PostStartup,
                move |mut commands: Commands, env: Res<Env>, mut agents: Query<(Entity, &mut Transcript), With<Focused>>| {
                    for (agent, mut transcript) in &mut agents {
                        match load(&evals_dir(&env), &name) {
                            Ok(script) => {
                                transcript.log(EntryKind::Info, format!("replaying {name} offline"));
                                commands.entity(agent).insert(Run::new(name.clone(), script));
                            }
                            Err(error) => transcript.log(EntryKind::Error, error),
                        }
                    }
                },
            );
        }
    }
}

/// What a turn did.
#[derive(Clone, Debug, Default)]
struct Observed {
    calls: Vec<ExpectedCall>,
    answer: Option<String>,
    ended: bool,
}

/// Records a live session as a script, written when the process exits.
#[derive(Component)]
struct Recorder {
    name: String,
    script: Script,
    model: String,
    turn: Observed,
}

/// A script being run on this agent.
#[derive(Component)]
pub struct Run {
    pub name: String,
    script: Script,
    next: usize,
    /// The prompt step awaiting its turn's end, and what the turn did.
    waiting: Option<(usize, Observed)>,
    pub failures: Vec<String>,
    pub finished: bool,
}

impl Run {
    pub fn new(name: String, script: Script) -> Self {
        Self {
            name,
            script,
            next: 0,
            waiting: None,
            failures: Vec::new(),
            finished: false,
        }
    }
}

fn observe(
    mut events: MessageReader<AgentEvent>,
    mut runs: Query<&mut Run>,
    mut recorders: Query<(&mut Recorder, &Model)>,
) {
    for event in events.read() {
        let apply = |turn: &mut Observed| match &event.kind {
            EventKind::CallStarted { name, arguments } => turn.calls.push(ExpectedCall {
                name: name.clone(),
                arguments: arguments.clone(),
            }),
            EventKind::Text(text) => turn.answer = Some(text.clone()),
            EventKind::TurnEnded => turn.ended = true,
            _ => {}
        };
        if let Ok(mut run) = runs.get_mut(event.agent)
            && let Some((_, turn)) = &mut run.waiting
        {
            apply(turn);
        }
        if let Ok((mut recorder, model)) = recorders.get_mut(event.agent) {
            let recorder = &mut *recorder;
            if let EventKind::Input(text) = &event.kind {
                let current = model.0.to_string();
                if current != recorder.model {
                    recorder.script.steps.push(Step::Model { model: current.clone() });
                    recorder.model = current;
                }
                recorder.script.steps.push(Step::Prompt {
                    prompt: text.clone(),
                    calls: Vec::new(),
                    answer: None,
                    answer_contains: None,
                });
                recorder.turn = Observed::default();
            }
            apply(&mut recorder.turn);
            if recorder.turn.ended
                && let Some(Step::Prompt { calls, answer, .. }) = recorder.script.steps.last_mut()
            {
                *calls = std::mem::take(&mut recorder.turn.calls);
                *answer = recorder.turn.answer.take();
                recorder.turn.ended = false;
            }
        }
    }
}

/// Feed each run its next step once its agent is idle, and check finished
/// prompt steps.
fn drive(mut runs: Query<(&mut Run, &mut Model, &mut Inbox, &Turn, &mut Transcript)>) {
    for (mut run, mut model, mut inbox, turn, mut transcript) in &mut runs {
        let run = &mut *run;
        if run.finished {
            continue;
        }
        if let Some((index, observed)) = &run.waiting {
            if !observed.ended || !turn.is_idle() || !inbox.follow_ups.is_empty() {
                continue;
            }
            let failures = check(&run.script.steps[*index], observed);
            let step = *index + 1;
            if failures.is_empty() {
                transcript.log(EntryKind::Info, format!("{} step {step}: tool calls and answer match", run.name));
            }
            for failure in failures {
                transcript.log(EntryKind::Error, format!("{} step {step}: {failure}", run.name));
                run.failures.push(format!("step {step}: {failure}"));
            }
            run.waiting = None;
        }
        while let Some(step) = run.script.steps.get(run.next) {
            let index = run.next;
            run.next += 1;
            match step {
                Step::Model { model: reference } => match parse_model(reference) {
                    Ok(next) => model.0 = next,
                    Err(error) => run.failures.push(format!("step {}: {error}", index + 1)),
                },
                Step::Prompt { prompt, .. } => {
                    inbox.follow_ups.push_back(prompt.clone());
                    run.waiting = Some((index, Observed::default()));
                    break;
                }
            }
        }
        if run.waiting.is_none() && run.next >= run.script.steps.len() {
            run.finished = true;
            let verdict = if run.failures.is_empty() { "passed" } else { "failed" };
            transcript.log(EntryKind::Info, format!("{} {verdict}", run.name));
        }
    }
}

fn check(step: &Step, observed: &Observed) -> Vec<String> {
    let Step::Prompt {
        calls,
        answer,
        answer_contains,
        ..
    } = step
    else {
        return Vec::new();
    };
    let mut failures = Vec::new();
    if &observed.calls != calls {
        failures.push(format!(
            "tool calls differ: expected {}, got {}",
            serde_json::to_string(calls).unwrap_or_default(),
            serde_json::to_string(&observed.calls).unwrap_or_default()
        ));
    }
    let got = observed.answer.clone().unwrap_or_default();
    if let Some(answer) = answer
        && &got != answer
    {
        failures.push(format!("answer differs: expected {answer:?}, got {got:?}"));
    }
    if let Some(part) = answer_contains
        && !got.contains(part.as_str())
    {
        failures.push(format!("answer lacks {part:?}: got {got:?}"));
    }
    failures
}

/// Every eval's name, sorted.
pub fn all_names(env: &Env) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(evals_dir(env))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            (path.extension()? == "json").then(|| path.file_stem()?.to_str().map(str::to_owned))?
        })
        .collect();
    names.sort();
    names
}

/// An eval run started from the TUI, reporting to `to`.
#[derive(Component)]
struct ReportTo(Entity);

/// `/eval [NAME...]`: spawn eval agents in this world, replaying offline.
fn start_evals(
    mut commands_in: MessageReader<SlashCommand>,
    env: Res<Env>,
    prompt: Res<crate::prompt::Prompt>,
    tools: Res<crate::glue::Tools>,
    mut transcripts: Query<&mut Transcript>,
    mut commands: Commands,
) {
    for command in commands_in.read().filter(|c| c.name == "eval") {
        let names: Vec<String> = if command.args.is_empty() {
            all_names(&env)
        } else {
            command.args.split_whitespace().map(str::to_owned).collect()
        };
        for name in names {
            match load(&evals_dir(&env), &name) {
                Err(error) => {
                    if let Ok(mut transcript) = transcripts.get_mut(command.agent) {
                        transcript.log(EntryKind::Error, error);
                    }
                }
                Ok(script) => {
                    let cassette = script.cassette.clone().unwrap_or_else(|| name.clone());
                    let model = crate::session::default_model();
                    let agent = spawn_agent(
                        &mut commands,
                        prompt.build(&tools),
                        Kind::Eval,
                        &name,
                        model,
                        new_session_id(Kind::Eval),
                    );
                    commands.entity(agent).insert((
                        Cassette {
                            name: cassette,
                            mode: CassetteMode::Replay,
                        },
                        Run::new(name, script),
                        ReportTo(command.agent),
                    ));
                }
            }
        }
    }
}

/// Report finished TUI-started evals and remove their agents.
fn report_evals(
    runs: Query<(Entity, &Run, &ReportTo)>,
    mut transcripts: Query<&mut Transcript>,
    mut commands: Commands,
) {
    for (entity, run, to) in &runs {
        if !run.finished {
            continue;
        }
        if let Ok(mut transcript) = transcripts.get_mut(to.0) {
            if run.failures.is_empty() {
                transcript.log(EntryKind::Info, format!("eval {} passed (offline)", run.name));
            } else {
                transcript.log(
                    EntryKind::Error,
                    format!("eval {} failed:\n{}", run.name, run.failures.join("\n")),
                );
            }
        }
        commands.entity(entity).despawn();
    }
}

fn write_recordings(mut exits: MessageReader<AppExit>, recorders: Query<&Recorder>, env: Res<Env>) {
    if exits.read().next().is_none() {
        return;
    }
    for recorder in &recorders {
        let dir = evals_dir(&env);
        let path = dir.join(format!("{}.json", recorder.name));
        let written = std::fs::create_dir_all(&dir).and_then(|()| {
            let mut json = serde_json::to_vec_pretty(&recorder.script)?;
            json.push(b'\n');
            std::fs::write(&path, json)
        });
        if let Err(error) = written {
            error!("writing {}: {error}", path.display());
        }
    }
}

/// `rig-pi eval [NAME...]`: run scripts headless against their cassettes.
/// Returns the number of failed scripts (0 when all pass).
pub fn run(args: &[String]) -> i32 {
    let env = Env::from_process();
    let names: Vec<String> = if args.is_empty() {
        all_names(&env)
    } else {
        args.to_vec()
    };
    let state = std::env::temp_dir().join(format!("rig-pi-eval-{}", std::process::id()));
    let results = run_scripts(&names, &state);
    let _ = std::fs::remove_dir_all(&state);
    let mut failed = 0;
    for (name, failures) in &results {
        if failures.is_empty() {
            println!("pass  {name}");
        } else {
            failed += 1;
            println!("FAIL  {name}");
            for failure in failures {
                println!("      {failure}");
            }
        }
    }
    println!("{} passed, {failed} failed", results.len() - failed);
    failed as i32
}

/// Run each named script as an eval agent in one headless world, replaying
/// its cassette, and return each script's failures.
pub fn run_scripts(names: &[String], state: &Path) -> Vec<(String, Vec<String>)> {
    let env = Env {
        state: state.to_path_buf(),
        port: 0,
        note: None,
        ready_file: None,
        restarted: false,
        ..Env::from_process()
    };
    let options = Options {
        disable: vec!["mcp".into(), "telemetry".into(), "durable".into()],
        ..Options::default()
    };
    let mut app = crate::app(env.clone(), &options, false);
    let mut results = Vec::new();
    let mut runs = Vec::new();
    app.finish();
    app.cleanup();
    for name in names {
        match load(&evals_dir(&env), name) {
            Err(error) => results.push((name.clone(), vec![error])),
            Ok(script) => {
                let cassette = script.cassette.clone().unwrap_or_else(|| name.clone());
                let world = app.world_mut();
                let preamble = crate::prompt::current(world);
                let mut commands = world.commands();
                let model = crate::session::default_model();
                let agent = spawn_agent(&mut commands, preamble, Kind::Eval, name, model, new_session_id(Kind::Eval));
                commands.entity(agent).insert((
                    Cassette {
                        name: cassette,
                        mode: CassetteMode::Replay,
                    },
                    Run::new(name.clone(), script),
                ));
                world.flush();
                runs.push(agent);
            }
        }
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    loop {
        app.update();
        let world = app.world_mut();
        let done = runs.iter().all(|agent| world.get::<Run>(*agent).is_none_or(|run| run.finished));
        if done || std::time::Instant::now() > deadline {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    for agent in runs {
        if let Some(run) = app.world().get::<Run>(agent) {
            let mut failures = run.failures.clone();
            if !run.finished {
                failures.push("did not finish within 300s".into());
            }
            results.push((run.name.clone(), failures));
        }
    }
    app.world_mut().write_message(AppExit::Success);
    app.update();
    results
}
