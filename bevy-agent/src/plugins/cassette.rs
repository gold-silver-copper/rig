//! Recording and replay of provider traffic with rig-cassette's HTTP engine.
//! `--record <name>` sends every model request of the session through a
//! recording proxy and writes one scrubbed cassette per provider under
//! `<root>/<name>/` when the agent exits. `--replay <name>` answers the same
//! requests from those cassettes with no network and no API keys, and
//! reports requests that do not match or interactions left unplayed. The
//! system prompt names paths relative to the working directory in both
//! modes, so a recording replays on any machine.
//!
//! A recording also saves what was typed into the TUI session and what it
//! showed, as `session.json`. A replay types the same lines again, one turn
//! at a time, and compares the tool calls and the final answer with the
//! recording; `--headless` exits when it is done, with status 0 only for a
//! match.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use rig_cassette::http::{CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
use rig_core::providers::registry::ProviderRef;

use serde::{Deserialize, Serialize};

use crate::State;
use crate::glue::{ModelFactory, RigRuntime, Turn, report};
use crate::session::{self, Entry, Inputs, Origin, Prompts, Transcript};

pub struct CassettePlugin {
    pub mode: CassetteMode,
    /// The cassette directory for this session: one file per provider.
    pub dir: PathBuf,
    /// Record or replay what the TUI session typed; evals drive their own.
    pub session: bool,
}

/// What a recorded session typed and showed.
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct Recorded {
    pub inputs: Vec<String>,
    pub transcript: Vec<Entry>,
}

/// The lines a replay still has to type.
#[derive(Resource)]
struct Script {
    recorded: Recorded,
    next: usize,
}

/// The open cassettes, by provider.
#[derive(Resource, Clone, Default)]
pub struct Cassettes(Arc<Mutex<HashMap<String, ProviderCassette>>>);

impl Plugin for CassettePlugin {
    fn build(&self, app: &mut App) {
        let handle = app.world().resource::<RigRuntime>().0.handle().clone();
        let cassettes = Cassettes::default();
        let open = cassettes.0.clone();
        let (mode, dir) = (self.mode, self.dir.clone());
        app.insert_resource(ModelFactory(Box::new(move |reference: &ProviderRef| {
            let vendor = reference
                .id()
                .map(|id| id.vendor())
                .ok_or("only registered providers can be recorded")?;
            let provider = cassette_provider(vendor);
            let key = match mode {
                CassetteMode::Record => {
                    let variable = reference
                        .id()
                        .map(|id| id.api_key_env())
                        .unwrap_or_default();
                    std::env::var(variable)
                        .map_err(|_| format!("{variable} must be set to record"))?
                }
                CassetteMode::Replay => "[REDACTED]".to_owned(),
            };
            let config = reference.config(key);
            let real_base_url = serde_json::to_value(&config)
                .ok()
                .and_then(|json| {
                    json.as_object()?
                        .values()
                        .next()?
                        .get("base_url")?
                        .as_str()
                        .map(str::to_owned)
                })
                .ok_or("the provider has no base URL")?;
            let mut open = open.lock().map_err(|error| error.to_string())?;
            if !open.contains_key(provider) {
                let path = dir.join(format!("{provider}.yaml"));
                if mode == CassetteMode::Replay && !path.exists() {
                    return Err(format!("no recording for {provider} at {}", path.display()));
                }
                let cassette = handle
                    .block_on(ProviderCassette::try_start_at(
                        RecordVia::Proxy,
                        provider,
                        CassetteSpec::new("bevy-agent"),
                        &real_base_url,
                        mode,
                        path,
                    ))
                    .map_err(|error| report(&error))?;
                open.insert(provider.to_owned(), cassette);
            }
            let base_url = open
                .get(provider)
                .map(ProviderCassette::base_url)
                .unwrap_or_default();
            Ok(config
                .with_base_url(base_url)
                .completion_model(reference.model(), rig_reqwest::shared()))
        })))
        .insert_resource(cassettes)
        .add_systems(Last, finish_on_exit);
        if !self.session {
            return;
        }
        let path = self.dir.join("session.json");
        match self.mode {
            CassetteMode::Record => {
                app.insert_resource(SessionFile(path));
            }
            CassetteMode::Replay => {
                let recorded = std::fs::read(&path)
                    .ok()
                    .and_then(|bytes| serde_json::from_slice(&bytes).ok())
                    .unwrap_or_default();
                let recorded: Recorded = recorded;
                // The session starts on the model it was recorded with.
                if let Some(model) = recorded
                    .inputs
                    .first()
                    .and_then(|line| line.strip_prefix("/model "))
                {
                    app.world_mut().resource_mut::<State>().model = model.to_owned();
                }
                app.insert_resource(Script { recorded, next: 0 })
                    .add_systems(Update, replay_inputs.before(crate::glue::GlueSystems));
            }
        }
    }
}

/// Where a recording saves the TUI session.
#[derive(Resource)]
struct SessionFile(PathBuf);

fn tui_agent(world: &mut World) -> Option<Entity> {
    world
        .query::<(Entity, &Origin)>()
        .iter(world)
        .find(|(_, origin)| **origin == Origin::Tui)
        .map(|(entity, _)| entity)
}

/// Types the recorded lines into the TUI session, each once the previous
/// turn is over, then compares the result with the recording.
fn replay_inputs(world: &mut World) {
    let Some(agent) = tui_agent(world) else {
        return;
    };
    let idle = world.get::<Turn>(agent) == Some(&Turn::Idle)
        && world
            .get::<Prompts>(agent)
            .is_some_and(|prompts| prompts.queue.is_empty())
        && !world.resource::<crate::reload::Reload>().building();
    if !idle {
        return;
    }
    let (line, done) = {
        let script = world.resource::<Script>();
        (
            script.recorded.inputs.get(script.next).cloned(),
            script.next > script.recorded.inputs.len(),
        )
    };
    if done {
        return;
    }
    world.resource_mut::<Script>().next += 1;
    if let Some(line) = line {
        session::submit(world, agent, &line);
        return;
    }
    let recorded = world.resource::<Script>().recorded.transcript.clone();
    let replayed = world
        .get::<Transcript>(agent)
        .map(|transcript| transcript.0.clone())
        .unwrap_or_default();
    let (verdict, matched) = match compare(&recorded, &replayed) {
        None => (
            "Replay matched the recording: the same tool calls and final answer.".to_owned(),
            true,
        ),
        Some(difference) => (
            format!("Replay differs from the recording: {difference}"),
            false,
        ),
    };
    session::log(world, agent, Entry::Notice(verdict.clone()));
    if !world.resource::<State>().tui {
        println!("{verdict}");
        let failures = finish(world);
        for failure in &failures {
            println!("cassette {failure}");
        }
        world.write_message(if matched && failures.is_empty() {
            AppExit::Success
        } else {
            AppExit::error()
        });
    }
}

/// The tool calls and the last answer of a transcript.
pub fn outline(transcript: &[Entry]) -> (Vec<String>, Option<String>) {
    let calls = transcript
        .iter()
        .filter_map(|entry| match entry {
            Entry::Call(call) => Some(call.clone()),
            _ => None,
        })
        .collect();
    let answer = transcript.iter().rev().find_map(|entry| match entry {
        Entry::Assistant(text) => Some(text.clone()),
        _ => None,
    });
    (calls, answer)
}

/// How two transcripts differ in tool calls or final answer, if they do.
pub fn compare(recorded: &[Entry], replayed: &[Entry]) -> Option<String> {
    let (recorded_calls, recorded_answer) = outline(recorded);
    let (replayed_calls, replayed_answer) = outline(replayed);
    if recorded_calls != replayed_calls {
        return Some(format!(
            "tool calls {recorded_calls:?} were recorded, {replayed_calls:?} replayed"
        ));
    }
    if recorded_answer != replayed_answer {
        return Some(format!(
            "the final answer {recorded_answer:?} was recorded, {replayed_answer:?} replayed"
        ));
    }
    None
}

fn save_session(world: &mut World) {
    let Some(SessionFile(path)) = world
        .get_resource::<SessionFile>()
        .map(|file| SessionFile(file.0.clone()))
    else {
        return;
    };
    let Some(agent) = tui_agent(world) else {
        return;
    };
    let recorded = Recorded {
        inputs: world
            .get::<Inputs>(agent)
            .map(|inputs| {
                inputs
                    .0
                    .iter()
                    .filter(|line| !matches!(line.as_str(), "/quit" | "/reload"))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default(),
        transcript: world
            .get::<Transcript>(agent)
            .map(|transcript| transcript.0.clone())
            .unwrap_or_default(),
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_vec_pretty(&recorded) {
        let _ = std::fs::write(path, json);
    }
}

/// The name rig-cassette knows a provider by.
fn cassette_provider(vendor: &str) -> &'static str {
    match vendor {
        "openai" => "openai",
        "anthropic" => "anthropic",
        "gcp.gemini" => "gemini",
        "deepseek" => "deepseek",
        _ => "provider",
    }
}

/// Writes recordings, or checks that replays were played in full. Returns
/// one line per provider that failed.
pub fn finish(world: &World) -> Vec<String> {
    let Some(cassettes) = world.get_resource::<Cassettes>() else {
        return Vec::new();
    };
    let open: Vec<(String, ProviderCassette)> = cassettes
        .0
        .lock()
        .map(|mut open| open.drain().collect())
        .unwrap_or_default();
    let runtime = &world.resource::<RigRuntime>().0;
    open.into_iter()
        .filter_map(|(provider, cassette)| {
            runtime
                .block_on(cassette.try_finish())
                .err()
                .map(|error| format!("{provider}: {error}"))
        })
        .collect()
}

fn finish_on_exit(world: &mut World) {
    if world.resource::<Messages<AppExit>>().is_empty() {
        return;
    }
    save_session(world);
    for failure in finish(world) {
        eprintln!("cassette {failure}");
    }
}
