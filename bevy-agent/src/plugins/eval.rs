//! Evals: behavioral checks of the agent, in the spirit of pi's evals. A
//! suite is a JSON file of cases; each case runs as an eval agent entity in
//! the same world, one after another, and passes when it called the
//! expected tools and its final answer contains the expected text. With the
//! cassette plugin, a suite records its provider traffic once and replays
//! it afterwards without API keys.

use std::path::Path;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use serde::Deserialize;

use crate::glue::Turn;
use crate::plugins::cassette;
use crate::session::{self, Entry, Origin, Prompts, Transcript};

#[derive(Deserialize, Clone)]
pub struct Suite {
    /// A directory of files the cases work on, relative to the suite file.
    pub workspace: Option<String>,
    pub cases: Vec<Case>,
}

#[derive(Deserialize, Clone)]
pub struct Case {
    pub name: String,
    pub model: String,
    pub prompts: Vec<String>,
    #[serde(default)]
    pub expect: Expect,
}

#[derive(Deserialize, Clone, Default)]
pub struct Expect {
    /// Tools the case must call, in any order.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Text the final answer must contain, ignoring case.
    #[serde(default)]
    pub answer_contains: Vec<String>,
}

impl Suite {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
    }
}

/// What one case did.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Runs a suite, then exits: status 0 only when every case passed.
pub struct EvalPlugin {
    pub suite: Suite,
    pub timeout: Duration,
}

#[derive(Resource)]
pub struct Evals {
    cases: Vec<Case>,
    timeout: Duration,
    running: Option<(Entity, Instant)>,
    pub outcomes: Vec<Outcome>,
    done: bool,
}

impl Plugin for EvalPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Evals {
            cases: self.suite.cases.iter().rev().cloned().collect(),
            timeout: self.timeout,
            running: None,
            outcomes: Vec::new(),
            done: false,
        })
        .add_systems(Update, run_cases);
    }
}

fn run_cases(world: &mut World) {
    if let Some((agent, started)) = world.resource::<Evals>().running {
        let finished = world.get::<Turn>(agent) == Some(&Turn::Idle)
            && world
                .get::<Prompts>(agent)
                .is_some_and(|prompts| prompts.queue.is_empty());
        let timed_out = started.elapsed() > world.resource::<Evals>().timeout;
        if !finished && !timed_out {
            return;
        }
        let transcript = world
            .get::<Transcript>(agent)
            .map(|transcript| transcript.0.clone())
            .unwrap_or_default();
        crate::glue::cancel(world, agent);
        world.despawn(agent);
        let mut evals = world.resource_mut::<Evals>();
        evals.running = None;
        let outcome = judge(
            evals.pending_name(),
            &transcript,
            timed_out,
            evals.pending_expect(),
        );
        println!(
            "{} {}: {}",
            if outcome.passed { "PASS" } else { "FAIL" },
            outcome.name,
            outcome.detail
        );
        evals.outcomes.push(outcome);
        evals.cases.pop();
    }
    let Some(case) = world.resource::<Evals>().cases.last().cloned() else {
        finish(world);
        return;
    };
    let agent = session::spawn_agent(world, session::new_session_id(), Origin::Eval, &case.model);
    for prompt in &case.prompts {
        session::submit(world, agent, prompt);
    }
    world.resource_mut::<Evals>().running = Some((agent, Instant::now()));
}

impl Evals {
    fn pending_name(&self) -> String {
        self.cases
            .last()
            .map(|case| case.name.clone())
            .unwrap_or_default()
    }

    fn pending_expect(&self) -> Expect {
        self.cases
            .last()
            .map(|case| case.expect.clone())
            .unwrap_or_default()
    }
}

/// Checks a finished case's transcript against its expectations.
pub fn judge(name: String, transcript: &[Entry], timed_out: bool, expect: Expect) -> Outcome {
    let (calls, answer) = cassette::outline(transcript);
    let called: Vec<&str> = calls
        .iter()
        .filter_map(|call| call.split_whitespace().next())
        .collect();
    let answer = answer.unwrap_or_default();
    let mut problems = Vec::new();
    if timed_out {
        problems.push("timed out".to_owned());
    }
    for entry in transcript {
        if let Entry::Error(error) = entry {
            problems.push(format!("error: {error}"));
        }
    }
    for tool in &expect.tools {
        if !called.contains(&tool.as_str()) {
            problems.push(format!("did not call {tool}"));
        }
    }
    for text in &expect.answer_contains {
        if !answer.to_lowercase().contains(&text.to_lowercase()) {
            problems.push(format!("answer lacks {text:?}"));
        }
    }
    let passed = problems.is_empty();
    let detail = if passed {
        format!(
            "called {called:?}; answered {:?}",
            answer.lines().next().unwrap_or_default()
        )
    } else {
        problems.join("; ")
    };
    Outcome {
        name,
        passed,
        detail,
    }
}

fn finish(world: &mut World) {
    if std::mem::replace(&mut world.resource_mut::<Evals>().done, true) {
        return;
    }
    let failures = cassette::finish(world);
    for failure in &failures {
        println!("FAIL cassette {failure}");
    }
    let outcomes = &world.resource::<Evals>().outcomes;
    let passed = outcomes.iter().filter(|outcome| outcome.passed).count();
    println!("{passed}/{} cases passed", outcomes.len());
    let success = passed == outcomes.len() && failures.is_empty();
    world.write_message(if success {
        AppExit::Success
    } else {
        AppExit::error()
    });
}
