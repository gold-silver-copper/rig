//! Remote sessions over BRP, so another process can drive agents in this
//! one, next to the TUI session and each with its own conversation and
//! model:
//!
//! - `session.create` with `{"model"?}` opens a session and returns `{"id"}`.
//! - `session.list` returns every session with its origin, model and state.
//! - `session.prompt` with `{"id", "text"}` sends a prompt or command, as typing it would.
//! - `session.transcript` with `{"id", "since"?}` returns the transcript
//!   entries from `since` on, the next index, and whether the session is busy.
//! - `session.cancel` and `session.close` with `{"id"}` stop a turn and end a session.
//!
//! Requires the BRP plugin.

use bevy::prelude::*;
use bevy_remote::{BrpError, BrpResult, error_codes};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::glue::{self, Model, Turn};
use crate::plugins::brp::{add_method, params};
use crate::session::{self, Origin, Prompts, SessionId, SessionStore, Transcript};
use crate::{KeepSessions, State};

pub struct RemoteSessionsPlugin;

impl Plugin for RemoteSessionsPlugin {
    fn build(&self, app: &mut App) {
        add_method(app, "session.create", create);
        add_method(app, "session.list", list);
        add_method(app, "session.prompt", prompt);
        add_method(app, "session.transcript", transcript);
        add_method(app, "session.cancel", cancel);
        add_method(app, "session.close", close);
    }
}

#[derive(Deserialize)]
struct Id {
    id: String,
}

fn find(world: &mut World, id: &str) -> BrpResult<Entity> {
    world
        .query::<(Entity, &SessionId)>()
        .iter(world)
        .find(|(_, session)| session.0 == id)
        .map(|(entity, _)| entity)
        .ok_or_else(|| BrpError {
            code: error_codes::INVALID_PARAMS,
            message: format!("no session `{id}`"),
            data: None,
        })
}

#[derive(Deserialize, Default)]
struct CreateParams {
    model: Option<String>,
}

fn create(In(raw): In<Option<Value>>, world: &mut World) -> BrpResult {
    let CreateParams { model } = raw
        .map(|raw| params(Some(raw)))
        .transpose()?
        .unwrap_or_default();
    let model = model.unwrap_or_else(|| world.resource::<State>().model.clone());
    let id = session::new_session_id();
    let agent = session::spawn_agent(world, id.clone(), Origin::Remote, &model);
    if let Some(error) = world.get::<Model>(agent).and_then(Model::error) {
        let message = format!("cannot use {model}: {error}");
        world.despawn(agent);
        return Err(BrpError {
            code: error_codes::INVALID_PARAMS,
            message,
            data: None,
        });
    }
    Ok(json!({"id": id}))
}

fn busy(turn: &Turn, prompts: &Prompts) -> bool {
    *turn != Turn::Idle || !prompts.queue.is_empty()
}

fn list(In(_): In<Option<Value>>, world: &mut World) -> BrpResult {
    let sessions: Vec<Value> = world
        .query::<(&SessionId, &Origin, &Model, &Turn, &Prompts)>()
        .iter(world)
        .map(|(id, origin, model, turn, prompts)| {
            json!({
                "id": id.0,
                "origin": format!("{origin:?}"),
                "model": model.spec(),
                "busy": busy(turn, prompts),
            })
        })
        .collect();
    Ok(Value::Array(sessions))
}

#[derive(Deserialize)]
struct PromptParams {
    id: String,
    text: String,
}

fn prompt(In(raw): In<Option<Value>>, world: &mut World) -> BrpResult {
    let PromptParams { id, text } = params(raw)?;
    let agent = find(world, &id)?;
    session::submit(world, agent, &text);
    Ok(Value::Null)
}

#[derive(Deserialize)]
struct TranscriptParams {
    id: String,
    #[serde(default)]
    since: usize,
}

fn transcript(In(raw): In<Option<Value>>, world: &mut World) -> BrpResult {
    let TranscriptParams { id, since } = params(raw)?;
    let agent = find(world, &id)?;
    let entity = world.entity(agent);
    let (Some(transcript), Some(turn), Some(prompts)) = (
        entity.get::<Transcript>(),
        entity.get::<Turn>(),
        entity.get::<Prompts>(),
    ) else {
        return Ok(Value::Null);
    };
    let building = world.resource::<crate::reload::Reload>().building();
    Ok(json!({
        "entries": transcript.0.get(since..).unwrap_or_default(),
        "next": transcript.0.len(),
        "busy": busy(turn, prompts) || building,
        "model": entity.get::<Model>().map(Model::spec),
    }))
}

fn cancel(In(raw): In<Option<Value>>, world: &mut World) -> BrpResult {
    let Id { id } = params(raw)?;
    let agent = find(world, &id)?;
    glue::cancel(world, agent);
    Ok(Value::Null)
}

fn close(In(raw): In<Option<Value>>, world: &mut World) -> BrpResult {
    let Id { id } = params(raw)?;
    let agent = find(world, &id)?;
    if world.get::<Origin>(agent) == Some(&Origin::Tui) {
        return Err(BrpError {
            code: error_codes::INVALID_PARAMS,
            message: "the TUI session cannot be closed remotely".into(),
            data: None,
        });
    }
    glue::cancel(world, agent);
    world.despawn(agent);
    if !world.contains_resource::<KeepSessions>() {
        world.resource::<SessionStore>().remove(&id);
    }
    Ok(Value::Null)
}
