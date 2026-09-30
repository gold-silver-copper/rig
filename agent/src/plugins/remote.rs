//! Remote sessions, served over BRP. A remote session is an agent entity like
//! the TUI's, with its own conversation and model, running alongside it.
//! Clients can also drive the TUI session by its id.
//!
//! - `rig_pi.session.create {model?, name?}` → `{session}`
//! - `rig_pi.session.list` → every agent
//! - `rig_pi.session.prompt {session, text, steer?}`
//! - `rig_pi.session.model {session, model}`
//! - `rig_pi.session.get {session, since?}` → state, model and transcript
//!   entries from `since`
//! - `rig_pi.session.watch+watch {session, stream, since?}` streams new
//!   transcript entries; `stream` is any id unique to the connection
//! - `rig_pi.session.interrupt {session}`, `rig_pi.session.close {session}`

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_remote::BrpResult;
use rig_core::providers::registry::ProviderRef;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::Env;
use crate::brp::{add_method, add_watching_method, invalid, params, turn_state};
use crate::glue::{Agent, Inbox, Interrupt, Model, Session, Tools, Turn};
use crate::prompt::Prompt;
use crate::session::{Kind, Store, Transcript, default_model, forget, new_session_id, spawn_agent};
use crate::tui::parse_model;

pub struct RemoteSessionsPlugin;

impl Plugin for RemoteSessionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Cursors>();
        add_method(app, "rig_pi.session.create", create);
        add_method(app, "rig_pi.session.list", list);
        add_method(app, "rig_pi.session.prompt", prompt);
        add_method(app, "rig_pi.session.model", model);
        add_method(app, "rig_pi.session.get", get);
        add_method(app, "rig_pi.session.interrupt", interrupt);
        add_method(app, "rig_pi.session.close", close);
        add_watching_method(app, "rig_pi.session.watch+watch", watch);
    }
}

/// How far each watch stream has read.
#[derive(Resource, Default)]
struct Cursors(HashMap<String, usize>);

#[derive(Deserialize)]
struct Target {
    session: String,
}

fn find<'a, T>(
    agents: impl IntoIterator<Item = (Entity, &'a Session, T)>,
    session: &str,
) -> BrpResult<(Entity, T)> {
    agents
        .into_iter()
        .find(|(_, s, _)| s.0.as_str() == session)
        .map(|(entity, _, rest)| (entity, rest))
        .ok_or_else(|| invalid(format!("no session `{session}`")))
}

#[derive(Deserialize, Default)]
struct Create {
    model: Option<String>,
    name: Option<String>,
}

fn create(
    In(raw): In<Option<Value>>,
    prompt: Res<Prompt>,
    tools: Res<Tools>,
    mut commands: Commands,
) -> BrpResult {
    let create: Create = if raw.is_none() { Create::default() } else { params(raw)? };
    let model = match create.model {
        Some(model) => parse_model(&model).map_err(invalid)?,
        None => default_model(),
    };
    let id = new_session_id(Kind::Remote);
    let name = create.name.unwrap_or_else(|| "remote".into());
    spawn_agent(&mut commands, prompt.build(&tools), Kind::Remote, &name, model.clone(), id.clone());
    Ok(json!({"session": id.as_str(), "model": model.to_string()}))
}

fn list(
    In(_): In<Option<Value>>,
    agents: Query<(&Session, &Name, &Kind, &Model, &Turn), With<Agent>>,
) -> BrpResult {
    Ok(Value::Array(
        agents
            .iter()
            .map(|(session, name, kind, model, turn)| {
                json!({"session": session.0.as_str(), "name": name.as_str(), "kind": format!("{kind:?}"),
                       "model": model.0.to_string(), "state": turn_state(turn)})
            })
            .collect(),
    ))
}

#[derive(Deserialize)]
struct PromptParams {
    session: String,
    text: String,
    #[serde(default)]
    steer: bool,
}

fn prompt(
    In(raw): In<Option<Value>>,
    mut agents: Query<(Entity, &Session, (&mut Inbox, &Turn))>,
) -> BrpResult {
    let request: PromptParams = params(raw)?;
    let (_, (mut inbox, turn)) = find(agents.iter_mut().map(|(e, s, r)| (e, s, r)), &request.session)?;
    if request.steer && !turn.is_idle() {
        inbox.steering.push(request.text);
    } else {
        inbox.follow_ups.push_back(request.text);
    }
    Ok(json!({"queued": inbox.follow_ups.len() + inbox.steering.len()}))
}

#[derive(Deserialize)]
struct ModelParams {
    session: String,
    model: String,
}

fn model(In(raw): In<Option<Value>>, mut agents: Query<(Entity, &Session, &mut Model)>) -> BrpResult {
    let request: ModelParams = params(raw)?;
    let next: ProviderRef = parse_model(&request.model).map_err(invalid)?;
    let (_, mut model) = find(agents.iter_mut(), &request.session)?;
    model.0 = next;
    Ok(json!({"model": model.0.to_string()}))
}

#[derive(Deserialize)]
struct Get {
    session: String,
    #[serde(default)]
    since: usize,
}

fn entries(transcript: &Transcript, since: usize) -> Vec<Value> {
    transcript.0[since.min(transcript.0.len())..]
        .iter()
        .map(|entry| json!({"kind": format!("{:?}", entry.kind), "text": entry.text}))
        .collect()
}

fn get(
    In(raw): In<Option<Value>>,
    agents: Query<(Entity, &Session, (&Model, &Turn, &Transcript, &Inbox))>,
) -> BrpResult {
    let request: Get = params(raw)?;
    let (_, (model, turn, transcript, inbox)) = find(agents.iter(), &request.session)?;
    Ok(json!({
        "model": model.0.to_string(),
        "state": turn_state(turn),
        "queued": inbox.follow_ups.len() + inbox.steering.len(),
        "entries": entries(transcript, request.since),
        "next": transcript.0.len(),
    }))
}

#[derive(Deserialize)]
struct Watch {
    session: String,
    stream: String,
    #[serde(default)]
    since: usize,
}

/// Runs every frame for every open watch stream.
fn watch(
    In(raw): In<Option<Value>>,
    agents: Query<(Entity, &Session, &Transcript)>,
    mut cursors: ResMut<Cursors>,
) -> BrpResult<Option<Value>> {
    let request: Watch = params(raw)?;
    let (_, transcript) = find(agents.iter(), &request.session)?;
    let cursor = cursors.0.entry(request.stream).or_insert(request.since);
    if *cursor >= transcript.0.len() {
        return Ok(None);
    }
    let batch = entries(transcript, *cursor);
    *cursor = transcript.0.len();
    Ok(Some(json!({"entries": batch, "next": *cursor})))
}

fn interrupt(
    In(raw): In<Option<Value>>,
    agents: Query<(Entity, &Session, ())>,
    mut interrupts: MessageWriter<Interrupt>,
) -> BrpResult {
    let target: Target = params(raw)?;
    let (entity, ()) = find(agents.iter(), &target.session)?;
    interrupts.write(Interrupt(entity));
    Ok(json!({"ok": true}))
}

fn close(
    In(raw): In<Option<Value>>,
    agents: Query<(Entity, &Session, &Kind)>,
    env: Res<Env>,
    store: Res<Store>,
    mut commands: Commands,
) -> BrpResult {
    let target: Target = params(raw)?;
    let (entity, kind) = find(agents.iter(), &target.session)?;
    if *kind != Kind::Remote {
        return Err(invalid("only remote sessions can be closed"));
    }
    commands.entity(entity).despawn();
    forget(&env, &store, &rig_core::id::ConversationId::new(target.session));
    Ok(json!({"ok": true}))
}
