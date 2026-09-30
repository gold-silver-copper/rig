//! The Bevy Remote Protocol server, on a port that stays the same across
//! reloads. Plugins add their methods with [`add_method`] and
//! [`add_watching_method`]; `rig_pi.status` describes the process and every
//! agent. The built-in `world.*` methods work too.

use bevy::prelude::*;
use bevy_remote::http::RemoteHttpPlugin;
use bevy_remote::{
    BrpError, BrpResult, RemoteMethodSystemId, RemoteMethods, RemotePlugin, error_codes,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::Env;
use crate::glue::{Agent, Model, Session, Tools, Turn};
use crate::session::{Focused, Kind};

pub struct BrpPlugin;

impl Plugin for BrpPlugin {
    fn build(&self, app: &mut App) {
        let port = app.world().resource::<Env>().port;
        // bevy_remote reports a failed bind nowhere, so check the port first.
        if let Err(error) = std::net::TcpListener::bind(("127.0.0.1", port)) {
            error!("BRP port {port} is unavailable: {error}");
        }
        app.add_plugins((
            RemotePlugin::default().with_method_main("rig_pi.status", status),
            RemoteHttpPlugin::default().with_port(port),
        ));
    }
}

/// Serve `name` with `handler`, which answers once.
pub fn add_method<M>(
    app: &mut App,
    name: &str,
    handler: impl IntoSystem<In<Option<Value>>, BrpResult, M> + 'static,
) {
    let id = app.world_mut().register_system(handler);
    app.world_mut()
        .resource_mut::<RemoteMethods>()
        .insert(name, RemoteMethodSystemId::Instant(id));
}

/// Serve `name` (which must end in `+watch`) with `handler`, which runs every
/// frame while the stream is open and sends each `Some` it returns.
pub fn add_watching_method<M>(
    app: &mut App,
    name: &str,
    handler: impl IntoSystem<In<Option<Value>>, BrpResult<Option<Value>>, M> + 'static,
) {
    let id = app.world_mut().register_system(handler);
    app.world_mut()
        .resource_mut::<RemoteMethods>()
        .insert(name, RemoteMethodSystemId::Watching(id));
}

/// Parse method parameters, answering bad ones with `INVALID_PARAMS`.
pub fn params<T: DeserializeOwned>(params: Option<Value>) -> BrpResult<T> {
    serde_json::from_value(params.unwrap_or(Value::Null)).map_err(|error| invalid(error.to_string()))
}

pub fn invalid(message: impl Into<String>) -> BrpError {
    BrpError {
        code: error_codes::INVALID_PARAMS,
        message: message.into(),
        data: None,
    }
}

pub fn turn_state(turn: &Turn) -> &'static str {
    match turn {
        Turn::Idle => "idle",
        Turn::Ready | Turn::Thinking(_) => "thinking",
        Turn::Acting => "tools",
    }
}

fn status(
    In(_): In<Option<Value>>,
    agents: Query<(&Name, &Session, &Model, &Turn, &Kind, Has<Focused>), With<Agent>>,
    tools: Res<Tools>,
    env: Res<Env>,
) -> BrpResult {
    let agents: Vec<Value> = agents
        .iter()
        .map(|(name, session, model, turn, kind, focused)| {
            json!({
                "session": session.0.as_str(),
                "name": name.as_str(),
                "kind": format!("{kind:?}"),
                "model": model.0.to_string(),
                "state": turn_state(turn),
                "focused": focused,
            })
        })
        .collect();
    Ok(json!({
        "pid": std::process::id(),
        "port": env.port,
        "tools": tools.0.keys().collect::<Vec<_>>(),
        "agents": agents,
    }))
}
