//! Cassettes: record an agent's provider traffic with rig-cassette's HTTP
//! engine, or replay it with no API keys and no network.
//!
//! An agent with a [`Cassette`] has each vendor it talks to routed through a
//! `ProviderCassette` named after the cassette: a recording proxy in front
//! of the real API, or a replay server. Cassettes are written to
//! `fixtures/cassettes/<vendor>/<name>.yaml` in the agent's crate when the
//! process exits. `--record NAME` and `--replay NAME` put one on the TUI
//! session; evals put one on each eval run.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;

use bevy::prelude::*;
use futures::FutureExt;
use rig_cassette::http::{CassetteMode, ProviderCassette, RecordVia, cassette_path};
use rig_core::providers::registry::ProviderId;

use crate::Env;
use crate::glue::{AgentSet, Endpoint, Endpoints, Model};
use crate::plugins::runtime::tokio;
use crate::session::{EntryKind, Focused, Transcript};

/// Route this agent's provider traffic through cassette `name`.
#[derive(Component, Clone, Debug)]
pub struct Cassette {
    pub name: String,
    pub mode: CassetteMode,
}

pub struct CassettePlugin {
    pub record: Option<String>,
    pub replay: Option<String>,
}

/// The open cassette sessions, by cassette name and vendor.
#[derive(Resource)]
pub struct Cassettes {
    pub root: PathBuf,
    open: HashMap<(String, String), ProviderCassette>,
}

impl Plugin for CassettePlugin {
    fn build(&self, app: &mut App) {
        let root = app.world().resource::<Env>().source.join("fixtures").join("cassettes");
        app.insert_resource(Cassettes {
            root,
            open: HashMap::new(),
        })
        .add_systems(Update, route.in_set(AgentSet::Route))
        .add_systems(Last, finish_on_exit);
        let focused = match (&self.record, &self.replay) {
            (Some(name), _) => Some(Cassette {
                name: name.clone(),
                mode: CassetteMode::Record,
            }),
            (None, Some(name)) => Some(Cassette {
                name: name.clone(),
                mode: CassetteMode::Replay,
            }),
            (None, None) => None,
        };
        if let Some(cassette) = focused {
            app.add_systems(PostStartup, move |mut commands: Commands, agents: Query<Entity, With<Focused>>| {
                for agent in &agents {
                    commands.entity(agent).insert(cassette.clone());
                }
            });
        }
    }
}

/// Make sure every cassette agent's current vendor has an open session and
/// an endpoint pointing at it, before its next request.
fn route(world: &mut World) {
    let mut wanted = Vec::new();
    for (entity, cassette, model, endpoints) in world
        .query::<(Entity, &Cassette, &Model, Option<&Endpoints>)>()
        .iter(world)
    {
        let Some(id) = model.0.id() else { continue };
        let vendor = id.vendor().to_owned();
        if !endpoints.is_some_and(|e| e.0.contains_key(&vendor)) {
            wanted.push((entity, cassette.clone(), id, vendor));
        }
    }
    if wanted.is_empty() {
        return;
    }
    let runtime = tokio(world);
    for (entity, cassette, id, vendor) in wanted {
        let key = (cassette.name.clone(), vendor.clone());
        let endpoint = match open(world, &runtime.0, &cassette, id, &vendor) {
            Ok(endpoint) => endpoint,
            Err(error) => {
                if let Some(mut transcript) = world.get_mut::<Transcript>(entity) {
                    transcript.log(EntryKind::Error, format!("cassette {}: {error}", key.0));
                }
                // Never fall through to the network: send to a closed port.
                Endpoint {
                    base_url: "http://127.0.0.1:9".into(),
                    api_key: "[REDACTED]".into(),
                }
            }
        };
        let mut entity = world.entity_mut(entity);
        match entity.get_mut::<Endpoints>() {
            Some(mut endpoints) => {
                endpoints.0.insert(vendor, endpoint);
            }
            None => {
                entity.insert(Endpoints([(vendor, endpoint)].into_iter().collect()));
            }
        }
    }
}

fn open(
    world: &mut World,
    runtime: &tokio::runtime::Runtime,
    cassette: &Cassette,
    id: ProviderId,
    vendor: &str,
) -> Result<Endpoint, String> {
    let mut cassettes = world.resource_mut::<Cassettes>();
    let key = (cassette.name.clone(), vendor.to_owned());
    if !cassettes.open.contains_key(&key) {
        let path = cassette_path(&cassettes.root, vendor, &cassette.name);
        if cassette.mode == CassetteMode::Replay && !path.exists() {
            return Err(format!("no recording for {vendor} at {}", path.display()));
        }
        let upstream = id.config("").base_url().to_owned();
        let session = runtime.block_on(ProviderCassette::start_named(
            RecordVia::Proxy,
            vendor,
            &cassette.name,
            &upstream,
            cassette.mode,
            path,
        ));
        cassettes.open.insert(key.clone(), session);
    }
    let session = &cassettes.open[&key];
    let api_key = match cassette.mode {
        CassetteMode::Record => std::env::var(id.api_key_env())
            .map_err(|_| format!("{} is not set", id.api_key_env()))?,
        CassetteMode::Replay => session.api_key(id.api_key_env()),
    };
    Ok(Endpoint {
        base_url: session.base_url(),
        api_key,
    })
}

/// Write recordings, and check replays were used up, when the process ends.
fn finish_on_exit(world: &mut World) {
    let exiting = world.resource_mut::<Messages<AppExit>>().iter_current_update_messages().next().is_some();
    if !exiting {
        return;
    }
    let sessions: Vec<_> = world.resource_mut::<Cassettes>().open.drain().collect();
    if sessions.is_empty() {
        return;
    }
    let runtime = tokio(world);
    for ((name, vendor), session) in sessions {
        let outcome = runtime.0.block_on(AssertUnwindSafe(session.finish()).catch_unwind());
        if let Err(payload) = outcome {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default();
            error!("cassette {name} ({vendor}): {message}");
        }
    }
}
