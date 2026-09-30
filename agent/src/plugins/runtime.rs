//! One tokio runtime for the plugins whose Rig integrations need tokio (MCP,
//! cassettes). The glue itself runs on Bevy's task pools.

use std::sync::Arc;

use bevy::prelude::*;

#[derive(Resource, Clone)]
pub struct Tokio(pub Arc<tokio::runtime::Runtime>);

/// The shared runtime, created on first use.
pub fn tokio(world: &mut World) -> Tokio {
    if let Some(runtime) = world.get_resource::<Tokio>() {
        return runtime.clone();
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("rig-pi-tokio")
        .enable_all()
        .build()
        .unwrap_or_else(|error| panic!("tokio runtime: {error}"));
    let runtime = Tokio(Arc::new(runtime));
    world.insert_resource(runtime.clone());
    runtime
}
