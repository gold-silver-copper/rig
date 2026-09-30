//! The tool registry. Native plugins add tools with [`AgentAppExt::add_tool`];
//! external processes add them over BRP (see `brp.rs`).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use serde_json::Value;

/// A native tool body. It runs on its own thread, so it may block.
pub type ToolFn = Arc<dyn Fn(Value) -> Result<String, String> + Send + Sync>;

pub enum Handler {
    Native(ToolFn),
    /// Answered by the external process that registered it, over BRP.
    Remote { plugin: String },
    /// Rebuilds the agent and restarts into the new binary.
    Reload,
}

pub struct Tool {
    pub def: ToolDefinition,
    pub handler: Handler,
}

#[derive(Resource, Default)]
pub struct Tools(pub BTreeMap<String, Tool>);

impl Tools {
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.0.values().map(|tool| tool.def.clone()).collect()
    }

    pub fn insert(&mut self, def: ToolDefinition, handler: Handler) {
        self.0.insert(def.name.clone(), Tool { def, handler });
    }
}

pub fn definition(name: &str, description: &str, parameters: Value) -> ToolDefinition {
    let name = ToolName::new(name).unwrap_or_else(|_| panic!("tool names are nonempty"));
    ToolDefinition::new(name, description, parameters)
}

pub trait AgentAppExt {
    /// Register a native tool: `run` gets the model's JSON arguments.
    fn add_tool(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        run: impl Fn(Value) -> Result<String, String> + Send + Sync + 'static,
    ) -> &mut Self;
}

impl AgentAppExt for App {
    fn add_tool(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        run: impl Fn(Value) -> Result<String, String> + Send + Sync + 'static,
    ) -> &mut Self {
        self.init_resource::<Tools>();
        self.world_mut().resource_mut::<Tools>().insert(
            definition(name, description, parameters),
            Handler::Native(Arc::new(run)),
        );
        self
    }
}

/// A result filled in from another thread (a tool thread, a build, a BRP call).
pub type Slot<T> = Arc<Mutex<Option<T>>>;

pub fn slot<T>() -> Slot<T> {
    Arc::new(Mutex::new(None))
}

pub fn take<T>(slot: &Slot<T>) -> Option<T> {
    slot.lock().ok()?.take()
}

pub fn fill<T>(slot: &Slot<T>, value: T) {
    if let Ok(mut guard) = slot.lock() {
        *guard = Some(value);
    }
}

/// Run a native tool on its own thread; its output lands in the returned slot.
pub fn spawn_native(run: ToolFn, args: Value) -> Slot<String> {
    let out = slot();
    let sink = out.clone();
    std::thread::spawn(move || {
        let text = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(args))) {
            Ok(Ok(text)) => text,
            Ok(Err(error)) => format!("error: {error}"),
            Err(_) => "error: the tool panicked".to_owned(),
        };
        fill(&sink, text);
    });
    out
}
