//! External processes register tools, poll calls, and return results over BRP.
#[cfg(test)]
mod tests;
use crate::{Agent, reload::Reloader};
use anyhow::{Result, anyhow, ensure};
use bevy::prelude::*;
use bevy_remote::{BrpError, BrpResult, RemotePlugin};
use rig_core::{completion::ToolDefinition, message::ToolName};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};

#[derive(Clone, Resource, Default)]
pub struct Extensions(pub Arc<Mutex<Registry>>);

#[derive(Default)]
pub struct Registry {
    tools: BTreeMap<String, ToolDefinition>,
    pending: BTreeMap<u64, Pending>,
    next_id: u64,
}

struct Pending {
    call: Call,
    result: mpsc::Sender<String>,
}

#[derive(Clone, Serialize)]
struct Call {
    id: u64,
    name: String,
    arguments: Value,
}

#[derive(Deserialize)]
struct Registration {
    name: String,
    description: String,
    parameters: Value,
}

impl Extensions {
    pub fn definitions(&self) -> Result<Vec<ToolDefinition>> {
        Ok(self
            .0
            .lock()
            .map_err(|_| anyhow!("extension lock poisoned"))?
            .tools
            .values()
            .cloned()
            .collect())
    }

    pub fn call(&self, name: &str, arguments: Value) -> Result<String> {
        let (tx, rx) = mpsc::channel();
        let id = {
            let mut registry = self
                .0
                .lock()
                .map_err(|_| anyhow!("extension lock poisoned"))?;
            ensure!(registry.tools.contains_key(name), "unknown tool: {name}");
            registry.next_id += 1;
            let id = registry.next_id;
            registry.pending.insert(
                id,
                Pending {
                    call: Call {
                        id,
                        name: name.into(),
                        arguments,
                    },
                    result: tx,
                },
            );
            id
        };
        let result = rx.recv_timeout(Duration::from_secs(30));
        self.0
            .lock()
            .map_err(|_| anyhow!("extension lock poisoned"))?
            .pending
            .remove(&id);
        Ok(result?)
    }
}

pub fn plugin() -> RemotePlugin {
    RemotePlugin::default()
        .with_method_main("sprout.register", register)
        .with_method_main("sprout.poll", poll)
        .with_method_main("sprout.result", result)
        .with_method_main("sprout.status", status)
}

fn register(In(params): In<Option<Value>>, extensions: Res<Extensions>) -> BrpResult {
    let registration: Registration =
        serde_json::from_value(params.unwrap_or(Value::Null)).map_err(BrpError::internal)?;
    if registration.name == "shell"
        || !registration.parameters.is_object()
        || registration.description.len() > 4096
    {
        return Err(BrpError::internal("invalid or reserved tool definition"));
    }
    let name = ToolName::new(&registration.name).map_err(BrpError::internal)?;
    let mut registry = extensions.0.lock().map_err(BrpError::internal)?;
    if registry.tools.len() >= 32 || registry.tools.contains_key(&registration.name) {
        return Err(BrpError::internal("tool limit or duplicate name"));
    }
    registry.tools.insert(
        registration.name,
        ToolDefinition::new(name, registration.description, registration.parameters),
    );
    Ok(json!({"registered": true}))
}

fn poll(In(params): In<Option<Value>>, extensions: Res<Extensions>) -> BrpResult {
    let name = params
        .as_ref()
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .ok_or_else(|| BrpError::internal("name required"))?;
    let registry = extensions.0.lock().map_err(BrpError::internal)?;
    Ok(json!(
        registry
            .pending
            .values()
            .filter(|p| p.call.name == name)
            .map(|p| &p.call)
            .collect::<Vec<_>>()
    ))
}

fn result(In(params): In<Option<Value>>, extensions: Res<Extensions>) -> BrpResult {
    let params = params.ok_or_else(|| BrpError::internal("params required"))?;
    let id = params
        .get("id")
        .and_then(Value::as_u64)
        .ok_or_else(|| BrpError::internal("id required"))?;
    let text = params
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| BrpError::internal("text required"))?;
    if text.len() > 32768 {
        return Err(BrpError::internal("result exceeds 32 KiB"));
    }
    let pending = extensions
        .0
        .lock()
        .map_err(BrpError::internal)?
        .pending
        .remove(&id)
        .ok_or_else(|| BrpError::internal("unknown or expired call"))?;
    pending
        .result
        .send(text.into())
        .map_err(BrpError::internal)?;
    Ok(json!({"accepted": true}))
}

fn status(
    In(_): In<Option<Value>>,
    agent: Res<Agent>,
    native: Res<sprout_native::NativeState>,
    reload: Res<Reloader>,
) -> BrpResult {
    Ok(
        json!({"pid": std::process::id(), "busy": agent.busy, "transcript": agent.lines,
        "native": native.label, "ticks": native.ticks, "generation": reload.generation}),
    )
}
