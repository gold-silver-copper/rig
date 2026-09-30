use crate::{
    UiState,
    tools::{Extension, Extensions, validate_extension},
};
use bevy::prelude::*;
use bevy_remote::{BrpError, BrpResult, RemotePlugin, http::RemoteHttpPlugin};
use serde_json::{Value, json};

#[derive(Resource)]
pub struct Registry(pub Extensions);

pub fn plugins(port: u16) -> (RemotePlugin, RemoteHttpPlugin) {
    (
        RemotePlugin::default()
            .with_method_main("agent/status", status)
            .with_method_main("agent/prompt", prompt)
            .with_method_main("agent/extend", extend)
            .with_method_main("agent/unextend", unextend),
        RemoteHttpPlugin::default()
            .with_address(std::net::Ipv4Addr::LOCALHOST)
            .with_port(port),
    )
}

fn status(
    In(_): In<Option<Value>>,
    ui: Res<UiState>,
    native: Res<crate::native::Native>,
    registry: Res<Registry>,
) -> BrpResult {
    let extensions: Vec<_> = registry
        .0
        .read()
        .map_err(BrpError::internal)?
        .keys()
        .cloned()
        .collect();
    Ok(
        json!({"pid":std::process::id(), "busy":ui.busy, "native":native.status, "generation":native.generation, "extensions":extensions, "lines":ui.lines}),
    )
}

fn prompt(In(params): In<Option<Value>>, mut ui: ResMut<UiState>) -> BrpResult {
    let params = params.ok_or_else(|| BrpError::internal("missing params"))?;
    let text = params
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| BrpError::internal("missing text"))?;
    ui.submit(text.to_owned()).map_err(BrpError::internal)?;
    Ok(json!({"accepted":true}))
}

fn extend(
    In(params): In<Option<Value>>,
    registry: Res<Registry>,
    mut ui: ResMut<UiState>,
) -> BrpResult {
    let extension: Extension =
        serde_json::from_value(params.ok_or_else(|| BrpError::internal("missing params"))?)
            .map_err(BrpError::internal)?;
    validate_extension(&extension).map_err(BrpError::internal)?;
    let name = extension.name.clone();
    registry
        .0
        .write()
        .map_err(BrpError::internal)?
        .insert(name.clone(), extension);
    ui.lines.push(format!("BRP extension registered: {name}"));
    Ok(json!({"registered":name}))
}

fn unextend(In(params): In<Option<Value>>, registry: Res<Registry>) -> BrpResult {
    let params = params.ok_or_else(|| BrpError::internal("missing params"))?;
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| BrpError::internal("missing name"))?;
    let removed = registry
        .0
        .write()
        .map_err(BrpError::internal)?
        .remove(name)
        .is_some();
    Ok(json!({"removed":removed}))
}
