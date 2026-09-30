//! MCP: the tools of the MCP servers listed in a JSON file become the
//! model's tools, named `mcp__<server>__<tool>`. Rig's `rig-rmcp` turns each
//! server tool into a `DynamicTool`; calls run on the shared tokio runtime.
//!
//! ```json
//! {"servers": {"words": {"command": "target/debug/examples/mcp_server", "args": []}}}
//! ```
//!
//! `--mcp FILE` names the file; without it `./mcp.json` is used if present.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use bevy::prelude::*;
use rig_core::tool::{DynamicTool, ToolExecutionError};
use rmcp::ServiceExt;
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use serde::Deserialize;

use crate::glue::{Tool, Tools};
use crate::plugins::runtime::tokio;
use crate::session::{EntryKind, Focused, Transcript};

pub struct McpPlugin {
    pub config: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Config {
    #[serde(alias = "mcpServers")]
    servers: BTreeMap<String, Server>,
}

#[derive(Deserialize)]
struct Server {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

/// The live connections; dropping one closes its server.
#[derive(Resource, Default)]
struct Connections(Vec<RunningService<RoleClient, ()>>);

/// What connecting reported, shown once the TUI session exists.
#[derive(Resource, Default)]
struct Report(Vec<(EntryKind, String)>);

impl Plugin for McpPlugin {
    fn build(&self, app: &mut App) {
        let path = self.config.clone().or_else(|| {
            let default = PathBuf::from("mcp.json");
            default.exists().then_some(default)
        });
        let Some(path) = path else { return };
        let mut report = Report::default();
        let mut connections = Connections::default();
        match read(&path) {
            Err(error) => report.0.push((EntryKind::Error, error)),
            Ok(config) => {
                let runtime = tokio(app.world_mut());
                for (name, server) in config.servers {
                    match runtime.0.block_on(connect(&server)) {
                        Ok((service, tools)) => {
                            let handle = runtime.0.handle().clone();
                            let names: Vec<String> = tools
                                .into_iter()
                                .map(|tool| {
                                    let tool = on_runtime(&name, tool.into(), handle.clone());
                                    let registered = tool.name().to_owned();
                                    app.world_mut().get_resource_or_init::<Tools>().add(Tool::Task(tool));
                                    registered
                                })
                                .collect();
                            report.0.push((
                                EntryKind::Info,
                                format!("MCP server `{name}`: {}", names.join(", ")),
                            ));
                            connections.0.push(service);
                        }
                        Err(error) => report.0.push((EntryKind::Error, format!("MCP server `{name}`: {error}"))),
                    }
                }
            }
        }
        app.insert_resource(connections)
            .insert_resource(report)
            .add_systems(Update, show_report);
    }
}

fn read(path: &Path) -> Result<Config, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

async fn connect(
    server: &Server,
) -> Result<(RunningService<RoleClient, ()>, Vec<rig_rmcp::McpTool>), String> {
    let mut command = tokio::process::Command::new(&server.command);
    command.args(&server.args).envs(&server.env);
    let transport = TokioChildProcess::new(command).map_err(|e| e.to_string())?;
    let service = tokio::time::timeout(Duration::from_secs(20), ().serve(transport))
        .await
        .map_err(|_| "did not initialize within 20s".to_owned())?
        .map_err(|e| e.to_string())?;
    let tools = service.list_all_tools().await.map_err(|e| e.to_string())?;
    let tools = rig_rmcp::tools_from_server(tools, service.peer());
    Ok((service, tools))
}

/// `tool` renamed for its server, with calls spawned on `runtime` (rmcp
/// needs tokio; the agent's tool pool is not tokio).
fn on_runtime(server: &str, tool: DynamicTool, runtime: tokio::runtime::Handle) -> DynamicTool {
    let definition = tool.definition();
    let name: String = format!("mcp__{server}__{}", definition.name)
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .take(64)
        .collect();
    DynamicTool::new(name, definition.description, definition.parameters, move |args| {
        let (tool, runtime) = (tool.clone(), runtime.clone());
        Box::pin(async move {
            runtime
                .spawn(async move { tool.execute(args).await })
                .await
                .map_err(|error| ToolExecutionError::other(error.to_string()))?
        })
    })
}

fn show_report(mut report: ResMut<Report>, mut focused: Query<&mut Transcript, With<Focused>>) {
    if report.0.is_empty() {
        return;
    }
    if let Ok(mut transcript) = focused.single_mut() {
        for (kind, text) in report.0.drain(..) {
            transcript.log(kind, text);
        }
    }
}
