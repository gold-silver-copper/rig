//! MCP: tools from Model Context Protocol servers, through rig-rmcp. Each
//! server runs as a child process over stdio; its tools join the agent's as
//! `<server>__<tool>`, run by the glue like any Rig `DynamicTool`. Servers
//! come from `--mcp name=command args…` or an `mcp.json` in the
//! `{"mcpServers": {"name": {"command", "args"}}}` shape.

use std::path::Path;

use bevy::prelude::*;
use rig_core::tool::DynamicTool;
use rig_rmcp::McpTool;
use rmcp::ServiceExt;
use rmcp::service::RunningService;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use serde::Deserialize;

use crate::glue::{RigRuntime, Tool, Tools};
use crate::session::{Entry, Origin, Transcript};

#[derive(Clone, Deserialize)]
pub struct Server {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

pub struct McpPlugin {
    pub servers: Vec<(String, Server)>,
}

/// Parses `name=command args…`.
pub fn parse_server(text: &str) -> Option<(String, Server)> {
    let (name, command) = text.split_once('=')?;
    let mut words = command.split_whitespace().map(str::to_owned);
    Some((
        name.to_owned(),
        Server {
            command: words.next()?,
            args: words.collect(),
        },
    ))
}

/// Reads servers from an `mcp.json`.
pub fn load_config(path: &Path) -> Vec<(String, Server)> {
    #[derive(Deserialize)]
    struct Config {
        #[serde(rename = "mcpServers", default)]
        servers: std::collections::BTreeMap<String, Server>,
    }
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Config>(&bytes).ok())
        .map(|config| config.servers.into_iter().collect())
        .unwrap_or_default()
}

type Connected = Result<
    (
        String,
        RunningService<rmcp::RoleClient, ()>,
        Vec<DynamicTool>,
    ),
    String,
>;

/// The connected servers, kept alive, and what to tell the TUI about them.
#[derive(Resource)]
struct Mcp {
    #[allow(dead_code)]
    running: Vec<RunningService<rmcp::RoleClient, ()>>,
    notices: Vec<Entry>,
}

impl Plugin for McpPlugin {
    fn build(&self, app: &mut App) {
        // Servers connect before the first frame, so every request, the
        // first after a reload included, offers the same tools.
        let servers = self.servers.clone();
        let results = app.world().resource::<RigRuntime>().0.block_on(async {
            let connecting = servers.into_iter().map(|(name, server)| async move {
                let timeout = std::time::Duration::from_secs(20);
                tokio::time::timeout(timeout, connect(name.clone(), server))
                    .await
                    .unwrap_or_else(|_| Err(format!("{name}: timed out connecting")))
            });
            futures::future::join_all(connecting).await
        });
        let mut mcp = Mcp {
            running: Vec::new(),
            notices: Vec::new(),
        };
        let mut tools = app.world_mut().resource_mut::<Tools>();
        for result in results {
            match result {
                Ok((name, service, found)) => {
                    let names: Vec<String> =
                        found.iter().map(|tool| tool.name().to_owned()).collect();
                    for tool in found {
                        tools.0.insert(
                            tool.name().to_owned(),
                            Tool {
                                definition: tool.definition(),
                                run: Some(tool),
                            },
                        );
                    }
                    mcp.running.push(service);
                    mcp.notices.push(Entry::Notice(format!(
                        "MCP server {name}: {}",
                        names.join(", ")
                    )));
                }
                Err(error) => mcp
                    .notices
                    .push(Entry::Error(format!("MCP server {error}"))),
            }
        }
        app.insert_resource(mcp).add_systems(Update, announce);
    }
}

async fn connect(name: String, server: Server) -> Connected {
    let command = tokio::process::Command::new(&server.command).configure(|command| {
        command.args(&server.args);
    });
    let transport = TokioChildProcess::new(command).map_err(|error| format!("{name}: {error}"))?;
    let service = ().serve(transport).await.map_err(|error| format!("{name}: {error}"))?;
    let listed = service
        .list_all_tools()
        .await
        .map_err(|error| format!("{name}: {error}"))?;
    let tools = rig_rmcp::tools_from_server(listed, service.peer())
        .into_iter()
        .map(|tool| prefixed(&name, tool))
        .collect();
    Ok((name, service, tools))
}

/// The server's tool under `<server>__<tool>`, so servers cannot collide.
fn prefixed(server: &str, tool: McpTool) -> DynamicTool {
    let inner = DynamicTool::from(tool);
    let definition = inner.definition();
    DynamicTool::new(
        format!("{server}__{}", definition.name),
        definition.description,
        definition.parameters,
        move |arguments| {
            let inner = inner.clone();
            Box::pin(async move { inner.execute(arguments).await })
        },
    )
}

/// Tells the TUI session which servers connected.
fn announce(mut mcp: ResMut<Mcp>, mut agents: Query<(&Origin, &mut Transcript)>) {
    if mcp.notices.is_empty() {
        return;
    }
    let Some((_, mut transcript)) = agents
        .iter_mut()
        .find(|(origin, _)| **origin == Origin::Tui)
    else {
        return;
    };
    for notice in std::mem::take(&mut mcp.notices) {
        transcript.log(notice);
    }
}
