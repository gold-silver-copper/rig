//! MCP client integration for the rmcp-native tool registry.

use std::sync::Arc;

use ::rmcp::ServiceExt;
use tokio::sync::RwLock;

use crate::tool::server::{ToolServerError, ToolServerHandle};

#[derive(Debug, thiserror::Error)]
pub enum McpClientError {
    #[error("MCP connection error: {0}")]
    ConnectionError(String),

    #[error("Failed to fetch MCP tool list: {0}")]
    ToolFetchError(#[from] ::rmcp::ServiceError),

    #[error("Tool server error: {0}")]
    ToolServerError(#[from] ToolServerError),
}

/// MCP client handler that refreshes an rmcp-native Rig tool registry when a
/// remote server sends `notifications/tools/list_changed`.
pub struct McpClientHandler {
    client_info: ::rmcp::model::ClientInfo,
    tool_server_handle: ToolServerHandle,
    managed_tool_names: Arc<RwLock<Vec<String>>>,
}

impl McpClientHandler {
    pub fn new(
        client_info: ::rmcp::model::ClientInfo,
        tool_server_handle: ToolServerHandle,
    ) -> Self {
        Self {
            client_info,
            tool_server_handle,
            managed_tool_names: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn connect<T, E, A>(
        self,
        transport: T,
    ) -> Result<::rmcp::service::RunningService<::rmcp::service::RoleClient, Self>, McpClientError>
    where
        T: ::rmcp::transport::IntoTransport<::rmcp::service::RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let service = ServiceExt::serve(self, transport)
            .await
            .map_err(|e| McpClientError::ConnectionError(e.to_string()))?;

        let tools = service.peer().list_all_tools().await?;
        let tool_names = tools
            .iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();

        {
            let handler = service.service();
            handler
                .tool_server_handle
                .set_remote_tools(tools, service.peer().clone())
                .await?;
            *handler.managed_tool_names.write().await = tool_names;
        }

        Ok(service)
    }
}

impl ::rmcp::handler::client::ClientHandler for McpClientHandler {
    fn get_info(&self) -> ::rmcp::model::ClientInfo {
        self.client_info.clone()
    }

    async fn on_tool_list_changed(
        &self,
        context: ::rmcp::service::NotificationContext<::rmcp::service::RoleClient>,
    ) {
        let tools = match context.peer.list_all_tools().await {
            Ok(tools) => tools,
            Err(e) => {
                tracing::error!("Failed to re-fetch MCP tool list: {e}");
                return;
            }
        };

        let mut managed = self.managed_tool_names.write().await;
        for name in managed.drain(..) {
            if let Err(e) = self.tool_server_handle.remove_tool(&name).await {
                tracing::warn!("Failed to remove MCP tool '{name}' during refresh: {e}");
            }
        }

        let tool_names = tools
            .iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();
        if let Err(e) = self
            .tool_server_handle
            .set_remote_tools(tools, context.peer.clone())
            .await
        {
            tracing::error!("Failed to refresh MCP tools: {e}");
            return;
        }
        *managed = tool_names;
    }
}
