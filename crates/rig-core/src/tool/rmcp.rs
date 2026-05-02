//! MCP client integration for the rmcp-native tool registry.

use std::sync::Arc;

use ::rmcp::ServiceExt;
use tokio::sync::RwLock;

use crate::tool::server::{RemoteToolRegistration, ToolServerError, ToolServerHandle};

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
    managed_tools: Arc<RwLock<Option<RemoteToolRegistration>>>,
}

impl McpClientHandler {
    pub fn new(
        client_info: ::rmcp::model::ClientInfo,
        tool_server_handle: ToolServerHandle,
    ) -> Self {
        Self {
            client_info,
            tool_server_handle,
            managed_tools: Arc::new(RwLock::new(None)),
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
        {
            let handler = service.service();
            let mut managed = handler.managed_tools.write().await;
            let registration = handler
                .tool_server_handle
                .replace_remote_tools(managed.clone(), tools, service.peer().clone())
                .await?;
            *managed = Some(registration);
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

        let mut managed = self.managed_tools.write().await;
        let registration = match self
            .tool_server_handle
            .replace_remote_tools(managed.clone(), tools, context.peer.clone())
            .await
        {
            Ok(registration) => registration,
            Err(e) => {
                tracing::error!("Failed to refresh MCP tools: {e}");
                return;
            }
        };
        *managed = Some(registration);
    }
}
