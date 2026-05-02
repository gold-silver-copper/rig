use std::{collections::HashMap, future::Future, sync::Arc};

use ::rmcp::model::{CallToolRequestParams, CallToolResult, Content, Tool};
use tokio::sync::RwLock;

use crate::{
    completion::CompletionError,
    tool::{ToolError, call_tool_result_to_text},
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndexDyn, request::Filter},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

pub trait RmcpToolProvider: WasmCompatSend + WasmCompatSync {
    fn list_tools<'a>(&'a self) -> WasmBoxedFuture<'a, Result<Vec<Tool>, ToolServerError>>;

    fn call_tool<'a>(
        &'a self,
        params: CallToolRequestParams,
    ) -> WasmBoxedFuture<'a, Result<CallToolResult, ToolServerError>>;
}

pub trait LocalRmcpTool: WasmCompatSend + WasmCompatSync + 'static {
    const NAME: &'static str;

    type Error: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;
    type Args: serde::de::DeserializeOwned + WasmCompatSend;
    type Output: serde::Serialize + WasmCompatSend;

    fn definition(
        &self,
        prompt: String,
    ) -> impl Future<Output = crate::completion::ToolDefinition> + WasmCompatSend;

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + WasmCompatSend;
}

impl LocalRmcpTool for Tool {
    const NAME: &'static str = "";

    type Error = ToolError;
    type Args = serde_json::Value;
    type Output = CallToolResult;

    async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
        crate::completion::ToolDefinition::from(self)
    }

    async fn call(
        &self,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(error_result(format!(
            "No local handler is registered for rmcp tool '{}'",
            self.name
        )))
    }
}

impl LocalRmcpTool for crate::completion::ToolDefinition {
    const NAME: &'static str = "";

    type Error = ToolError;
    type Args = serde_json::Value;
    type Output = CallToolResult;

    async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
        self.clone()
    }

    async fn call(
        &self,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(error_result(format!(
            "No local handler is registered for rmcp tool '{}'",
            self.name
        )))
    }
}

#[derive(Clone)]
struct RegisteredTool {
    definition: Tool,
    provider: Arc<dyn RmcpToolProvider>,
}

struct ClosureToolProvider<F> {
    definition: Tool,
    handler: F,
}

impl<F, Fut> RmcpToolProvider for ClosureToolProvider<F>
where
    F: Fn(CallToolRequestParams) -> Fut + WasmCompatSend + WasmCompatSync,
    Fut: Future<Output = Result<CallToolResult, ToolServerError>> + WasmCompatSend + 'static,
{
    fn list_tools<'a>(&'a self) -> WasmBoxedFuture<'a, Result<Vec<Tool>, ToolServerError>> {
        Box::pin(async move { Ok(vec![self.definition.clone()]) })
    }

    fn call_tool<'a>(
        &'a self,
        params: CallToolRequestParams,
    ) -> WasmBoxedFuture<'a, Result<CallToolResult, ToolServerError>> {
        Box::pin((self.handler)(params))
    }
}

#[derive(Clone)]
pub struct RemoteToolProvider {
    sink: ::rmcp::service::ServerSink,
}

impl RemoteToolProvider {
    pub fn new(sink: ::rmcp::service::ServerSink) -> Self {
        Self { sink }
    }
}

impl RmcpToolProvider for RemoteToolProvider {
    fn list_tools<'a>(&'a self) -> WasmBoxedFuture<'a, Result<Vec<Tool>, ToolServerError>> {
        Box::pin(async move {
            let result = self.sink.list_tools(Default::default()).await?;
            Ok(result.tools)
        })
    }

    fn call_tool<'a>(
        &'a self,
        params: CallToolRequestParams,
    ) -> WasmBoxedFuture<'a, Result<CallToolResult, ToolServerError>> {
        Box::pin(async move { self.sink.call_tool(params).await.map_err(Into::into) })
    }
}

struct ToolServerState {
    static_tool_names: Vec<String>,
    dynamic_tools: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
    tools: HashMap<String, RegisteredTool>,
}

pub struct ToolServer {
    static_tool_names: Vec<String>,
    dynamic_tools: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
    tools: HashMap<String, RegisteredTool>,
}

impl Default for ToolServer {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolServer {
    pub fn new() -> Self {
        Self {
            static_tool_names: Vec::new(),
            dynamic_tools: Vec::new(),
            tools: HashMap::new(),
        }
    }

    pub(crate) fn static_tool_names(mut self, names: Vec<String>) -> Self {
        self.static_tool_names = names;
        self
    }

    pub(crate) fn add_registry(mut self, registry: RmcpToolRegistry) -> Self {
        self.tools.extend(registry.tools);
        self
    }

    pub(crate) fn add_dynamic_tools(
        mut self,
        dyn_tools: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
    ) -> Self {
        self.dynamic_tools = dyn_tools;
        self
    }

    pub fn rmcp_tool<F, Fut>(mut self, tool: Tool, handler: F) -> Self
    where
        F: Fn(CallToolRequestParams) -> Fut + WasmCompatSend + WasmCompatSync + 'static,
        Fut: Future<Output = Result<CallToolResult, ToolServerError>> + WasmCompatSend + 'static,
    {
        let name = tool.name.to_string();
        let provider = Arc::new(ClosureToolProvider {
            definition: tool.clone(),
            handler,
        });
        self.tools.insert(
            name.clone(),
            RegisteredTool {
                definition: tool,
                provider,
            },
        );
        self.static_tool_names.push(name);
        self
    }

    pub fn local_rmcp_tool<T>(self, tool: T) -> Self
    where
        T: LocalRmcpTool,
    {
        let definition = futures::executor::block_on(tool.definition(String::new()));
        let tool = Arc::new(tool);
        self.rmcp_tool(definition.into(), move |params| {
            let tool = tool.clone();
            async move {
                let args = serde_json::from_value::<T::Args>(
                    params.arguments.unwrap_or_default().into(),
                )?;
                let output = tool
                    .call(args)
                    .await
                    .map_err(|e| ToolServerError::from(e.to_string()))?;
                let value = serde_json::to_value(output)?;
                Ok(CallToolResult::structured(value))
            }
        })
    }

    pub fn remote_rmcp_tools(
        mut self,
        tools: Vec<Tool>,
        sink: ::rmcp::service::ServerSink,
    ) -> Self {
        let provider = Arc::new(RemoteToolProvider::new(sink));
        for tool in tools {
            let name = tool.name.to_string();
            self.tools.insert(
                name.clone(),
                RegisteredTool {
                    definition: tool,
                    provider: provider.clone(),
                },
            );
            self.static_tool_names.push(name);
        }
        self
    }

    pub fn dynamic_tools(
        mut self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + Send + Sync + 'static,
        registry: RmcpToolRegistry,
    ) -> Self {
        self.dynamic_tools.push((sample, Arc::new(dynamic_tools)));
        self.tools.extend(registry.tools);
        self
    }

    pub fn run(self) -> ToolServerHandle {
        ToolServerHandle(Arc::new(RwLock::new(ToolServerState {
            static_tool_names: self.static_tool_names,
            dynamic_tools: self.dynamic_tools,
            tools: self.tools,
        })))
    }
}

#[derive(Default)]
pub struct RmcpToolRegistry {
    tools: HashMap<String, RegisteredTool>,
}

impl RmcpToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_tool<F, Fut>(&mut self, tool: Tool, handler: F)
    where
        F: Fn(CallToolRequestParams) -> Fut + WasmCompatSend + WasmCompatSync + 'static,
        Fut: Future<Output = Result<CallToolResult, ToolServerError>> + WasmCompatSend + 'static,
    {
        let name = tool.name.to_string();
        let provider = Arc::new(ClosureToolProvider {
            definition: tool.clone(),
            handler,
        });
        self.tools.insert(
            name,
            RegisteredTool {
                definition: tool,
                provider,
            },
        );
    }

    pub fn add_local_tool<T>(&mut self, tool: T)
    where
        T: LocalRmcpTool,
    {
        let definition = futures::executor::block_on(tool.definition(String::new()));
        let tool = Arc::new(tool);
        self.add_tool(definition.into(), move |params| {
            let tool = tool.clone();
            async move {
                let args = serde_json::from_value::<T::Args>(
                    params.arguments.unwrap_or_default().into(),
                )?;
                let output = tool
                    .call(args)
                    .await
                    .map_err(|e| ToolServerError::from(e.to_string()))?;
                let value = serde_json::to_value(output)?;
                Ok(CallToolResult::structured(value))
            }
        });
    }

    pub fn add_remote_tools(&mut self, tools: Vec<Tool>, sink: ::rmcp::service::ServerSink) {
        let provider = Arc::new(RemoteToolProvider::new(sink));
        for tool in tools {
            self.tools.insert(
                tool.name.to_string(),
                RegisteredTool {
                    definition: tool,
                    provider: provider.clone(),
                },
            );
        }
    }

    pub fn extend(&mut self, registry: RmcpToolRegistry) {
        self.tools.extend(registry.tools);
    }
}

#[derive(Clone)]
pub struct ToolServerHandle(Arc<RwLock<ToolServerState>>);

impl ToolServerHandle {
    pub async fn add_rmcp_tool<F, Fut>(&self, tool: Tool, handler: F) -> Result<(), ToolServerError>
    where
        F: Fn(CallToolRequestParams) -> Fut + WasmCompatSend + WasmCompatSync + 'static,
        Fut: Future<Output = Result<CallToolResult, ToolServerError>> + WasmCompatSend + 'static,
    {
        let mut registry = RmcpToolRegistry::new();
        registry.add_tool(tool, handler);
        self.append_registry(registry).await
    }

    pub async fn add_local_rmcp_tool<T>(&self, tool: T) -> Result<(), ToolServerError>
    where
        T: LocalRmcpTool,
    {
        let mut registry = RmcpToolRegistry::new();
        registry.add_local_tool(tool);
        self.append_registry(registry).await
    }

    pub async fn append_registry(&self, registry: RmcpToolRegistry) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        state.tools.extend(registry.tools);
        Ok(())
    }

    pub async fn set_remote_tools(
        &self,
        tools: Vec<Tool>,
        sink: ::rmcp::service::ServerSink,
    ) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        let provider = Arc::new(RemoteToolProvider::new(sink));
        for tool in tools {
            let name = tool.name.to_string();
            state.static_tool_names.push(name.clone());
            state.tools.insert(
                name,
                RegisteredTool {
                    definition: tool,
                    provider: provider.clone(),
                },
            );
        }
        Ok(())
    }

    pub async fn remove_tool(&self, tool_name: &str) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        state.static_tool_names.retain(|x| x != tool_name);
        state.tools.remove(tool_name);
        Ok(())
    }

    pub async fn call_tool(
        &self,
        params: CallToolRequestParams,
    ) -> Result<CallToolResult, ToolServerError> {
        let tool_name = params.name.to_string();
        let provider = {
            let state = self.0.read().await;
            state
                .tools
                .get(&tool_name)
                .map(|tool| tool.provider.clone())
        };

        match provider {
            Some(provider) => provider.call_tool(params).await,
            None => Err(ToolServerError::ToolsetError(ToolError::ToolNotFoundError(
                tool_name,
            ))),
        }
    }

    pub async fn call_tool_text(
        &self,
        params: CallToolRequestParams,
    ) -> Result<String, ToolServerError> {
        let result = self.call_tool(params).await?;
        if result.is_error == Some(true) {
            return Ok(call_tool_result_to_text(&result)?);
        }
        Ok(call_tool_result_to_text(&result)?)
    }

    pub async fn get_tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<Tool>, ToolServerError> {
        let (static_tool_names, dynamic_tools) = {
            let state = self.0.read().await;
            (state.static_tool_names.clone(), state.dynamic_tools.clone())
        };

        let mut tool_names = Vec::new();
        if let Some(ref text) = prompt {
            let search_futures = dynamic_tools.iter().map(|(num_sample, index)| {
                let text = text.clone();
                let num_sample = *num_sample;
                let index = index.clone();

                async move {
                    let req = VectorSearchRequest::builder()
                        .query(text)
                        .samples(num_sample as u64)
                        .build();

                    let ids = index
                        .as_ref()
                        .top_n_ids(req.map_filter(Filter::interpret))
                        .await?
                        .into_iter()
                        .map(|(_, id)| id)
                        .collect::<Vec<String>>();

                    Ok::<_, VectorStoreError>(ids)
                }
            });

            tool_names.extend(
                futures::future::try_join_all(search_futures)
                    .await
                    .map_err(|e| {
                        ToolServerError::DefinitionError(CompletionError::RequestError(Box::new(e)))
                    })?
                    .into_iter()
                    .flatten(),
            );
        }

        tool_names.extend(static_tool_names);

        let state = self.0.read().await;
        Ok(tool_names
            .iter()
            .filter_map(|name| {
                let tool = state.tools.get(name);
                if tool.is_none() {
                    tracing::warn!("Tool implementation not found in registry: {}", name);
                }
                tool.map(|tool| tool.definition.clone())
            })
            .collect())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolServerError {
    #[error("Tool error: {0}")]
    ToolsetError(#[from] ToolError),

    #[error("Definition error: {0}")]
    DefinitionError(#[from] CompletionError),

    #[error("Remote MCP tool error: {0}")]
    RemoteMcp(#[from] ::rmcp::ServiceError),
}

impl From<String> for ToolServerError {
    fn from(error: String) -> Self {
        ToolError::ToolCallError(error).into()
    }
}

impl From<&str> for ToolServerError {
    fn from(error: &str) -> Self {
        error.to_string().into()
    }
}

impl From<serde_json::Error> for ToolServerError {
    fn from(error: serde_json::Error) -> Self {
        ToolError::JsonError(error).into()
    }
}

pub fn error_result(message: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![Content::text(message.into())])
}
