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

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

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

    fn name(&self) -> String {
        self.name.to_string()
    }

    async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
        crate::completion::ToolDefinition::from(self)
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
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

    fn name(&self) -> String {
        self.name.clone()
    }

    async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
        self.clone()
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(error_result(format!(
            "No local handler is registered for rmcp tool '{}'",
            self.name
        )))
    }
}

#[derive(Clone)]
struct RegisteredTool {
    definition: Arc<dyn RmcpToolDefinitionProvider>,
    provider: Arc<dyn RmcpToolProvider>,
}

trait RmcpToolDefinitionProvider: WasmCompatSend + WasmCompatSync {
    fn definition<'a>(
        &'a self,
        prompt: String,
    ) -> WasmBoxedFuture<'a, Result<Tool, ToolServerError>>;
}

struct StaticToolDefinitionProvider {
    definition: Tool,
}

impl RmcpToolDefinitionProvider for StaticToolDefinitionProvider {
    fn definition<'a>(
        &'a self,
        _prompt: String,
    ) -> WasmBoxedFuture<'a, Result<Tool, ToolServerError>> {
        Box::pin(async move { Ok(self.definition.clone()) })
    }
}

struct LocalToolDefinitionProvider<T> {
    tool: Arc<T>,
}

impl<T> RmcpToolDefinitionProvider for LocalToolDefinitionProvider<T>
where
    T: LocalRmcpTool,
{
    fn definition<'a>(
        &'a self,
        prompt: String,
    ) -> WasmBoxedFuture<'a, Result<Tool, ToolServerError>> {
        Box::pin(async move { Ok(self.tool.definition(prompt).await.into()) })
    }
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

struct LocalToolProvider<T> {
    tool: Arc<T>,
}

impl<T> RmcpToolProvider for LocalToolProvider<T>
where
    T: LocalRmcpTool,
{
    fn list_tools<'a>(&'a self) -> WasmBoxedFuture<'a, Result<Vec<Tool>, ToolServerError>> {
        Box::pin(async move {
            let definition = self.tool.definition(String::new()).await;
            Ok(vec![definition.into()])
        })
    }

    fn call_tool<'a>(
        &'a self,
        params: CallToolRequestParams,
    ) -> WasmBoxedFuture<'a, Result<CallToolResult, ToolServerError>> {
        Box::pin(async move {
            let args =
                serde_json::from_value::<T::Args>(params.arguments.unwrap_or_default().into())?;
            let output = self
                .tool
                .call(args)
                .await
                .map_err(|e| ToolServerError::from(e.to_string()))?;
            let value = serde_json::to_value(output)?;
            Ok(CallToolResult::structured(value))
        })
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

fn static_registered_tool(tool: Tool, provider: Arc<dyn RmcpToolProvider>) -> RegisteredTool {
    RegisteredTool {
        definition: Arc::new(StaticToolDefinitionProvider {
            definition: tool.clone(),
        }),
        provider,
    }
}

fn local_registered_tool<T>(tool: Arc<T>, provider: Arc<dyn RmcpToolProvider>) -> RegisteredTool
where
    T: LocalRmcpTool,
{
    RegisteredTool {
        definition: Arc::new(LocalToolDefinitionProvider { tool }),
        provider,
    }
}

fn push_unique_tool_name(names: &mut Vec<String>, name: String) {
    if !names.contains(&name) {
        names.push(name);
    }
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
        self.tools
            .insert(name.clone(), static_registered_tool(tool, provider));
        self.static_tool_names.push(name);
        self
    }

    pub fn local_rmcp_tool<T>(mut self, tool: T) -> Self
    where
        T: LocalRmcpTool,
    {
        let name = tool.name();
        let tool = Arc::new(tool);
        let provider = Arc::new(LocalToolProvider { tool: tool.clone() });
        self.tools
            .insert(name.clone(), local_registered_tool(tool, provider));
        self.static_tool_names.push(name);
        self
    }

    pub fn remote_rmcp_tools(
        mut self,
        tools: Vec<Tool>,
        sink: ::rmcp::service::ServerSink,
    ) -> Self {
        let provider = Arc::new(RemoteToolProvider::new(sink));
        for tool in tools {
            let name = tool.name.to_string();
            self.tools
                .insert(name.clone(), static_registered_tool(tool, provider.clone()));
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
        self.tools
            .insert(name, static_registered_tool(tool, provider));
    }

    pub fn add_local_tool<T>(&mut self, tool: T)
    where
        T: LocalRmcpTool,
    {
        let name = tool.name();
        let tool = Arc::new(tool);
        let provider = Arc::new(LocalToolProvider { tool: tool.clone() });
        self.tools
            .insert(name, local_registered_tool(tool, provider));
    }

    pub fn add_remote_tools(&mut self, tools: Vec<Tool>, sink: ::rmcp::service::ServerSink) {
        let provider = Arc::new(RemoteToolProvider::new(sink));
        for tool in tools {
            self.tools.insert(
                tool.name.to_string(),
                static_registered_tool(tool, provider.clone()),
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
        for name in registry.tools.keys() {
            push_unique_tool_name(&mut state.static_tool_names, name.clone());
        }
        state.tools.extend(registry.tools);
        Ok(())
    }

    pub async fn set_remote_tools(
        &self,
        tools: Vec<Tool>,
        sink: ::rmcp::service::ServerSink,
    ) -> Result<(), ToolServerError> {
        self.replace_remote_tools(Vec::new(), tools, sink)
            .await
            .map(|_| ())
    }

    pub async fn replace_remote_tools(
        &self,
        previous_tool_names: Vec<String>,
        tools: Vec<Tool>,
        sink: ::rmcp::service::ServerSink,
    ) -> Result<Vec<String>, ToolServerError> {
        self.replace_remote_tools_with_provider(
            previous_tool_names,
            tools,
            Arc::new(RemoteToolProvider::new(sink)),
        )
        .await
    }

    async fn replace_remote_tools_with_provider(
        &self,
        previous_tool_names: Vec<String>,
        tools: Vec<Tool>,
        provider: Arc<dyn RmcpToolProvider>,
    ) -> Result<Vec<String>, ToolServerError> {
        let mut state = self.0.write().await;
        for name in previous_tool_names {
            state.static_tool_names.retain(|x| x != &name);
            state.tools.remove(&name);
        }

        let mut current_tool_names = Vec::new();
        for tool in tools {
            let name = tool.name.to_string();
            push_unique_tool_name(&mut state.static_tool_names, name.clone());
            push_unique_tool_name(&mut current_tool_names, name.clone());
            state
                .tools
                .insert(name, static_registered_tool(tool, provider.clone()));
        }
        Ok(current_tool_names)
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
            return Err(ToolServerError::ToolResultError(call_tool_result_to_text(
                &result,
            )?));
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

        let definition_providers = {
            let state = self.0.read().await;
            tool_names
                .iter()
                .filter_map(|name| {
                    let tool = state.tools.get(name);
                    if tool.is_none() {
                        tracing::warn!("Tool implementation not found in registry: {}", name);
                    }
                    tool.map(|tool| tool.definition.clone())
                })
                .collect::<Vec<_>>()
        };

        let prompt = prompt.unwrap_or_default();
        let definitions = definition_providers.into_iter().map(|provider| {
            let prompt = prompt.clone();
            async move { provider.definition(prompt).await }
        });
        futures::future::try_join_all(definitions).await
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

    #[error("MCP tool returned error: {0}")]
    ToolResultError(String),
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

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use serde::{Deserialize, Serialize};

    use super::*;

    #[tokio::test]
    async fn add_rmcp_tool_advertises_runtime_tool_definition() {
        let handle = ToolServer::new().run();
        let tool = crate::tool::tool_from_schema(
            "runtime_tool",
            "Runtime registered tool",
            serde_json::json!({
                "type": "object",
                "properties": {},
            }),
        );

        handle
            .add_rmcp_tool(tool, |_params| async {
                Ok(CallToolResult::success(vec![Content::text("ok")]))
            })
            .await
            .expect("tool should register");

        let defs = handle
            .get_tool_defs(None)
            .await
            .expect("tool definitions should load");
        let names = defs
            .iter()
            .map(|tool| tool.name.as_ref())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["runtime_tool"]);
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct EchoTool;

    #[derive(Debug, Serialize, Deserialize)]
    struct EchoArgs {
        message: String,
    }

    impl LocalRmcpTool for EchoTool {
        const NAME: &'static str = "echo";

        type Error = ToolError;
        type Args = EchoArgs;
        type Output = String;

        async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
            crate::completion::ToolDefinition {
                name: Self::NAME.to_string(),
                description: "Echo a message".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    },
                    "required": ["message"]
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(args.message)
        }
    }

    #[tokio::test]
    async fn add_local_rmcp_tool_advertises_runtime_tool_definition() {
        let handle = ToolServer::new().run();

        handle
            .add_local_rmcp_tool(EchoTool)
            .await
            .expect("tool should register");

        let defs = handle
            .get_tool_defs(None)
            .await
            .expect("tool definitions should load");
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");

        let result = handle
            .call_tool_text(
                CallToolRequestParams::new("echo").with_arguments(
                    serde_json::json!({ "message": "hello" })
                        .as_object()
                        .cloned()
                        .expect("object"),
                ),
            )
            .await
            .expect("tool call should succeed");
        assert_eq!(result, "hello");
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct PromptAwareTool;

    impl LocalRmcpTool for PromptAwareTool {
        const NAME: &'static str = "prompt_aware";

        type Error = ToolError;
        type Args = EchoArgs;
        type Output = String;

        async fn definition(&self, prompt: String) -> crate::completion::ToolDefinition {
            crate::completion::ToolDefinition {
                name: Self::NAME.to_string(),
                description: format!("Definition for prompt: {prompt}"),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    },
                    "required": ["message"]
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(args.message)
        }
    }

    #[tokio::test]
    async fn local_rmcp_tool_definition_uses_request_prompt() {
        let handle = ToolServer::new().local_rmcp_tool(PromptAwareTool).run();

        let defs = handle
            .get_tool_defs(Some("current user request".to_string()))
            .await
            .expect("tool definitions should load");

        assert_eq!(defs.len(), 1);
        assert_eq!(
            defs[0].description.as_deref(),
            Some("Definition for prompt: current user request")
        );
    }

    #[derive(Clone)]
    struct CountingDefinitionTool {
        calls: Arc<AtomicUsize>,
    }

    impl LocalRmcpTool for CountingDefinitionTool {
        const NAME: &'static str = "counting_definition";

        type Error = ToolError;
        type Args = EchoArgs;
        type Output = String;

        async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
            self.calls.fetch_add(1, Ordering::SeqCst);
            crate::completion::ToolDefinition {
                name: Self::NAME.to_string(),
                description: "Count definition calls".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    },
                    "required": ["message"]
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(args.message)
        }
    }

    #[tokio::test]
    async fn local_rmcp_tool_registration_does_not_resolve_definition() {
        let calls = Arc::new(AtomicUsize::new(0));
        let handle = ToolServer::new()
            .local_rmcp_tool(CountingDefinitionTool {
                calls: calls.clone(),
            })
            .run();

        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let defs = handle
            .get_tool_defs(None)
            .await
            .expect("tool definitions should load");

        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "counting_definition");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn call_tool_text_returns_error_for_mcp_error_result() {
        let handle = ToolServer::new()
            .rmcp_tool(
                crate::tool::tool_from_schema(
                    "failing_tool",
                    "Fails",
                    serde_json::json!({
                        "type": "object",
                        "properties": {},
                    }),
                ),
                |_params| async { Ok(error_result("remote failure")) },
            )
            .run();

        let error = handle
            .call_tool_text(CallToolRequestParams::new("failing_tool"))
            .await
            .expect_err("MCP error results should be returned as errors");

        assert!(matches!(
            error,
            ToolServerError::ToolResultError(message) if message == "remote failure"
        ));
    }

    fn test_tool(name: &str) -> Tool {
        crate::tool::tool_from_schema(
            name.to_string(),
            format!("{name} description"),
            serde_json::json!({
                "type": "object",
                "properties": {},
            }),
        )
    }

    #[tokio::test]
    async fn set_remote_tools_replaces_previous_remote_tools() {
        let handle = ToolServer::new().local_rmcp_tool(EchoTool).run();
        let provider = Arc::new(ClosureToolProvider {
            definition: test_tool("remote_provider"),
            handler: |_params| async { Ok(CallToolResult::success(vec![Content::text("ok")])) },
        });

        let managed = handle
            .replace_remote_tools_with_provider(
                Vec::new(),
                vec![test_tool("remote_a"), test_tool("remote_b")],
                provider.clone(),
            )
            .await
            .expect("initial remote tools should register");

        let names = handle
            .get_tool_defs(None)
            .await
            .expect("tool definitions should load")
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();
        assert!(names.contains(&"echo".to_string()));
        assert!(names.contains(&"remote_a".to_string()));
        assert!(names.contains(&"remote_b".to_string()));

        handle
            .replace_remote_tools_with_provider(managed, vec![test_tool("remote_c")], provider)
            .await
            .expect("remote tools should replace");

        let names = handle
            .get_tool_defs(None)
            .await
            .expect("tool definitions should load")
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();

        assert!(names.contains(&"echo".to_string()));
        assert_eq!(
            names
                .iter()
                .filter(|name| name.starts_with("remote_"))
                .count(),
            1
        );
        assert!(names.contains(&"remote_c".to_string()));
        assert!(!names.contains(&"remote_a".to_string()));
        assert!(!names.contains(&"remote_b".to_string()));
    }

    #[tokio::test]
    async fn remote_tool_replacement_only_removes_tools_owned_by_caller() {
        let handle = ToolServer::new().local_rmcp_tool(EchoTool).run();
        let provider_a = Arc::new(ClosureToolProvider {
            definition: test_tool("provider_a"),
            handler: |_params| async { Ok(CallToolResult::success(vec![Content::text("a")])) },
        });
        let provider_b = Arc::new(ClosureToolProvider {
            definition: test_tool("provider_b"),
            handler: |_params| async { Ok(CallToolResult::success(vec![Content::text("b")])) },
        });

        let managed_a = handle
            .replace_remote_tools_with_provider(
                Vec::new(),
                vec![test_tool("remote_a")],
                provider_a.clone(),
            )
            .await
            .expect("first remote server should register");
        let managed_b = handle
            .replace_remote_tools_with_provider(
                Vec::new(),
                vec![test_tool("remote_b")],
                provider_b.clone(),
            )
            .await
            .expect("second remote server should register");

        let managed_a = handle
            .replace_remote_tools_with_provider(managed_a, vec![test_tool("remote_a2")], provider_a)
            .await
            .expect("first remote server should refresh");

        let names = handle
            .get_tool_defs(None)
            .await
            .expect("tool definitions should load")
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();

        assert!(names.contains(&"echo".to_string()));
        assert!(names.contains(&"remote_a2".to_string()));
        assert!(names.contains(&"remote_b".to_string()));
        assert!(!names.contains(&"remote_a".to_string()));
        assert_eq!(managed_a, vec!["remote_a2".to_string()]);
        assert_eq!(managed_b, vec!["remote_b".to_string()]);
    }

    #[test]
    fn push_unique_tool_name_deduplicates_existing_names() {
        let mut names = vec!["search".to_string()];

        push_unique_tool_name(&mut names, "search".to_string());
        push_unique_tool_name(&mut names, "lookup".to_string());

        assert_eq!(names, vec!["search", "lookup"]);
    }
}
