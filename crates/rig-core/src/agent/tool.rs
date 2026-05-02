use std::sync::Arc;

use ::rmcp::model::{CallToolRequestParams, CallToolResult, Content, Tool};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    agent::Agent,
    completion::{CompletionModel, Prompt},
    tool::server::{LocalRmcpTool, ToolServerError},
};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AgentToolArgs {
    /// The prompt for the agent to call.
    prompt: String,
}

impl<M> Agent<M>
where
    M: CompletionModel + 'static,
{
    pub fn rmcp_tool_definition(&self) -> Tool {
        let description = format!(
            "
            Prompt a sub-agent to do a task for you.

            Agent name: {name}
            Agent description: {description}
            Agent system prompt: {sysprompt}
            ",
            name = self.name(),
            description = self.description.clone().unwrap_or_default(),
            sysprompt = self.preamble.clone().unwrap_or_default()
        );

        crate::tool::tool_from_schema(
            self.name
                .clone()
                .unwrap_or_else(|| "agent_tool".to_string()),
            description,
            json!(schema_for!(AgentToolArgs)),
        )
    }

    pub fn into_rmcp_tool_handler(
        self,
    ) -> (
        Tool,
        impl Fn(
            CallToolRequestParams,
        )
            -> crate::wasm_compat::WasmBoxedFuture<'static, Result<CallToolResult, ToolServerError>>
        + Clone
        + crate::wasm_compat::WasmCompatSend
        + crate::wasm_compat::WasmCompatSync
        + 'static,
    ) {
        let tool = self.rmcp_tool_definition();
        let agent = Arc::new(self);
        let handler = move |params: CallToolRequestParams| {
            let agent = agent.clone();
            Box::pin(async move {
                let args = serde_json::from_value::<AgentToolArgs>(
                    params.arguments.unwrap_or_default().into(),
                )?;
                let output = agent
                    .prompt(args.prompt)
                    .await
                    .map_err(|e| ToolServerError::from(format!("Agent tool prompt failed: {e}")))?;
                Ok(CallToolResult::success(vec![Content::text(output)]))
            })
                as crate::wasm_compat::WasmBoxedFuture<
                    'static,
                    Result<CallToolResult, ToolServerError>,
                >
        };

        (tool, handler)
    }
}

impl<M> LocalRmcpTool for Agent<M>
where
    M: CompletionModel + 'static,
{
    const NAME: &'static str = "agent_tool";

    type Error = crate::completion::PromptError;
    type Args = AgentToolArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
        crate::completion::ToolDefinition::from(self.rmcp_tool_definition())
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.prompt(args.prompt).await
    }
}
