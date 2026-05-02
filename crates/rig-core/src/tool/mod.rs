//! Rmcp-native tool orchestration.
//!
//! Rig uses `::rmcp::model::Tool` as the canonical tool definition,
//! `::rmcp::model::CallToolRequestParams` as the canonical invocation request,
//! and `::rmcp::model::CallToolResult` as the canonical execution result.

pub mod server;

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
pub mod rmcp;

pub use ::rmcp::model::Tool;
use ::rmcp::model::{CallToolResult, Content, RawContent, ResourceContents};
use serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// Could not find a tool.
    #[error("ToolNotFoundError: {0}")]
    ToolNotFoundError(String),

    /// Error returned while executing a tool.
    #[error("ToolCallError: {0}")]
    ToolCallError(String),

    /// JSON serialization or deserialization failed.
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Tool call was interrupted. Primarily useful for agent multi-step/turn prompting.
    #[error("Tool call interrupted")]
    Interrupted,
}

/// Convert an rmcp tool definition into Rig's provider-boundary function schema.
pub fn tool_to_function_schema(tool: &Tool) -> crate::completion::ToolDefinition {
    crate::completion::ToolDefinition {
        name: tool.name.to_string(),
        description: tool.description.clone().unwrap_or_default().to_string(),
        parameters: tool.schema_as_json_value(),
    }
}

/// Build an rmcp tool definition from the provider-boundary function schema shape.
pub fn tool_from_schema(
    name: impl Into<std::borrow::Cow<'static, str>>,
    description: impl Into<std::borrow::Cow<'static, str>>,
    parameters: serde_json::Value,
) -> Tool {
    Tool::new(
        name,
        description,
        match parameters {
            serde_json::Value::Object(map) => map,
            _ => serde_json::Map::new(),
        },
    )
}

impl From<crate::completion::ToolDefinition> for Tool {
    fn from(definition: crate::completion::ToolDefinition) -> Self {
        tool_from_schema(
            definition.name,
            definition.description,
            definition.parameters,
        )
    }
}

/// Convert an rmcp tool result to the text fallback used by providers that do not
/// accept richer MCP result content directly.
pub fn call_tool_result_to_text(result: &CallToolResult) -> Result<String, ToolError> {
    if let Some(value) = &result.structured_content {
        return Ok(value_to_model_text(value));
    }

    let mut text = String::new();
    for item in &result.content {
        text.push_str(&content_to_text(item)?);
    }
    Ok(text)
}

fn value_to_model_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        value => value.to_string(),
    }
}

fn content_to_text(content: &Content) -> Result<String, ToolError> {
    match &content.raw {
        RawContent::Text(raw) => Ok(raw.text.clone()),
        RawContent::Image(raw) => Ok(format!("data:{};base64,{}", raw.mime_type, raw.data)),
        RawContent::Resource(raw) => match &raw.resource {
            ResourceContents::TextResourceContents {
                uri,
                mime_type,
                text,
                ..
            } => Ok(format!(
                "{mime_type}{uri}:{text}",
                mime_type = mime_type
                    .as_ref()
                    .map(|m| format!("data:{m};"))
                    .unwrap_or_default(),
            )),
            ResourceContents::BlobResourceContents {
                uri,
                mime_type,
                blob,
                ..
            } => Ok(format!(
                "{mime_type}{uri}:{blob}",
                mime_type = mime_type
                    .as_ref()
                    .map(|m| format!("data:{m};"))
                    .unwrap_or_default(),
            )),
        },
        RawContent::Audio(_) => Err(ToolError::ToolCallError(
            "MCP tool returned audio content, which Rig does not support yet".to_string(),
        )),
        other => Err(ToolError::ToolCallError(format!(
            "MCP tool returned unsupported content: {other:?}"
        ))),
    }
}
