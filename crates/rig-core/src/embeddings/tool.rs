//! Embeddable metadata for rmcp tool definitions.

use crate::Embed;
use serde::Serialize;

use super::embed::EmbedError;

#[derive(Clone, Serialize, Default, Eq, PartialEq)]
pub struct ToolSchema {
    pub name: String,
    pub context: serde_json::Value,
    pub embedding_docs: Vec<String>,
}

impl Embed for ToolSchema {
    fn embed(&self, embedder: &mut super::embed::TextEmbedder) -> Result<(), EmbedError> {
        for doc in &self.embedding_docs {
            embedder.embed(doc.clone());
        }
        Ok(())
    }
}

impl ToolSchema {
    pub fn from_rmcp_tool(tool: &::rmcp::model::Tool) -> Self {
        let schema = crate::tool::tool_to_function_schema(tool);
        Self {
            name: schema.name.clone(),
            context: serde_json::json!({ "tool": schema }),
            embedding_docs: vec![format!(
                "Tool: {}\nDefinition:\n{}",
                schema.name,
                serde_json::to_string_pretty(&schema).unwrap_or_else(|_| schema.name.clone())
            )],
        }
    }
}
