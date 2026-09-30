//! A small MCP server over stdio, for trying the MCP plugin:
//!
//! ```sh
//! cargo build --example mcp_server
//! echo '{"servers": {"words": {"command": "target/debug/examples/mcp_server"}}}' > mcp.json
//! ```

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ErrorData, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct Text {
    /// The text to transform.
    text: String,
}

#[derive(Clone)]
struct Words {
    tool_router: ToolRouter<Words>,
}

#[tool_router]
impl Words {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Encode text with ROT13.")]
    fn rot13(&self, Parameters(Text { text }): Parameters<Text>) -> Result<CallToolResult, ErrorData> {
        let encoded: String = text
            .chars()
            .map(|c| match c {
                'a'..='z' => (((c as u8 - b'a' + 13) % 26) + b'a') as char,
                'A'..='Z' => (((c as u8 - b'A' + 13) % 26) + b'A') as char,
                other => other,
            })
            .collect();
        Ok(CallToolResult::success(vec![ContentBlock::text(encoded)]))
    }

    #[tool(description = "Count the words in text.")]
    fn word_count(&self, Parameters(Text { text }): Parameters<Text>) -> Result<CallToolResult, ErrorData> {
        Ok(CallToolResult::success(vec![ContentBlock::text(
            text.split_whitespace().count().to_string(),
        )]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Words {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = Words::new().serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
