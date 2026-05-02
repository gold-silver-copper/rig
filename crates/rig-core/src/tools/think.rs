use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
pub struct ThinkArgs {
    pub thought: String,
}

#[derive(Deserialize, Serialize)]
pub struct ThinkTool;

impl ThinkTool {
    pub fn definition() -> ::rmcp::model::Tool {
        ::rmcp::model::Tool::new(
            "think",
            "Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.",
            match json!({
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "A thought to think about."
                    }
                },
                "required": ["thought"]
            }) {
                serde_json::Value::Object(map) => map,
                _ => serde_json::Map::new(),
            },
        )
    }

    pub async fn call(
        params: ::rmcp::model::CallToolRequestParams,
    ) -> Result<::rmcp::model::CallToolResult, crate::tool::server::ToolServerError> {
        let args: ThinkArgs = serde_json::from_value(serde_json::Value::Object(
            params.arguments.unwrap_or_default(),
        ))?;
        Ok(::rmcp::model::CallToolResult::success(vec![
            ::rmcp::model::Content::text(args.thought),
        ]))
    }
}
