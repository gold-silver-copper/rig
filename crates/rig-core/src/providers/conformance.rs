use crate::{
    OneOrMany,
    completion::{CompletionRequest, ToolDefinition},
    message::Message,
};

pub(crate) const DEFAULT_MODEL: &str = "default-model";
pub(crate) const REQUEST_MODEL: &str = "request-model";
pub(crate) const SYSTEM_TEXT: &str = "Use concise answers.";
pub(crate) const USER_TEXT: &str = "What is the weather in Paris?";
pub(crate) const ASSISTANT_TEXT: &str = "I can check that.";
pub(crate) const TOOL_NAME: &str = "get_weather";

pub(crate) fn completion_request_fixture() -> CompletionRequest {
    CompletionRequest {
        model: Some(REQUEST_MODEL.to_string()),
        chat_history: OneOrMany::many(vec![
            Message::system(SYSTEM_TEXT),
            Message::user(USER_TEXT),
            Message::assistant(ASSISTANT_TEXT),
        ])
        .expect("fixture chat history should be non-empty"),
        documents: vec![],
        tools: vec![weather_tool()],
        temperature: Some(0.2),
        max_tokens: Some(128),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    }
}

pub(crate) fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: TOOL_NAME.to_string(),
        description: "Get the current weather for a city.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        }),
    }
}

pub(crate) fn assert_json_contains(value: &serde_json::Value, expected: &str) {
    let serialized = serde_json::to_string(value).expect("fixture JSON should serialize");
    assert!(
        serialized.contains(expected),
        "expected serialized provider request to contain {expected:?}, got {serialized}"
    );
}
