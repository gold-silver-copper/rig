//! Live Gemini canary for the Cortex/Rig `default_api` tool-call leak.

use futures::StreamExt;
use rig::agent::{FinalResponse, MultiTurnStreamItem, StreamingResult};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{ToolDefinition, Usage};
use rig::message::ToolResultContent;
use rig::providers::gemini::{
    self,
    completion::gemini_api_types::{AdditionalParameters, GenerationConfig, ThinkingConfig},
};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt};
use rig::tool::Tool;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

const CORTEX_GEMINI_MODEL: &str = "gemini-3.1-pro-preview";
const THINKING_BUDGET: u32 = 1024 * 16;
const CANARY_MAX_TURNS: usize = 8;
const JAVASCRIPT_MARKER: &str = "cortex-bundle-marker";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct JavaScriptProgram {
    title: String,
    description: String,
    code: String,
}

#[repr(transparent)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExecutorResponse(std::result::Result<serde_json::Value, String>);

impl ExecutorResponse {
    fn ok(value: serde_json::Value) -> Self {
        Self(Ok(value))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("JavaScript tool error")]
struct JavaScriptToolError;

#[derive(Clone)]
struct JavaScript;

impl Tool for JavaScript {
    const NAME: &'static str = "JavaScript";
    type Error = JavaScriptToolError;
    type Args = JavaScriptProgram;
    type Output = ExecutorResponse;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "JavaScript runtime with an array of tools for completing the tasks assigned by the user".into(),
            parameters: schema_for!(JavaScriptProgram).to_value(),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(ExecutorResponse::ok(json!({
            "bundle": JAVASCRIPT_MARKER,
            "receivedCode": args.code,
        })))
    }
}

#[derive(Debug, Default)]
struct CortexStreamObservation {
    streamed_text: String,
    reasoning_text: String,
    final_response: Option<FinalResponse>,
    tool_calls: Vec<String>,
    tool_call_deltas: usize,
    executions: Vec<JavaScriptProgram>,
    executor_results: Vec<ExecutorResponse>,
    completion_calls: Vec<String>,
    events: Vec<&'static str>,
}

impl CortexStreamObservation {
    fn final_response_text(&self) -> Option<&str> {
        self.final_response
            .as_ref()
            .map(|response| response.response())
    }

    fn diagnostic_summary(&self) -> String {
        format!(
            "events={:?}, tool_calls={:?}, tool_call_deltas={}, executions={:?}, executor_results={:?}, streamed_text={:?}, reasoning_text={:?}, completion_calls={:?}, final={:?}, final_usage={:?}",
            self.events,
            self.tool_calls,
            self.tool_call_deltas,
            self.executions,
            self.executor_results,
            self.streamed_text,
            self.reasoning_text,
            self.completion_calls,
            self.final_response_text(),
            self.final_response
                .as_ref()
                .map(|response| response.usage())
        )
    }
}

fn cortex_additional_params() -> serde_json::Value {
    let additional_params = AdditionalParameters {
        generation_config: Some(GenerationConfig {
            thinking_config: Some(ThinkingConfig {
                include_thoughts: Some(true),
                thinking_budget: Some(THINKING_BUDGET),
                thinking_level: None,
            }),
            ..Default::default()
        }),
        additional_params: None,
    };

    serde_json::to_value(&additional_params).expect("GenerationConfig should always serialize")
}

async fn consume_cortex_like_stream(
    mut stream: StreamingResult<gemini::streaming::StreamingCompletionResponse>,
) -> Result<CortexStreamObservation, String> {
    let mut observation = CortexStreamObservation::default();

    while let Some(item) = stream.next().await {
        match item.map_err(|error| error.to_string())? {
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) => {
                observation.events.push("text");
                observation.streamed_text.push_str(&text.text);
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            )) => {
                observation.events.push("reasoning");
                observation
                    .reasoning_text
                    .push_str(&reasoning.display_text());
            }
            MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { reasoning, .. },
            ) => {
                observation.events.push("reasoning_delta");
                observation.reasoning_text.push_str(&reasoning);
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call,
                ..
            }) => {
                observation.events.push("tool_call");
                observation.tool_calls.push(tool_call.function.name.clone());
                let execution: JavaScriptProgram =
                    serde_json::from_value(tool_call.function.arguments.clone())
                        .map_err(|error| error.to_string())?;
                observation.executions.push(execution);
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCallDelta {
                ..
            }) => {
                observation.events.push("tool_call_delta");
                observation.tool_call_deltas += 1;
            }
            MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result,
                ..
            }) => {
                observation.events.push("tool_result");
                let ToolResultContent::Text(text) = tool_result.content.first() else {
                    return Err("JS Runtime can only respond with JSON text".to_string());
                };
                let result: ExecutorResponse =
                    serde_json::from_str(&text.text).map_err(|error| error.to_string())?;
                observation.executor_results.push(result);
            }
            MultiTurnStreamItem::CompletionCall(completion_call) => {
                observation.events.push("completion_call");
                observation.completion_calls.push(format!(
                    "call_index={}, usage={:?}",
                    completion_call.call_index, completion_call.usage
                ));
            }
            MultiTurnStreamItem::FinalResponse(final_response) => {
                observation.events.push("final_response");
                observation.final_response = Some(final_response);
                return Ok(observation);
            }
            _ => {}
        }
    }

    Ok(observation)
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY and calls the live Gemini API"]
async fn streaming_javascript_tool_does_not_surface_default_api_or_empty_final_response() {
    let client = gemini::Client::from_env().expect("client should build");
    let agent = client
        .agent(CORTEX_GEMINI_MODEL)
        .name("cortex-default-api-canary")
        .preamble(
            "You are testing a Cortex/Rig JavaScript tool flow. Use the JavaScript runtime tool \
             for every request that asks you to inspect, fetch, compute, query, or execute code. \
             Do not answer from the prompt alone. Do not emit markdown tool_code, executableCode, \
             or default_api. After the tool result is available, answer directly in text.",
        )
        .additional_params(cortex_additional_params())
        .tool(JavaScript)
        .default_max_turns(CANARY_MAX_TURNS)
        .temperature(0.0)
        .build();

    let stream = agent
        .stream_prompt(
            "You must inspect a Ryzome bundle by calling the JavaScript tool exactly once. Put \
             exactly this JavaScript in the code field: async function findBundle() { try { const \
             bundle = await Ryzome.listBundle(\"6a07410250db81016def9dd1\", 1); return bundle; } \
             catch (e) { return `Error: ${e.message}`; } } findBundle(); Do not write a normal \
             answer until after the JavaScript tool result. After the tool result, answer in one \
             sentence containing cortex-bundle-marker.",
        )
        .with_history(Vec::<rig::message::Message>::new())
        .await;
    let observation = consume_cortex_like_stream(stream)
        .await
        .expect("Cortex-like stream should complete without protocol errors");

    assert!(
        observation
            .tool_calls
            .iter()
            .all(|name| !name.contains("default_api")),
        "Gemini leaked default_api into tool calls: {:?}",
        observation.tool_calls
    );
    assert!(
        !observation.reasoning_text.contains("default_api")
            && !observation.streamed_text.contains("default_api"),
        "Gemini leaked default_api outside structured tool calls: {}",
        observation.diagnostic_summary()
    );
    assert!(
        !observation.reasoning_text.contains("tool_code")
            && !observation.streamed_text.contains("tool_code"),
        "Gemini emitted markdown tool_code text instead of a function call: {}",
        observation.diagnostic_summary()
    );
    assert!(
        observation
            .tool_calls
            .iter()
            .any(|name| name == JavaScript::NAME),
        "expected live Gemini to call JavaScript. {}",
        observation.diagnostic_summary()
    );
    assert!(
        observation
            .executions
            .iter()
            .any(|execution| execution.code.contains("Ryzome.listBundle")),
        "expected Cortex-shaped JavaScript execution, got {:?}",
        observation.executions
    );
    assert!(
        !observation.executor_results.is_empty(),
        "expected at least one parsed ExecutorResponse"
    );

    let final_response = observation
        .final_response
        .as_ref()
        .expect("stream should yield a final response");
    assert!(
        final_response.usage() != Usage::new(),
        "live Gemini final response should include usage"
    );
    assert!(
        !final_response.response().trim().is_empty(),
        "final response should not be empty after the tool roundtrip"
    );
    assert!(
        final_response.response().contains(JAVASCRIPT_MARKER),
        "final response should mention the tool marker, got {:?}",
        final_response.response()
    );
}
