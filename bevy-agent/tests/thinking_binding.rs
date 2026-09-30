//! Regression: after a reload or a plugin adds a tool, Claude Opus 5.5
//! rejected the conversation, because each replayed thinking block is bound
//! to the tool list it was created with. The agent's Anthropic
//! configuration drops such blocks instead. Replays by default; record with
//! `RIG_PROVIDER_TEST_MODE=record` and `ANTHROPIC_API_KEY`.

use bevy_agent::glue::provider_config;
use rig_cassette::http::{CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
use rig_core::completion::{CompletionRequest, ToolDefinition};
use rig_core::message::{AssistantContent, Message, ToolResultContent};
use rig_core::providers::registry::ProviderRef;
use serde_json::json;

fn tool(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: description.into(),
        parameters: json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"]
        }),
    }
}

#[tokio::test]
async fn an_anthropic_conversation_survives_a_tool_added_between_turns() {
    let mode = CassetteMode::current();
    let cassette = ProviderCassette::try_start_at(
        RecordVia::Proxy,
        "anthropic",
        CassetteSpec::new("tool-added-between-turns"),
        "https://api.anthropic.com",
        mode,
        bevy_agent::cassette_root().join("regressions/tool-added-between-turns/anthropic.yaml"),
    )
    .await
    .expect("the cassette opens");
    let key = match mode {
        CassetteMode::Record => std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY"),
        CassetteMode::Replay => "[REDACTED]".to_owned(),
    };
    let reference = ProviderRef::parse("anthropic:claude-opus-5-5").expect("a reference");
    let model = provider_config(&reference, key)
        .with_base_url(cassette.base_url())
        .completion_model(reference.model(), rig_reqwest::shared());

    let prompt = Message::user(
        "Carefully reason about it first, then use the add tool to compute 1234+5678.",
    );
    let mut first = CompletionRequest::new(prompt.clone());
    first.tools = vec![tool("add", "Add two integers")];
    first.additional_params =
        Some(json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}));
    let reply = model.call(first).await.expect("the first turn");
    assert!(
        reply
            .choice
            .iter()
            .any(|part| matches!(part, AssistantContent::Reasoning(_))),
        "the first turn thinks, so the second replays a thinking block"
    );
    let call = reply.tool_calls().next().cloned().expect("a tool call");

    let mut second = CompletionRequest::new(prompt);
    second
        .chat_history
        .push(reply.message().expect("an assistant message"));
    second.chat_history.push(Message::tool_results(vec![
        call.result(vec![ToolResultContent::text("6912")]),
    ]));
    second.tools = vec![
        tool("add", "Add two integers"),
        tool("mul", "Multiply two integers"),
    ];
    second.additional_params =
        Some(json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}));
    let answer = model
        .call(second)
        .await
        .expect("the second turn, with a tool added");
    assert!(
        answer.text().contains("6912") || answer.text().contains("6,912"),
        "{}",
        answer.text()
    );

    cassette
        .try_finish()
        .await
        .expect("the recording is complete");
}
