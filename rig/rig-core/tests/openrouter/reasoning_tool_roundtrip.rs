//! OpenRouter reasoning tool roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test openrouter openrouter::reasoning_tool_roundtrip::streaming -- --ignored --nocapture`

use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Chat, Message};
use rig::providers::openrouter;
use rig::streaming::StreamingChat;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn streaming() -> Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = openrouter::Client::from_env()?;
    let agent = client
        .agent("openai/gpt-5.2")
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "reasoning": { "effort": "high" },
            "include_reasoning": true
        }))
        .build();

    let stream = agent
        .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .await;

    let stats = reasoning::collect_stream_stats(stream, "openrouter").await;
    reasoning::assert_universal(&stats, &call_count, "openrouter");
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn nonstreaming() -> Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = openrouter::Client::from_env()?;
    let agent = client
        .agent("openai/gpt-5.2")
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "reasoning": { "effort": "high" },
            "include_reasoning": true
        }))
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await?;

    reasoning::assert_nonstreaming_universal(&result, &call_count, "openrouter");
    Ok(())
}
