//! xAI reasoning tool roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test xai xai::reasoning_tool_roundtrip::streaming -- --ignored --nocapture`

use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Chat, Message};
use rig::providers::xai;
use rig::streaming::StreamingChat;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires XAI_API_KEY - validate with grok-4-0725 once key is available"]
async fn streaming() -> Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = xai::Client::from_env()?;
    let agent = client
        .agent(xai::GROK_3_MINI)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .build();

    let stream = agent
        .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .await;

    let stats = reasoning::collect_stream_stats(stream, "xai").await;
    reasoning::assert_universal(&stats, &call_count, "xai");
    Ok(())
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY - validate with grok-4-0725 once key is available"]
async fn nonstreaming() -> Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = xai::Client::from_env()?;
    let agent = client
        .agent(xai::GROK_3_MINI)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await?;

    reasoning::assert_nonstreaming_universal(&result, &call_count, "xai");
    Ok(())
}
