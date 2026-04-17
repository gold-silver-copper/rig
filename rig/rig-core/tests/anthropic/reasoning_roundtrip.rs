//! Anthropic reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test anthropic anthropic::reasoning_roundtrip::streaming -- --ignored --nocapture`

use rig::client::{CompletionClient, ProviderClient};
use rig::models::anthropic::CLAUDE_SONNET_4_6;

use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn streaming() {
    let client = rig::providers::anthropic::Client::from_env();
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model(CLAUDE_SONNET_4_6),
        Some(serde_json::json!({
            "thinking": { "type": "adaptive" }
        })),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn nonstreaming() {
    let client = rig::providers::anthropic::Client::from_env();
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model(CLAUDE_SONNET_4_6),
        Some(serde_json::json!({
            "thinking": { "type": "adaptive" }
        })),
    ))
    .await;
}
