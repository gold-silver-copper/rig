//! Copilot reasoning roundtrip tests.

use anyhow::Result;
use rig::client::CompletionClient;

use crate::copilot::{live_client, live_responses_model};
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn streaming() -> Result<()> {
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        live_client()?.completion_model(live_responses_model()),
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    ))
    .await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn nonstreaming() -> Result<()> {
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        live_client()?.completion_model(live_responses_model()),
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    ))
    .await;
    Ok(())
}
