//! OpenAI reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test openai openai::reasoning_roundtrip::streaming -- --ignored --nocapture`

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::providers::openai;

use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn streaming() -> Result<()> {
    let client = openai::Client::from_env()?;
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model("gpt-5.2"),
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    ))
    .await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn nonstreaming() -> Result<()> {
    let client = openai::Client::from_env()?;
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model("gpt-5.2"),
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    ))
    .await;
    Ok(())
}
