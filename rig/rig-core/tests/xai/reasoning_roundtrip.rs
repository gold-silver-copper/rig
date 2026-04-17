//! xAI reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test xai xai::reasoning_roundtrip::streaming -- --ignored --nocapture`

use rig::client::{CompletionClient, ProviderClient};

use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires XAI_API_KEY - validate with grok-4-0725 once key is available"]
async fn streaming() {
    let client = rig::providers::xai::Client::from_env();
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model(rig::models::xai::GROK_3_MINI),
        None,
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY - validate with grok-4-0725 once key is available"]
async fn nonstreaming() {
    let client = rig::providers::xai::Client::from_env();
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model(rig::models::xai::GROK_3_MINI),
        None,
    ))
    .await;
}
