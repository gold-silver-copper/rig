//! Anthropic structured output smoke test.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::anthropic::{self, completion::CLAUDE_SONNET_4_6};

use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn structured_output_smoke() -> Result<()> {
    let client = anthropic::Client::from_env()?;
    let agent = client
        .agent(CLAUDE_SONNET_4_6)
        .output_schema::<SmokeStructuredOutput>()
        .build();

    let response = agent.prompt(STRUCTURED_OUTPUT_PROMPT).await?;
    let structured: SmokeStructuredOutput = serde_json::from_str(&response)?;

    assert_smoke_structured_output(&structured);
    Ok(())
}
