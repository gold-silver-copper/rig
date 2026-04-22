//! xAI tools smoke test.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::xai;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn tools_smoke() -> Result<()> {
    let client = xai::Client::from_env()?;
    let agent = client
        .agent(xai::completion::GROK_3_MINI)
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent.prompt(TOOLS_PROMPT).await?;

    assert_mentions_expected_number(&response, -3);
    Ok(())
}
