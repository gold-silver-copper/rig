//! Mira tools smoke test.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::{anthropic, mira};

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn tools_smoke() -> Result<()> {
    let client = mira::Client::from_env()?;
    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent.prompt(TOOLS_PROMPT).await?;

    assert_mentions_expected_number(&response, -3);
    Ok(())
}
