//! Cohere tools smoke test.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::cohere;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn tools_smoke() -> Result<()> {
    let client = cohere::Client::from_env()?;
    let agent = client
        .agent(cohere::COMMAND_R)
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent.prompt(TOOLS_PROMPT).await?;

    assert_mentions_expected_number(&response, -3);
    Ok(())
}
