//! DeepSeek tools smoke test.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::deepseek;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn tools_smoke() -> Result<()> {
    let client = deepseek::Client::from_env()?;
    let agent = client
        .agent(deepseek::DEEPSEEK_CHAT)
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent.prompt(TOOLS_PROMPT).await?;

    assert_mentions_expected_number(&response, -3);
    Ok(())
}
