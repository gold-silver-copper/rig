//! Mira agent completion smoke test.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::{mira, openai};

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn completion_smoke() -> Result<()> {
    let client = mira::Client::from_env()?;
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent.prompt(BASIC_PROMPT).await?;

    assert_nonempty_response(&response);
    Ok(())
}
