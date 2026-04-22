//! llama.cpp agent completion smoke test.

use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::Prompt;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::support;

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn completion_smoke() -> Result<()> {
    let client = support::completions_client()?;
    let agent = client
        .agent(support::model_name())
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent.prompt(BASIC_PROMPT).await?;

    assert_nonempty_response(&response);
    Ok(())
}
