//! MiniMax OpenAI-compatible completion smoke test.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::minimax;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn openai_compatible_completion_smoke() -> Result<()> {
    let response = minimax::Client::from_env()?
        .agent(minimax::MINIMAX_M2_7)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await?;

    assert_nonempty_response(&response);
    Ok(())
}
