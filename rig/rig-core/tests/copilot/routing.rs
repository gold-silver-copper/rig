//! Copilot route-specific completion smoke tests.

use crate::copilot::{LIVE_MODEL, live_client, live_responses_model};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::Prompt;

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn chat_models_route_through_chat_completions() -> Result<()> {
    let response = live_client()?
        .agent(LIVE_MODEL)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await?;

    assert_nonempty_response(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn codex_models_route_through_responses() -> Result<()> {
    let response = live_client()?
        .agent(live_responses_model())
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt("In one short sentence, explain what refactoring is.")
        .await?;

    assert_nonempty_response(&response);
    Ok(())
}
