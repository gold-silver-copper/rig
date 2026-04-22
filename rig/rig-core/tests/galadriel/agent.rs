//! Galadriel agent completion smoke test.

use anyhow::{Context, Result};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::galadriel;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires GALADRIEL_API_KEY"]
async fn completion_smoke() -> Result<()> {
    let client = galadriel::Client::from_env()?;
    let agent = client
        .agent(galadriel::GPT_4O)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent.prompt(BASIC_PROMPT).await?;

    assert_nonempty_response(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires GALADRIEL_API_KEY"]
async fn builder_completion_smoke() -> Result<()> {
    let api_key = std::env::var("GALADRIEL_API_KEY").context("GALADRIEL_API_KEY must be set")?;
    let fine_tune_api_key = std::env::var("GALADRIEL_FINE_TUNE_API_KEY").ok();

    let mut builder = galadriel::Client::builder().api_key(&api_key);
    if let Some(fine_tune_api_key) = fine_tune_api_key.as_deref() {
        builder = builder.fine_tune_api_key(fine_tune_api_key);
    }

    let client = builder.build()?;
    let agent = client
        .agent(galadriel::GPT_4O)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = agent.prompt("Entertain me!").await?;

    assert_nonempty_response(&response);
    Ok(())
}
