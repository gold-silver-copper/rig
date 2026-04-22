//! Migrated from `examples/agent_with_ollama.rs`.

use rig::client::CompletionClient;
use rig::client::Nothing;
use rig::completion::Prompt;
use rig::providers::ollama;

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn completion_smoke() -> anyhow::Result<()> {
    let client = ollama::Client::new(Nothing)?;
    let agent = client
        .agent("qwen3:4b")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = agent.prompt("Entertain me!").await?;

    assert_nonempty_response(&response);
    Ok(())
}
