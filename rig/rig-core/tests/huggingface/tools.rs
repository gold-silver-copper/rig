//! Hugging Face tools smoke test.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::huggingface;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn tools_smoke() -> Result<()> {
    let client = huggingface::Client::from_env()?;
    let agent = client
        .agent("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent.prompt(TOOLS_PROMPT).await?;

    assert_mentions_expected_number(&response, -3);
    Ok(())
}
