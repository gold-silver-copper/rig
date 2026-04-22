//! Z.AI coding OpenAI-compatible completion smoke test.

use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::zai;

use crate::support::assert_nonempty_response;
use crate::zai::coding_client;

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn coding_openai_compatible_completion_smoke() -> Result<()> {
    let response = coding_client()?
        .agent(zai::GLM_4_6)
        .preamble("You are a concise coding assistant.")
        .build()
        .prompt("In one short sentence, explain what a unit test is.")
        .await?;

    assert_nonempty_response(&response);
    Ok(())
}
