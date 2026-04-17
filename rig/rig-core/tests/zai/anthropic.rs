//! Z.AI Anthropic-compatible completion smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use crate::zai::anthropic_client;

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let response = anthropic_client()
        .agent(rig::models::zai::GLM_4_6)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("Z.AI Anthropic-compatible completion should succeed");

    assert_nonempty_response(&response);
}
