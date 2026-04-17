//! xAI agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn completion_smoke() {
    let client = rig::providers::xai::Client::from_env();
    let agent = client
        .agent(rig::models::xai::GROK_3_MINI)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
