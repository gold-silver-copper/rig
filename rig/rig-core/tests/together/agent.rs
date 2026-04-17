//! Together agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn completion_smoke() {
    let client = rig::providers::together::Client::from_env();
    let agent = client
        .agent(rig::models::together::MIXTRAL_8X7B_INSTRUCT_V0_1)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
