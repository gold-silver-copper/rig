//! Together tools smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn tools_smoke() {
    let client = rig::providers::together::Client::from_env();
    let agent = client
        .agent(rig::models::together::MIXTRAL_8X7B_INSTRUCT_V0_1)
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
