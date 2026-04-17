//! DeepSeek tools smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn tools_smoke() {
    let client = rig::providers::deepseek::Client::from_env();
    let agent = client
        .agent(rig::models::deepseek::DEEPSEEK_CHAT)
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
