//! Migrated from `examples/openai_agent_completions_api.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn completions_api_agent_prompt() {
    let agent = rig::providers::openai::Client::from_env()
        .completion_model(rig::models::openai::GPT_4O)
        .completions_api()
        .into_agent_builder()
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent
        .prompt("Hello world!")
        .await
        .expect("completions api prompt should succeed");

    assert_nonempty_response(&response);
}
