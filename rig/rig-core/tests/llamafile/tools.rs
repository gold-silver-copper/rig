//! Llamafile tools smoke test.

use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::Prompt;

use crate::support::{Adder, Subtract, assert_mentions_expected_number};

use super::support;

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn tools_smoke() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let client = support::client()?;
    let agent = client
        .agent(support::model_name())
        .preamble(
            "You are a calculator. For arithmetic requests, call the appropriate tool exactly once. \
             After you receive the tool result, do not call any more tools and reply with the final numeric answer only.",
        )
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt("Calculate 2 - 5. Call `subtract` exactly once, then answer with just the result.")
        .max_turns(3)
        .await?;

    assert_mentions_expected_number(&response, -3);
    Ok(())
}
