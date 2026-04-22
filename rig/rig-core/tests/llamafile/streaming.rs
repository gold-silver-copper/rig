//! Llamafile streaming coverage.

use anyhow::Result;
use rig::client::CompletionClient;
use rig::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::support;

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn streaming_smoke() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let client = support::client()?;
    let agent = client
        .agent(support::model_name())
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream).await?;

    assert_nonempty_response(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn example_streaming_prompt() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let client = support::client()?;
    let agent = client
        .agent(support::model_name())
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await;
    let response = collect_stream_final_response(&mut stream).await?;

    assert_nonempty_response(&response);
    Ok(())
}
