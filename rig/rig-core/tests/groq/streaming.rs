//! Groq streaming smoke test.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::providers::groq;
use rig::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::STREAMING_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn streaming_smoke() -> Result<()> {
    let client = groq::Client::from_env()?;
    let agent = client
        .agent(STREAMING_MODEL)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream).await?;

    assert_nonempty_response(&response);
    Ok(())
}
