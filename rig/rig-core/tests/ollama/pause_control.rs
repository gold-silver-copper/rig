//! Migrated from `examples/ollama_streaming_pause_control.rs`.

use anyhow::Result;
use futures::StreamExt;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::providers::ollama;
use rig::streaming::StreamedAssistantContent;
use tokio::time::{Duration, sleep};

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn streaming_pause_and_resume() -> Result<()> {
    let model = ollama::Client::from_env()?.completion_model("gemma3:4b");
    let request = model
        .completion_request("Explain backpropagation in neural networks.")
        .preamble("You are a helpful AI assistant. Provide concise explanations.".to_string())
        .temperature(0.7)
        .build();
    let mut stream = model.stream(request).await?;

    let mut chunk_count = 0usize;
    let mut paused_once = false;
    while let Some(chunk) = stream.next().await {
        match chunk? {
            StreamedAssistantContent::Text(text) => {
                chunk_count += usize::from(!text.text.is_empty());
            }
            StreamedAssistantContent::ToolCall { .. } | StreamedAssistantContent::Reasoning(_) => {
                chunk_count += 1
            }
            StreamedAssistantContent::Final(_) => break,
            _ => {}
        }

        if !paused_once && chunk_count > 0 {
            stream.pause()?;
            sleep(Duration::from_millis(50)).await;
            stream.resume()?;
            paused_once = true;
        }
    }

    assert!(paused_once, "expected to exercise pause/resume");
    assert!(
        chunk_count > 0,
        "expected to process at least one stream chunk"
    );
    Ok(())
}
