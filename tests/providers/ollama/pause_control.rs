//! Migrated from `examples/ollama_streaming_pause_control.rs`.

use futures::StreamExt;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::model_event::ModelEvent;
use rig::providers::ollama;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn streaming_event_stream_processes_chunks() {
    let model = ollama::Client::from_env()
        .expect("client should build")
        .completion_model("gemma3:4b");
    let request = model
        .completion_request("Explain backpropagation in neural networks.")
        .preamble("You are a helpful AI assistant. Provide concise explanations.".to_string())
        .temperature(0.7)
        .build();
    let mut stream = model
        .stream_events(request)
        .await
        .expect("stream should start");

    let mut chunk_count = 0usize;
    while let Some(chunk) = stream.next().await {
        match chunk {
            ModelEvent::TextDelta { text } => {
                chunk_count += usize::from(!text.is_empty());
            }
            ModelEvent::ToolCallDone { .. } | ModelEvent::ReasoningDone { .. } => chunk_count += 1,
            ModelEvent::RawResponse { .. } => break,
            ModelEvent::Error { error } => panic!("stream chunk should succeed: {error}"),
            _ => {}
        }
    }

    assert!(
        chunk_count > 0,
        "expected to process at least one stream chunk"
    );
}
