//! Migrated from `examples/transcription.rs`.

use rig::client::ProviderClient;
use rig::prelude::TranscriptionClient;
use rig::transcription::TranscriptionModel;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn transcription_smoke() {
    let client = rig::providers::huggingface::Client::from_env();
    let model = client.transcription_model("whisper-large-v3");
    let response = model
        .transcription_request()
        .load_file(AUDIO_FIXTURE_PATH)
        .expect("should be able to load audio fixture")
        .send()
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
