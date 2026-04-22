//! Migrated from `examples/transcription.rs`.

use anyhow::Result;
use rig::client::ProviderClient;
use rig::prelude::TranscriptionClient;
use rig::providers::gemini;
use rig::transcription::TranscriptionModel;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn transcription_smoke() -> Result<()> {
    let client = gemini::Client::from_env()?;
    let model = client.transcription_model(gemini::completion::GEMINI_3_FLASH_PREVIEW);
    let response = model
        .transcription_request()
        .load_file(AUDIO_FIXTURE_PATH)?
        .send()
        .await?;

    assert_nonempty_response(&response.text);
    Ok(())
}
