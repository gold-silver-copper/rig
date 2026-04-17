//! OpenAI audio generation smoke test.

use rig::audio_generation::AudioGenerationModel;
use rig::client::ProviderClient;
use rig::client::audio_generation::AudioGenerationClient;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn audio_generation_smoke() {
    let client = rig::providers::openai::Client::from_env();
    let model = client.audio_generation_model(rig::models::openai::TTS_1);

    let response = model
        .audio_generation_request()
        .text(AUDIO_TEXT)
        .voice("alloy")
        .send()
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
