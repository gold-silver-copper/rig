//! xAI audio generation smoke test covering provider-specific additional parameters.

use rig::audio_generation::AudioGenerationModel;
use rig::client::ProviderClient;
use rig::client::audio_generation::AudioGenerationClient;
use serde_json::json;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn audio_generation_smoke() {
    let client = rig::providers::xai::Client::from_env();
    let model = client.audio_generation_model(rig::models::xai::TTS_1);

    let response = model
        .audio_generation_request()
        .text(AUDIO_TEXT)
        .voice("eve")
        .additional_params(json!({
            "language": "en",
        }))
        .send()
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
