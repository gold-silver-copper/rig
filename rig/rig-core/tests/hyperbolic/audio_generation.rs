//! Hyperbolic audio generation smoke test.

use anyhow::Result;
use rig::audio_generation::AudioGenerationModel;
use rig::client::ProviderClient;
use rig::client::audio_generation::AudioGenerationClient;
use rig::providers::hyperbolic;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn audio_generation_smoke() -> Result<()> {
    let client = hyperbolic::Client::from_env()?;
    let model = client.audio_generation_model("EN");

    let response = model
        .audio_generation_request()
        .text(AUDIO_TEXT)
        .voice("EN-US")
        .send()
        .await?;

    assert_nonempty_bytes(&response.audio);
    Ok(())
}
