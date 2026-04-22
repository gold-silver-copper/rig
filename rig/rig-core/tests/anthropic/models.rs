//! Anthropic model listing smoke test.

use anyhow::Result;
use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::anthropic;

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn list_models_smoke() -> Result<()> {
    let client = anthropic::Client::from_env()?;
    let models = client.list_models().await?;

    assert!(
        !models.is_empty(),
        "expected Anthropic to return at least one model\nModel list: {models:#?}"
    );
    Ok(())
}
