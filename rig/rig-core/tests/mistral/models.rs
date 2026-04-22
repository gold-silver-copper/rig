//! Mistral model listing smoke test.

use anyhow::Result;
use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::mistral;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn list_models_smoke() -> Result<()> {
    let client = mistral::Client::from_env()?;
    let models = client.list_models().await?;

    assert!(
        !models.is_empty(),
        "expected Mistral to return at least one model\nModel list: {models:#?}"
    );
    Ok(())
}
