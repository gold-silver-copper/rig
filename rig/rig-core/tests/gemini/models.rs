//! Gemini model listing smoke test.

use anyhow::Result;
use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::gemini;

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn list_models_smoke() -> Result<()> {
    let client = gemini::Client::from_env()?;
    let models = client.list_models().await?;

    println!("Gemini returned {} models", models.len());

    assert!(
        !models.is_empty(),
        "expected Gemini to return at least one model\nModel list: {models:#?}"
    );
    Ok(())
}
