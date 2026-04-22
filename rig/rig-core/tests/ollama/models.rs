//! Ollama model listing smoke test.

use anyhow::Result;
use rig::client::{ModelListingClient, Nothing};
use rig::providers::ollama;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn list_models_smoke() -> Result<()> {
    let client = ollama::Client::new(Nothing)?;
    let models = client.list_models().await?;

    assert!(
        !models.is_empty(),
        "expected Ollama to return at least one model\nModel list: {models:#?}"
    );
    Ok(())
}
