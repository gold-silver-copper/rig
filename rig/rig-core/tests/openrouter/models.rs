//! OpenRouter model listing smoke test.
//!
//! Run with:
//! `cargo test -p rig-core --test openrouter openrouter::models::list_models_smoke -- --ignored --nocapture`

use anyhow::Result;
use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::openrouter;

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn list_models_smoke() -> Result<()> {
    let client = openrouter::Client::from_env()?;
    let models = client.list_models().await?;

    assert!(
        !models.is_empty(),
        "expected OpenRouter to return at least one model\nModel list: {models:#?}"
    );

    println!("OpenRouter returned {} models", models.len());
    Ok(())
}
