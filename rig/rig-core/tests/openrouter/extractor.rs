//! OpenRouter extractor smoke test.

use anyhow::{Result, anyhow};
use rig::client::{CompletionClient, ProviderClient};
use rig::providers::openrouter;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

use super::DEFAULT_MODEL;

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn extractor_smoke() -> Result<()> {
    let client = openrouter::Client::from_env()?;
    let extractor = client.extractor::<SmokePerson>(DEFAULT_MODEL).build();

    let response = extractor.extract_with_usage(EXTRACTOR_TEXT).await?;

    let first_name = response
        .data
        .first_name
        .as_deref()
        .ok_or_else(|| anyhow!("first_name should be present"))?;
    let last_name = response
        .data
        .last_name
        .as_deref()
        .ok_or_else(|| anyhow!("last_name should be present"))?;
    let job = response
        .data
        .job
        .as_deref()
        .ok_or_else(|| anyhow!("job should be present"))?;

    assert_nonempty_response(first_name);
    assert_nonempty_response(last_name);
    assert_nonempty_response(job);
    assert!(response.usage.total_tokens > 0, "usage should be populated");
    Ok(())
}
