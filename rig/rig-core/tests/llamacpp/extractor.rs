//! llama.cpp extractor smoke test.

use anyhow::{Result, anyhow};

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

use super::support;

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn extractor_smoke() -> Result<()> {
    let client = support::completions_client()?;
    let extractor = client
        .extractor::<SmokePerson>(support::model_name())
        .build();

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
