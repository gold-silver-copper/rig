//! Migrated from `examples/huggingface_subproviders.rs`.

use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::huggingface::{self, SubProvider};

use crate::support::{Adder, Subtract, assert_mentions_expected_number};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn tool_prompt_across_subproviders() -> Result<()> {
    let api_key = std::env::var("HUGGINGFACE_API_KEY")?;
    let cases = [
        ("deepseek-ai/DeepSeek-V3", SubProvider::Together),
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            SubProvider::HFInference,
        ),
        ("Meta-Llama-3.1-8B-Instruct", SubProvider::SambaNova),
    ];

    for (model, subprovider) in cases {
        let client = huggingface::Client::builder()
            .api_key(&api_key)
            .subprovider(subprovider)
            .build()?;
        let agent = client
            .agent(model)
            .preamble(
                "You are a calculator here to help the user perform arithmetic operations. \
                 Use the provided tools to answer the user's question.",
            )
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let response = agent.prompt("Calculate 2 - 5").await?;
        assert_mentions_expected_number(&response, -3);
    }
    Ok(())
}
