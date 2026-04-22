use anyhow::Result;

mod anthropic;
mod coding;
mod general;

use rig::providers::zai;

pub(crate) fn api_key() -> Result<String> {
    Ok(std::env::var("ZAI_API_KEY")?)
}

pub(crate) fn general_client() -> Result<zai::Client> {
    Ok(zai::Client::builder()
        .api_key(api_key()?)
        .general()
        .build()?)
}

pub(crate) fn coding_client() -> Result<zai::Client> {
    Ok(zai::Client::builder()
        .api_key(api_key()?)
        .coding()
        .build()?)
}

pub(crate) fn anthropic_client() -> Result<zai::AnthropicClient> {
    Ok(zai::AnthropicClient::builder()
        .api_key(api_key()?)
        .general()
        .build()?)
}
