use anyhow::Result;
use rig::providers::openai;

const DEFAULT_API_BASE_URL: &str = "http://localhost:8080/v1";
const DEFAULT_API_KEY: &str = "none";
const DEFAULT_MODEL: &str = "model";

pub(super) fn api_base_url() -> String {
    std::env::var("LLAMACPP_API_BASE_URL").unwrap_or_else(|_| DEFAULT_API_BASE_URL.to_string())
}

pub(super) fn api_key() -> String {
    std::env::var("LLAMACPP_API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.to_string())
}

pub(super) fn model_name() -> String {
    std::env::var("LLAMACPP_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

pub(super) fn client() -> Result<openai::Client> {
    let api_key = api_key();
    let base_url = api_base_url();

    Ok(openai::Client::builder()
        .api_key(&api_key)
        .base_url(&base_url)
        .build()?)
}

pub(super) fn completions_client() -> Result<openai::CompletionsClient> {
    Ok(client()?.completions_api())
}
