//! The Rig model layer: pick a provider from the environment and stream one
//! completion, forwarding text as it arrives.

use bevy_tasks::futures_lite::StreamExt;
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::driver::DynModel;
use rig_core::error::ProviderError;
use rig_core::operation::Completion;
use rig_core::providers::{anthropic, openai};
use rig_core::streaming::{Item, StreamEvent};

/// The model named by `AGENT_PROVIDER` (`anthropic` or `openai`, defaulting to
/// whichever key is set) and `AGENT_MODEL`, with its display name.
pub fn from_env() -> anyhow::Result<(DynModel<Completion>, String)> {
    let provider = std::env::var("AGENT_PROVIDER").unwrap_or_else(|_| {
        if std::env::var_os("ANTHROPIC_API_KEY").is_some() {
            "anthropic"
        } else {
            "openai"
        }
        .to_owned()
    });
    let model = std::env::var("AGENT_MODEL").ok();
    Ok(match provider.as_str() {
        "anthropic" => {
            let model = model.unwrap_or_else(|| anthropic::CLAUDE_HAIKU_4_5.to_owned());
            (
                anthropic::Anthropic::from_env()?.completion(&model).erase(),
                model,
            )
        }
        "openai" => {
            let model = model.unwrap_or_else(|| openai::GPT_5_6.to_owned());
            (openai::OpenAI::from_env()?.responses(&model).erase(), model)
        }
        other => anyhow::bail!("unknown AGENT_PROVIDER `{other}`"),
    })
}

/// Stream `request`, sending each text fragment to `deltas`, and return the
/// folded response.
pub async fn complete(
    model: DynModel<Completion>,
    request: CompletionRequest,
    deltas: async_channel::Sender<String>,
) -> Result<CompletionResponse, ProviderError> {
    let mut stream = model.stream(request)?;
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text, .. }) = item? {
            let _ = deltas.send(text).await;
        }
    }
    stream.finish().await
}
