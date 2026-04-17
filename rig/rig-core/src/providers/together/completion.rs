// ================================================================
//! Together AI Completion Integration
//! From [Together AI Reference](https://docs.together.ai/docs/chat-overview)
// ================================================================

use crate::{
    completion::{self, CompletionError},
    http_client::HttpClientExt,
};

use super::client::{Client, together_ai_api_types::ApiResponse};
use crate::completion::CompletionRequest;
use crate::streaming::StreamingCompletionResponse;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, Level, enabled, info_span};

// ================================================================
// Together Completion Models
// ================================================================

// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct TogetherAICompletionRequest {
    model: String,
    pub messages: Vec<rig::providers::openai::Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<rig::providers::openai::ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for TogetherAICompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        if req.output_schema.is_some() {
            tracing::warn!("Structured outputs currently not supported for TogetherAI");
        }
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        let mut full_history: Vec<rig::providers::openai::Message> = match &req.preamble {
            Some(preamble) => vec![rig::providers::openai::Message::system(preamble)],
            None => vec![],
        };
        if let Some(docs) = req.normalized_documents() {
            let docs: Vec<rig::providers::openai::Message> = docs.try_into()?;
            full_history.extend(docs);
        }

        let chat_history: Vec<rig::providers::openai::Message> = req
            .chat_history
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<rig::providers::openai::Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        if full_history.is_empty() {
            return Err(CompletionError::RequestError(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Together request has no provider-compatible messages after conversion",
                )
                .into(),
            ));
        }

        let tool_choice = req
            .tool_choice
            .clone()
            .map(ToolChoice::try_from)
            .transpose()?;

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(rig::providers::openai::ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice,
            additional_params: req.additional_params,
        })
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = rig::providers::openai::CompletionResponse;
    type StreamingResponse = rig::providers::openai::StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<
        completion::CompletionResponse<rig::providers::openai::CompletionResponse>,
        CompletionError,
    > {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "together",
                gen_ai.request.model = self.model.to_string(),
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &completion_request.preamble);

        let request = TogetherAICompletionRequest::try_from((
            self.model.to_string().as_ref(),
            completion_request,
        ))?;

        if enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "TogetherAI completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/v1/chat/completions")?
            .body(body)
            .map_err(|x| CompletionError::HttpError(x.into()))?;

        async move {
            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<
                    ApiResponse<rig::providers::openai::CompletionResponse>,
                >(&response_body)?
                {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record("gen_ai.response.id", &response.id);
                        span.record("gen_ai.response.model_name", &response.model);
                        if let Some(ref usage) = response.usage {
                            span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
                            span.record(
                                "gen_ai.usage.output_tokens",
                                usage.total_tokens - usage.prompt_tokens,
                            );
                        }
                        if enabled!(Level::TRACE) {
                            tracing::trace!(
                                target: "rig::completions",
                                "TogetherAI completion response: {}",
                                serde_json::to_string_pretty(&response)?
                            );
                        }
                        response.try_into()
                    }
                    ApiResponse::Error(err) => Err(CompletionError::ProviderError(err.error)),
                }
            } else {
                Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        CompletionModel::stream(self, request).await
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum ToolChoice {
    None,
    Auto,
    Function(Vec<ToolChoiceFunctionKind>),
}

impl TryFrom<crate::message::ToolChoice> for ToolChoice {
    type Error = CompletionError;

    fn try_from(value: crate::message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            crate::message::ToolChoice::None => Self::None,
            crate::message::ToolChoice::Auto => Self::Auto,
            crate::message::ToolChoice::Specific { function_names } => {
                let vec: Vec<ToolChoiceFunctionKind> = function_names
                    .into_iter()
                    .map(|name| ToolChoiceFunctionKind::Function { name })
                    .collect();

                Self::Function(vec)
            }
            choice => {
                return Err(CompletionError::ProviderError(format!(
                    "Unsupported tool choice type: {choice:?}"
                )));
            }
        };

        Ok(res)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "function")]
pub enum ToolChoiceFunctionKind {
    Function { name: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OneOrMany, message};

    #[test]
    fn together_request_conversion_errors_when_all_messages_are_filtered() {
        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(message::Message::Assistant {
                id: None,
                content: OneOrMany::one(message::AssistantContent::reasoning("hidden")),
            }),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            model: None,
            output_schema: None,
        };

        let result = TogetherAICompletionRequest::try_from(("meta-llama/test-model", request));
        assert!(matches!(result, Err(CompletionError::RequestError(_))));
    }
}
