//! Codecs for OpenAI Chat Completions-compatible payloads.

use crate::{
    OneOrMany,
    completion::{
        self, CompletionCodec, CompletionError, CompletionRequest as CoreCompletionRequest,
        CompletionResponseCodec, ModelTurn,
    },
};

use super::{
    AssistantContent, CompletionRequest, CompletionResponse, GenericCompletionModel, Message,
    OpenAIRequestParams,
};

/// Encodes and decodes OpenAI Chat Completions API payloads.
#[derive(Clone, Debug)]
pub(crate) struct ChatCompletionCodec {
    model: String,
    strict_tools: bool,
    tool_result_array_content: bool,
}

impl ChatCompletionCodec {
    /// Creates a codec for one Chat Completions model configuration.
    pub(crate) fn new(
        model: impl Into<String>,
        strict_tools: bool,
        tool_result_array_content: bool,
    ) -> Self {
        Self {
            model: model.into(),
            strict_tools,
            tool_result_array_content,
        }
    }

    /// Decodes a non-streaming Chat Completions response into a normalized model turn.
    pub(crate) fn response_to_turn(
        response: CompletionResponse,
    ) -> Result<ModelTurn<CompletionResponse>, CompletionError> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
                ..
            } => {
                let mut content = content
                    .iter()
                    .filter_map(|c| {
                        let s = match c {
                            AssistantContent::Text { text } => text,
                            AssistantContent::Refusal { refusal } => refusal,
                        };
                        if s.is_empty() {
                            None
                        } else {
                            Some(completion::AssistantContent::text(s))
                        }
                    })
                    .collect::<Vec<_>>();

                if let Some(reasoning) = reasoning {
                    content.push(completion::AssistantContent::reasoning(reasoning));
                }

                content.extend(tool_calls.iter().map(|call| {
                    completion::AssistantContent::tool_call(
                        &call.id,
                        &call.function.name,
                        call.function.arguments.clone(),
                    )
                }));
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = OneOrMany::from_iter_optional(content).ok_or_else(|| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                total_tokens: usage.total_tokens as u64,
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens as u64)
                    .unwrap_or(0),
                cache_creation_input_tokens: 0,
            })
            .unwrap_or_default();

        Ok(ModelTurn::new(choice, usage).with_raw_response(response))
    }
}

impl CompletionCodec for ChatCompletionCodec {
    type Request = CompletionRequest;

    fn encode_request(
        &self,
        request: CoreCompletionRequest,
    ) -> Result<Self::Request, CompletionError> {
        CompletionRequest::from_params(OpenAIRequestParams {
            model: self.model.clone(),
            request,
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
        })
    }
}

impl CompletionResponseCodec for ChatCompletionCodec {
    type Response = CompletionResponse;
    type RawFinal = CompletionResponse;

    fn decode_response(
        &self,
        response: Self::Response,
    ) -> Result<ModelTurn<Self::RawFinal>, CompletionError> {
        Self::response_to_turn(response)
    }
}

impl<Ext, H> GenericCompletionModel<Ext, H> {
    /// Creates a codec from the model's request defaults.
    pub(crate) fn completion_codec(&self) -> ChatCompletionCodec {
        ChatCompletionCodec::new(
            self.model.clone(),
            self.strict_tools,
            self.tool_result_array_content,
        )
    }
}
