//! Codecs for the OpenAI Responses API.

use crate::{
    OneOrMany,
    completion::{
        self, CompletionCodec, CompletionError, CompletionRequest as CoreCompletionRequest,
        CompletionResponseCodec, ModelTurn,
    },
};

use super::{
    CompletionRequest, CompletionResponse, GenericResponsesCompletionModel, Output,
    ResponsesToolDefinition,
};

/// Encodes and decodes OpenAI Responses API completion payloads.
#[derive(Clone, Debug)]
pub(crate) struct ResponsesCompletionCodec {
    model: String,
    tools: Vec<ResponsesToolDefinition>,
    stream: Option<bool>,
}

impl ResponsesCompletionCodec {
    /// Creates a codec for one OpenAI Responses API model configuration.
    pub(crate) fn new(model: impl Into<String>, tools: Vec<ResponsesToolDefinition>) -> Self {
        Self {
            model: model.into(),
            tools,
            stream: None,
        }
    }

    /// Configures request encoding for the streaming endpoint.
    pub(crate) fn streaming(mut self) -> Self {
        self.stream = Some(true);
        self
    }

    /// Decodes a non-streaming Responses API payload into a normalized model turn.
    pub(crate) fn response_to_turn(
        response: CompletionResponse,
    ) -> Result<ModelTurn<CompletionResponse>, CompletionError> {
        if response.output.is_empty() {
            return Err(CompletionError::ResponseError(
                "Response contained no parts".to_owned(),
            ));
        }

        let message_id = response.output.iter().find_map(|item| match item {
            Output::Message(msg) => Some(msg.id.clone()),
            _ => None,
        });

        let content: Vec<completion::AssistantContent> = response
            .output
            .iter()
            .cloned()
            .flat_map(<Vec<completion::AssistantContent>>::from)
            .collect();

        let choice = OneOrMany::from_iter_optional(content).ok_or_else(|| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: usage.total_tokens,
                cached_input_tokens: usage
                    .input_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens)
                    .unwrap_or(0),
                cache_creation_input_tokens: 0,
            })
            .unwrap_or_default();

        Ok(ModelTurn::new(choice, usage)
            .with_message_id(message_id)
            .with_raw_response(response))
    }
}

impl CompletionCodec for ResponsesCompletionCodec {
    type Request = CompletionRequest;

    fn encode_request(
        &self,
        request: CoreCompletionRequest,
    ) -> Result<Self::Request, CompletionError> {
        let mut request = CompletionRequest::from_model(self.model.clone(), request)?;
        request.tools.extend(self.tools.clone());

        if self.stream.is_some() {
            request.stream = self.stream;
        }

        Ok(request)
    }
}

impl CompletionResponseCodec for ResponsesCompletionCodec {
    type Response = CompletionResponse;
    type RawFinal = CompletionResponse;

    fn decode_response(
        &self,
        response: Self::Response,
    ) -> Result<ModelTurn<Self::RawFinal>, CompletionError> {
        Self::response_to_turn(response)
    }
}

impl<Ext, H> GenericResponsesCompletionModel<Ext, H> {
    /// Creates a codec from the model's request defaults.
    pub(crate) fn completion_codec(&self) -> ResponsesCompletionCodec {
        ResponsesCompletionCodec::new(self.model.clone(), self.tools.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        message::Text,
        providers::openai::responses_api::{
            AdditionalParameters, AssistantContent, InputTokensDetails, OutputMessage, OutputRole,
            OutputTokensDetails, ResponseObject, ResponseStatus, ResponsesUsage,
        },
    };

    fn core_request() -> CoreCompletionRequest {
        CoreCompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one("hello".into()),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        }
    }

    fn usage() -> ResponsesUsage {
        ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: Some(InputTokensDetails { cached_tokens: 4 }),
            output_tokens: 5,
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 0,
            },
            total_tokens: 15,
        }
    }

    fn response_with_output(output: Vec<Output>) -> CompletionResponse {
        CompletionResponse {
            id: "resp_123".to_string(),
            object: ResponseObject::Response,
            created_at: 0,
            status: ResponseStatus::Completed,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-5".to_string(),
            usage: Some(usage()),
            output,
            tools: Vec::new(),
            additional_parameters: AdditionalParameters::default(),
        }
    }

    #[test]
    fn encode_request_can_force_streaming() {
        let codec = ResponsesCompletionCodec::new("gpt-5", Vec::new()).streaming();

        let request = codec
            .encode_request(core_request())
            .expect("request should encode");

        assert_eq!(request.stream, Some(true));
    }

    #[test]
    fn decode_response_returns_normalized_turn() {
        let response = response_with_output(vec![Output::Message(OutputMessage {
            id: "msg_123".to_string(),
            role: OutputRole::Assistant,
            status: ResponseStatus::Completed,
            content: vec![AssistantContent::OutputText(Text {
                text: "hello world".to_string(),
            })],
        })]);

        let turn = ResponsesCompletionCodec::response_to_turn(response)
            .expect("response should decode to a model turn");

        assert_eq!(turn.visible_text(), "hello world");
        assert_eq!(turn.message_id.as_deref(), Some("msg_123"));
        assert_eq!(turn.usage.input_tokens, 10);
        assert_eq!(turn.usage.cached_input_tokens, 4);
        assert!(turn.raw_response.is_some());
    }

    #[test]
    fn decode_response_rejects_empty_output() {
        let err = ResponsesCompletionCodec::response_to_turn(response_with_output(Vec::new()))
            .expect_err("empty output should fail");

        assert!(err.to_string().contains("Response contained no parts"));
    }
}
