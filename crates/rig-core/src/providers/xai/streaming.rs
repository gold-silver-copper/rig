//! xAI Responses API Streaming
//!
//! This module reuses OpenAI's Responses API streaming types since xAI's API
//! is designed to be compatible with OpenAI's format.

use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::GenericEventSource;
use crate::json_utils;
use crate::providers::openai::responses_api::streaming::{
    ResponsesStreamOptions, StreamingResponse, stream_from_event_source_with_options,
};
use crate::providers::xai::completion::{CompletionModel, XAICompletionRequest};

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    pub(crate) async fn stream_events(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<crate::model_event::ModelEventStream<StreamingResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let mut request =
            XAICompletionRequest::try_from((self.model.as_str(), completion_request))?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true}),
        );

        request.additional_params = Some(params);

        if enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "xAI streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/v1/responses")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "xai",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        send_xai_streaming_request(self.client.clone(), req)
            .instrument(span)
            .await
    }
}

/// Send a streaming request
pub(crate) async fn send_xai_streaming_request<T>(
    http_client: T,
    req: http::Request<Vec<u8>>,
) -> Result<crate::model_event::ModelEventStream<StreamingResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    let span = tracing::Span::current();
    let event_source = GenericEventSource::new(http_client, req);

    Ok(stream_from_event_source_with_options(
        event_source,
        span,
        "xAI",
        ResponsesStreamOptions::strict_with_immediate_tool_calls(),
    ))
}

#[cfg(test)]
mod tests {
    use super::send_xai_streaming_request;
    use crate::message::ReasoningContent;
    use crate::model_event::ModelEvent;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_json_events;
    use crate::providers::openai::responses_api::ReasoningSummary;
    use crate::providers::openai::responses_api::streaming::reasoning_choices_from_done_item;

    #[test]
    fn reasoning_done_item_emits_summary_then_encrypted() {
        let summary = vec![
            ReasoningSummary::SummaryText {
                text: "s1".to_string(),
            },
            ReasoningSummary::SummaryText {
                text: "s2".to_string(),
            },
        ];
        let choices: Vec<ModelEvent<()>> =
            reasoning_choices_from_done_item("xr_1", &summary, Some("enc"));

        assert_eq!(choices.len(), 3);
        assert!(matches!(
            choices.first(),
            Some(ModelEvent::ReasoningDone { reasoning })
                if reasoning.id.as_deref() == Some("xr_1")
                    && matches!(
                        reasoning.content.first(),
                        Some(ReasoningContent::Summary(text)) if text == "s1"
                    )
        ));
        assert!(matches!(
            choices.get(1),
            Some(ModelEvent::ReasoningDone { reasoning })
                if reasoning.id.as_deref() == Some("xr_1")
                    && matches!(
                        reasoning.content.first(),
                        Some(ReasoningContent::Summary(text)) if text == "s2"
                    )
        ));
        assert!(matches!(
            choices.get(2),
            Some(ModelEvent::ReasoningDone { reasoning })
                if reasoning.id.as_deref() == Some("xr_1")
                    && matches!(
                        reasoning.content.first(),
                        Some(ReasoningContent::Encrypted(data)) if data == "enc"
                    )
        ));
    }

    #[tokio::test]
    async fn xai_stream_surfaces_terminal_errors_after_completed_tool_calls() {
        use crate::http_client::mock::MockStreamingClient;
        use futures::StreamExt;
        use serde_json::json;

        let tool_call_done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "arguments": "{}",
                "call_id": "call_123",
                "name": "example_tool",
                "status": "completed"
            }
        });

        let failed = json!({
            "type": "response.failed",
            "sequence_number": 2,
            "response": {
                "id": "resp_123",
                "object": "response",
                "created_at": 0,
                "status": "failed",
                "error": {
                    "code": "server_error",
                    "message": "response stream failed"
                },
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "grok-4-0709",
                "usage": null,
                "output": [],
                "tools": []
            }
        });

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&[tool_call_done, failed]),
        };
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/responses")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_xai_streaming_request(client, req)
            .await
            .expect("stream should start");

        match stream.next().await.expect("stream should yield an item") {
            ModelEvent::ToolCallDone { tool_call, .. } => {
                assert_eq!(tool_call.id, "fc_123");
                assert_eq!(tool_call.call_id.as_deref(), Some("call_123"));
                assert_eq!(tool_call.function.name, "example_tool");
                assert_eq!(tool_call.function.arguments, serde_json::json!({}));
            }
            other => panic!("expected completed tool call before terminal error, got {other:?}"),
        }

        let err = stream
            .next()
            .await
            .expect("stream should yield terminal error");
        let ModelEvent::Error { error: err } = err else {
            panic!("stream should surface a provider error");
        };
        assert_eq!(
            err.to_string(),
            "ProviderError: server_error: response stream failed"
        );
        assert!(
            stream.next().await.is_none(),
            "stream should terminate immediately after the first terminal error"
        );
    }
}
