use std::collections::HashMap;

use crate::completion::{self, CompletionError, CompletionRequest as CoreCompletionRequest};
use crate::json_utils;
use crate::message;
use crate::streaming::{RawStreamingChoice, RawStreamingToolCall, ToolCallDeltaContent};

pub(crate) fn first_choice<T>(choices: &[T]) -> Result<&T, CompletionError> {
    choices
        .first()
        .ok_or_else(|| CompletionError::ResponseError("Response contained no choices".to_owned()))
}

pub(crate) fn map_finish_reason(reason: &str) -> completion::StopReason {
    match reason {
        "stop" => completion::StopReason::Stop,
        "tool_calls" => completion::StopReason::ToolCalls,
        "content_filter" => completion::StopReason::ContentFilter,
        "length" => completion::StopReason::MaxTokens,
        other => completion::StopReason::Other(other.to_string()),
    }
}

pub(crate) fn non_empty_text(text: impl AsRef<str>) -> Option<completion::AssistantContent> {
    let text = text.as_ref();
    if text.is_empty() {
        None
    } else {
        Some(completion::AssistantContent::text(text))
    }
}

pub(crate) fn build_completion_response<R, C>(
    raw_response: R,
    usage: completion::Usage,
    message_id: Option<String>,
    stop_reason: Option<completion::StopReason>,
    choice: C,
) -> completion::CompletionResponse<R>
where
    C: Into<completion::AssistantChoice>,
{
    completion::CompletionResponse {
        choice: choice.into(),
        usage,
        raw_response,
        message_id,
        stop_reason,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolsPolicy {
    Supported,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolChoicePolicy {
    PassThrough,
    Unsupported,
    RejectRequired,
    CoerceRequiredToAuto { steering_message: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputSchemaPolicy {
    Unsupported,
    NativeResponseFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdditionalParamsPolicy {
    PassThrough,
    Unsupported,
}

/// Shared unsupported-feature policy for request builders.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CompatibleFeaturePolicy {
    tools_policy: ToolsPolicy,
    tool_choice_policy: ToolChoicePolicy,
    output_schema_policy: OutputSchemaPolicy,
    additional_params_policy: AdditionalParamsPolicy,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CompatibleRequestAdjustments {
    steering_user_message: Option<&'static str>,
}

impl Default for CompatibleFeaturePolicy {
    fn default() -> Self {
        Self {
            tools_policy: ToolsPolicy::Supported,
            tool_choice_policy: ToolChoicePolicy::PassThrough,
            output_schema_policy: OutputSchemaPolicy::Unsupported,
            additional_params_policy: AdditionalParamsPolicy::PassThrough,
        }
    }
}

impl CompatibleFeaturePolicy {
    pub(crate) const fn with_tools_policy(mut self, policy: ToolsPolicy) -> Self {
        self.tools_policy = policy;
        self
    }

    pub(crate) const fn with_tool_choice_policy(mut self, policy: ToolChoicePolicy) -> Self {
        self.tool_choice_policy = policy;
        self
    }

    pub(crate) const fn with_output_schema_policy(mut self, policy: OutputSchemaPolicy) -> Self {
        self.output_schema_policy = policy;
        self
    }

    pub(crate) const fn with_additional_params_policy(
        mut self,
        policy: AdditionalParamsPolicy,
    ) -> Self {
        self.additional_params_policy = policy;
        self
    }

    pub(crate) fn apply(
        self,
        provider_name: &'static str,
        req: &mut CoreCompletionRequest,
    ) -> Result<CompatibleRequestAdjustments, CompletionError> {
        if req.output_schema.is_some()
            && matches!(self.output_schema_policy, OutputSchemaPolicy::Unsupported)
        {
            tracing::warn!(
                "Structured outputs currently not supported for {}",
                provider_name
            );
            req.output_schema = None;
        }

        if !req.tools.is_empty() && matches!(self.tools_policy, ToolsPolicy::Unsupported) {
            tracing::warn!("WARNING: `tools` not supported on {}", provider_name);
            req.tools.clear();
        }

        let mut adjustments = CompatibleRequestAdjustments::default();
        if let Some(choice) = req.tool_choice.clone() {
            match self.tool_choice_policy {
                ToolChoicePolicy::PassThrough => {}
                ToolChoicePolicy::Unsupported => {
                    tracing::warn!("WARNING: `tool_choice` not supported on {}", provider_name);
                    req.tool_choice = None;
                }
                ToolChoicePolicy::RejectRequired => {
                    if matches!(choice, crate::message::ToolChoice::Required) {
                        return Err(CompletionError::RequestError(
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                format!("{provider_name} does not support tool_choice=required"),
                            )
                            .into(),
                        ));
                    }
                }
                ToolChoicePolicy::CoerceRequiredToAuto { steering_message } => {
                    if matches!(choice, crate::message::ToolChoice::Required) {
                        tracing::warn!(
                            "{} does not support tool_choice=required; coercing to auto with an additional steering message",
                            provider_name
                        );
                        req.tool_choice = Some(crate::message::ToolChoice::Auto);
                        adjustments.steering_user_message = Some(steering_message);
                    }
                }
            }
        }

        if req.additional_params.is_some()
            && matches!(
                self.additional_params_policy,
                AdditionalParamsPolicy::Unsupported
            )
        {
            tracing::warn!(
                "WARNING: `additional_params` not supported on {}",
                provider_name
            );
            req.additional_params = None;
        }

        Ok(adjustments)
    }
}

/// Shared request-shaping profile for OpenAI-compatible chat providers.
#[derive(Debug, Clone, Copy)]
pub struct CompatibleChatProfile {
    provider_name: &'static str,
    require_messages: bool,
    feature_policy: CompatibleFeaturePolicy,
}

impl CompatibleChatProfile {
    pub(crate) const fn new(provider_name: &'static str) -> Self {
        Self {
            provider_name,
            require_messages: false,
            feature_policy: CompatibleFeaturePolicy {
                tools_policy: ToolsPolicy::Supported,
                tool_choice_policy: ToolChoicePolicy::PassThrough,
                output_schema_policy: OutputSchemaPolicy::Unsupported,
                additional_params_policy: AdditionalParamsPolicy::PassThrough,
            },
        }
    }

    pub(crate) const fn openai_chat_completions(provider_name: &'static str) -> Self {
        Self::new(provider_name)
            .require_messages()
            .native_response_format()
    }

    pub(crate) const fn require_messages(mut self) -> Self {
        self.require_messages = true;
        self
    }

    pub(crate) const fn native_response_format(mut self) -> Self {
        self.feature_policy = self
            .feature_policy
            .with_output_schema_policy(OutputSchemaPolicy::NativeResponseFormat);
        self
    }

    pub(crate) const fn unsupported_tools(mut self) -> Self {
        self.feature_policy = self
            .feature_policy
            .with_tools_policy(ToolsPolicy::Unsupported);
        self
    }

    pub(crate) const fn unsupported_tool_choice(mut self) -> Self {
        self.feature_policy = self
            .feature_policy
            .with_tool_choice_policy(ToolChoicePolicy::Unsupported);
        self
    }

    pub(crate) const fn reject_required_tool_choice(mut self) -> Self {
        self.feature_policy = self
            .feature_policy
            .with_tool_choice_policy(ToolChoicePolicy::RejectRequired);
        self
    }

    pub(crate) const fn coerce_required_tool_choice_to_auto(
        mut self,
        steering_message: &'static str,
    ) -> Self {
        self.feature_policy = self
            .feature_policy
            .with_tool_choice_policy(ToolChoicePolicy::CoerceRequiredToAuto { steering_message });
        self
    }
}

/// Provider-agnostic core request fields shared by OpenAI-compatible chat families.
#[derive(Debug)]
pub(crate) struct CompatibleRequestCore<M> {
    pub model: String,
    pub messages: Vec<M>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub tools: Vec<crate::completion::ToolDefinition>,
    pub tool_choice: Option<crate::message::ToolChoice>,
    pub additional_params: Option<serde_json::Value>,
}

fn merge_native_response_format(
    additional_params: Option<serde_json::Value>,
    schema: schemars::Schema,
) -> Option<serde_json::Value> {
    let name = schema
        .as_object()
        .and_then(|o| o.get("title"))
        .and_then(|v| v.as_str())
        .unwrap_or("response_schema")
        .to_string();
    let mut schema_value = schema.to_value();
    super::super::sanitize_schema(&mut schema_value);
    let response_format = serde_json::json!({
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": true,
                "schema": schema_value
            }
        }
    });

    Some(match additional_params {
        Some(existing) => json_utils::merge(existing, response_format),
        None => response_format,
    })
}

pub(crate) fn build_compatible_request_core<M, F>(
    default_model: &str,
    mut req: CoreCompletionRequest,
    profile: CompatibleChatProfile,
    system_message: fn(&str) -> M,
    user_message: Option<fn(&str) -> M>,
    message_has_tool_result: fn(&M) -> bool,
    mut convert_message: F,
) -> Result<CompatibleRequestCore<M>, CompletionError>
where
    F: FnMut(message::Message) -> Result<Vec<M>, CompletionError>,
{
    let adjustments = profile
        .feature_policy
        .apply(profile.provider_name, &mut req)?;

    let normalized_documents = req.normalized_documents();
    let CoreCompletionRequest {
        model: request_model,
        preamble,
        chat_history,
        tools,
        temperature,
        max_tokens,
        tool_choice,
        additional_params,
        output_schema,
        ..
    } = req;

    let mut partial_history = Vec::new();
    if let Some(docs) = normalized_documents {
        partial_history.push(docs);
    }
    partial_history.extend(chat_history);

    let mut messages = preamble
        .as_deref()
        .map_or_else(Vec::new, |preamble| vec![system_message(preamble)]);

    messages.extend(
        partial_history
            .into_iter()
            .map(&mut convert_message)
            .collect::<Result<Vec<Vec<M>>, _>>()?
            .into_iter()
            .flatten(),
    );

    if let Some(steering_message) = adjustments.steering_user_message {
        let Some(user_message) = user_message else {
            return Err(CompletionError::RequestError(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "{} requires a user-message adapter when coercing tool_choice",
                        profile.provider_name
                    ),
                )
                .into(),
            ));
        };
        messages.push(user_message(steering_message));
    }

    if profile.require_messages && messages.is_empty() {
        return Err(CompletionError::RequestError(
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "{} request has no provider-compatible messages after conversion",
                    profile.provider_name
                ),
            )
            .into(),
        ));
    }

    let mut additional_params = additional_params;
    if let Some(schema) = output_schema {
        match profile.feature_policy.output_schema_policy {
            OutputSchemaPolicy::NativeResponseFormat => {
                let should_apply_response_format =
                    tools.is_empty() || messages.iter().any(message_has_tool_result);
                if should_apply_response_format {
                    additional_params = merge_native_response_format(additional_params, schema);
                }
            }
            OutputSchemaPolicy::Unsupported => {}
        }
    }

    Ok(CompatibleRequestCore {
        model: request_model.unwrap_or_else(|| default_model.to_owned()),
        messages,
        temperature,
        max_tokens,
        tools,
        tool_choice,
        additional_params,
    })
}

/// Internal provider profile used by the OpenAI-compatible chat family.
#[doc(hidden)]
pub trait OpenAiChatProviderProfile {
    fn provider_name() -> &'static str;

    fn telemetry_provider_name() -> &'static str {
        Self::provider_name()
    }

    fn completions_path() -> &'static str {
        "/chat/completions"
    }

    fn request_profile() -> CompatibleChatProfile {
        CompatibleChatProfile::openai_chat_completions(Self::provider_name())
    }

    fn stream_tool_call_conflict_policy() -> ToolCallConflictPolicy {
        ToolCallConflictPolicy::EvictDistinctIdAndName
    }
}

impl OpenAiChatProviderProfile for super::super::client::OpenAICompletionsExt {
    fn provider_name() -> &'static str {
        "OpenAI Chat Completions"
    }

    fn telemetry_provider_name() -> &'static str {
        "openai"
    }
}

impl OpenAiChatProviderProfile for crate::providers::minimax::MiniMaxExt {
    fn provider_name() -> &'static str {
        "MiniMax"
    }

    fn telemetry_provider_name() -> &'static str {
        "minimax"
    }
}

impl OpenAiChatProviderProfile for crate::providers::zai::ZAiExt {
    fn provider_name() -> &'static str {
        "Z.AI"
    }

    fn telemetry_provider_name() -> &'static str {
        "z.ai"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallConflictPolicy {
    KeepIndex,
    EvictDistinctIdAndName,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompatibleStreamingToolCall<'a> {
    pub index: usize,
    pub id: Option<&'a str>,
    pub name: Option<&'a str>,
    pub arguments: Option<&'a str>,
}

pub(crate) fn apply_compatible_tool_call_deltas<'a, R>(
    tool_calls: &mut HashMap<usize, RawStreamingToolCall>,
    incoming: impl IntoIterator<Item = CompatibleStreamingToolCall<'a>>,
    conflict_policy: ToolCallConflictPolicy,
) -> Vec<RawStreamingChoice<R>>
where
    R: Clone,
{
    let mut events = Vec::new();

    for tool_call in incoming {
        let index = tool_call.index;

        if conflict_policy == ToolCallConflictPolicy::EvictDistinctIdAndName
            && should_evict_existing_tool_call(tool_calls.get(&index), tool_call.id, tool_call.name)
            && let Some(evicted) = tool_calls.remove(&index)
        {
            events.push(RawStreamingChoice::ToolCall(
                finalize_completed_streaming_tool_call(evicted),
            ));
        }

        let existing_tool_call = tool_calls
            .entry(index)
            .or_insert_with(RawStreamingToolCall::empty);

        if let Some(id) = tool_call.id
            && !id.is_empty()
        {
            existing_tool_call.id = id.to_owned();
        }

        if let Some(name) = tool_call.name
            && !name.is_empty()
        {
            existing_tool_call.name = name.to_owned();
            events.push(RawStreamingChoice::ToolCallDelta {
                id: existing_tool_call.id.clone(),
                internal_call_id: existing_tool_call.internal_call_id.clone(),
                content: ToolCallDeltaContent::Name(name.to_owned()),
            });
        }

        if let Some(chunk) = tool_call.arguments
            && !chunk.is_empty()
        {
            append_tool_call_arguments(&mut existing_tool_call.arguments, chunk);
            events.push(RawStreamingChoice::ToolCallDelta {
                id: existing_tool_call.id.clone(),
                internal_call_id: existing_tool_call.internal_call_id.clone(),
                content: ToolCallDeltaContent::Delta(chunk.to_owned()),
            });
        }
    }

    events
}

pub(crate) fn take_finalized_tool_calls<R>(
    tool_calls: &mut HashMap<usize, RawStreamingToolCall>,
) -> Vec<RawStreamingChoice<R>>
where
    R: Clone,
{
    std::mem::take(tool_calls)
        .into_values()
        .map(|tool_call| {
            RawStreamingChoice::ToolCall(finalize_completed_streaming_tool_call(tool_call))
        })
        .collect()
}

pub(crate) fn take_tool_calls<R>(
    tool_calls: &mut HashMap<usize, RawStreamingToolCall>,
) -> Vec<RawStreamingChoice<R>>
where
    R: Clone,
{
    std::mem::take(tool_calls)
        .into_values()
        .map(RawStreamingChoice::ToolCall)
        .collect()
}

pub(crate) fn finalize_completed_streaming_tool_call(
    mut tool_call: RawStreamingToolCall,
) -> RawStreamingToolCall {
    if tool_call.arguments.is_null() {
        tool_call.arguments = serde_json::Value::Object(serde_json::Map::new());
    }

    tool_call
}

fn should_evict_existing_tool_call(
    existing: Option<&RawStreamingToolCall>,
    new_id: Option<&str>,
    new_name: Option<&str>,
) -> bool {
    let Some(existing) = existing else {
        return false;
    };

    let Some(new_id) = new_id.filter(|id| !id.is_empty()) else {
        return false;
    };
    let Some(new_name) = new_name.filter(|name| !name.is_empty()) else {
        return false;
    };

    !existing.id.is_empty()
        && existing.id != new_id
        && !existing.name.is_empty()
        && existing.name != new_name
}

fn append_tool_call_arguments(arguments: &mut serde_json::Value, chunk: &str) {
    let current_arguments = match arguments {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(value) => value.clone(),
        ref value => value.to_string(),
    };
    let combined = format!("{current_arguments}{chunk}");

    if combined.trim_start().starts_with('{') && combined.trim_end().ends_with('}') {
        match serde_json::from_str(&combined) {
            Ok(parsed) => *arguments = parsed,
            Err(_) => *arguments = serde_json::Value::String(combined),
        }
    } else {
        *arguments = serde_json::Value::String(combined);
    }
}

#[cfg(test)]
pub(crate) mod request_conformance {
    use serde_json::{Value, json};

    use super::{
        AdditionalParamsPolicy, CompatibleChatProfile, OutputSchemaPolicy, ToolChoicePolicy,
        ToolsPolicy,
    };
    use crate::completion::{CompletionError, CompletionRequest, Document, ToolDefinition};
    use crate::message::{self, Message, ToolChoice, UserContent};
    use crate::{OneOrMany, completion};

    #[derive(Debug, Clone, Copy)]
    pub(crate) enum Fixture {
        ModelOverridePrecedence,
        PreambleDocumentHistoryOrdering,
        EmptyMessageRejection,
        UnsupportedFieldStripping,
        ToolChoiceCoercion,
        OutputSchemaHandling,
        AdditionalParamsPassthrough,
        AdditionalParamsDrop,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub(crate) enum Outcome<T> {
        Supported(T),
        RequestError,
    }

    pub(crate) trait Harness {
        #[allow(dead_code)]
        fn family_name() -> &'static str;

        fn run(case: Fixture) -> Outcome<Value>;

        fn assert(case: Fixture, actual: Outcome<Value>);
    }

    pub(crate) fn assert_case<H: Harness>(case: Fixture) {
        let actual = H::run(case);
        H::assert(case, actual);
    }

    pub(crate) fn serialize_request<T: serde::Serialize>(request: &T) -> Value {
        serde_json::to_value(request).expect("request should serialize")
    }

    pub(crate) fn serialize_case<T, F>(case: Fixture, convert: F) -> Outcome<Value>
    where
        T: serde::Serialize,
        F: FnOnce(CompletionRequest) -> Result<T, CompletionError>,
    {
        match convert(fixture_request(case)) {
            Ok(request) => Outcome::Supported(serialize_request(&request)),
            Err(CompletionError::RequestError(_)) => Outcome::RequestError,
            Err(err) => panic!("unexpected request conversion error for {case:?}: {err}"),
        }
    }

    pub(crate) fn fixture_request(case: Fixture) -> CompletionRequest {
        match case {
            Fixture::ModelOverridePrecedence => CompletionRequest {
                model: Some("override-model".to_owned()),
                preamble: None,
                chat_history: OneOrMany::one(Message::User {
                    content: OneOrMany::one(UserContent::text("hello")),
                }),
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
                output_schema: None,
            },
            Fixture::PreambleDocumentHistoryOrdering => CompletionRequest {
                model: None,
                preamble: Some("system".to_owned()),
                chat_history: OneOrMany::one(Message::User {
                    content: OneOrMany::one(UserContent::text("hello")),
                }),
                documents: vec![Document {
                    id: "doc-1".to_owned(),
                    text: "Document body".to_owned(),
                    additional_props: Default::default(),
                }],
                tools: Vec::new(),
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
                output_schema: None,
            },
            Fixture::EmptyMessageRejection => CompletionRequest {
                model: None,
                preamble: None,
                chat_history: OneOrMany::one(Message::Assistant {
                    id: None,
                    content: OneOrMany::one(message::AssistantContent::reasoning("hidden")),
                }),
                documents: vec![],
                tools: vec![],
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: None,
                output_schema: None,
            },
            Fixture::UnsupportedFieldStripping => CompletionRequest {
                model: None,
                preamble: None,
                chat_history: OneOrMany::one(Message::User {
                    content: OneOrMany::one(UserContent::text("hello")),
                }),
                documents: Vec::new(),
                tools: vec![ToolDefinition {
                    name: "lookup_weather".to_owned(),
                    description: "Lookup weather".to_owned(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" }
                        }
                    }),
                }],
                temperature: None,
                max_tokens: None,
                tool_choice: Some(ToolChoice::Auto),
                additional_params: Some(json!({"vendor_flag": true})),
                output_schema: Some(sample_schema()),
            },
            Fixture::ToolChoiceCoercion => CompletionRequest {
                model: None,
                preamble: None,
                chat_history: OneOrMany::one(Message::User {
                    content: OneOrMany::one(UserContent::text("hello")),
                }),
                documents: Vec::new(),
                tools: vec![ToolDefinition {
                    name: "lookup_weather".to_owned(),
                    description: "Lookup weather".to_owned(),
                    parameters: json!({"type":"object"}),
                }],
                temperature: None,
                max_tokens: None,
                tool_choice: Some(ToolChoice::Required),
                additional_params: None,
                output_schema: None,
            },
            Fixture::OutputSchemaHandling => CompletionRequest {
                model: None,
                preamble: None,
                chat_history: OneOrMany::one(Message::User {
                    content: OneOrMany::one(UserContent::text("hello")),
                }),
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: Some(json!({"vendor_flag": true})),
                output_schema: Some(sample_schema()),
            },
            Fixture::AdditionalParamsPassthrough | Fixture::AdditionalParamsDrop => {
                CompletionRequest {
                    model: None,
                    preamble: None,
                    chat_history: OneOrMany::one(Message::User {
                        content: OneOrMany::one(UserContent::text("hello")),
                    }),
                    documents: Vec::new(),
                    tools: Vec::new(),
                    temperature: None,
                    max_tokens: None,
                    tool_choice: None,
                    additional_params: Some(json!({"vendor_flag": true})),
                    output_schema: None,
                }
            }
        }
    }

    pub(crate) fn model(value: &Value) -> Option<&str> {
        value.get("model").and_then(Value::as_str)
    }

    pub(crate) fn tools_len(value: &Value) -> usize {
        value
            .get("tools")
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
    }

    pub(crate) fn tool_choice(value: &Value) -> Option<&Value> {
        value.get("tool_choice")
    }

    pub(crate) fn top_level<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
        value.get(key)
    }

    pub(crate) fn tool_choice_text(value: &Value) -> Option<&str> {
        value.get("tool_choice").and_then(Value::as_str)
    }

    pub(crate) fn message_summaries(value: &Value) -> Vec<String> {
        value
            .get("messages")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|message| {
                let role = message
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                format!("{role}:{}", extract_text(message))
            })
            .collect()
    }

    fn extract_text(message: &Value) -> String {
        let content = extract_text_from_value(message.get("content").unwrap_or(&Value::Null));
        if !content.is_empty() {
            return content;
        }

        if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str) {
            return reasoning.to_owned();
        }

        extract_text_from_value(message.get("reasoning_details").unwrap_or(&Value::Null))
    }

    fn extract_text_from_value(value: &Value) -> String {
        match value {
            Value::String(text) => text.clone(),
            Value::Array(items) => items
                .iter()
                .filter_map(extract_text_part)
                .collect::<Vec<_>>()
                .join("\n"),
            Value::Object(object) => {
                if let Some(text) = object.get("text").and_then(Value::as_str) {
                    return text.to_owned();
                }

                if let Some(text) = object.get("content").and_then(Value::as_str) {
                    return text.to_owned();
                }

                if let Some(content) = object.get("content") {
                    return extract_text_from_value(content);
                }

                String::new()
            }
            _ => String::new(),
        }
    }

    fn extract_text_part(value: &Value) -> Option<String> {
        match value {
            Value::String(text) => Some(text.clone()),
            Value::Object(object) => object
                .get("text")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .or_else(|| {
                    object
                        .get("content")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                })
                .or_else(|| object.get("content").map(extract_text_from_value))
                .filter(|text| !text.is_empty()),
            _ => None,
        }
    }

    pub(crate) fn expected_document_text() -> String {
        completion::Document {
            id: "doc-1".to_owned(),
            text: "Document body".to_owned(),
            additional_props: Default::default(),
        }
        .to_string()
    }

    fn sample_schema() -> schemars::Schema {
        serde_json::from_value(json!({
            "title": "fixture_schema",
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            }
        }))
        .expect("schema should deserialize")
    }

    #[derive(Debug, Clone, Copy)]
    pub(crate) struct CompatibleChatExpectation {
        profile: CompatibleChatProfile,
        preserves_reasoning_assistant_history: bool,
        includes_document_messages: bool,
    }

    impl CompatibleChatExpectation {
        pub(crate) const fn new(profile: CompatibleChatProfile) -> Self {
            Self {
                profile,
                preserves_reasoning_assistant_history: false,
                includes_document_messages: true,
            }
        }

        pub(crate) const fn preserves_reasoning_assistant_history(mut self) -> Self {
            self.preserves_reasoning_assistant_history = true;
            self
        }

        pub(crate) const fn omits_document_messages(mut self) -> Self {
            self.includes_document_messages = false;
            self
        }
    }

    pub(crate) fn assert_compatible_chat_case(
        expectation: CompatibleChatExpectation,
        default_model: &str,
        case: Fixture,
        actual: Outcome<Value>,
    ) {
        let profile = expectation.profile;
        match case {
            Fixture::ModelOverridePrecedence => {
                let request = expect_supported(case, actual);
                assert_eq!(model(&request), Some("override-model"));
            }
            Fixture::PreambleDocumentHistoryOrdering => {
                let request = expect_supported(case, actual);
                let mut expected = vec!["system:system".to_owned()];
                if expectation.includes_document_messages {
                    expected.push(format!("user:{}", expected_document_text()));
                }
                expected.push("user:hello".to_owned());
                assert_eq!(message_summaries(&request), expected,);
            }
            Fixture::EmptyMessageRejection => {
                if profile.require_messages {
                    assert!(
                        matches!(actual, Outcome::RequestError),
                        "expected request rejection for {case:?}, got {actual:?}"
                    );
                } else {
                    let request = expect_supported(case, actual);
                    let summaries = message_summaries(&request);
                    if expectation.preserves_reasoning_assistant_history {
                        assert_eq!(summaries, vec!["assistant:hidden".to_owned()]);
                    } else {
                        assert!(
                            summaries.is_empty(),
                            "expected no provider-compatible messages: {request:?}"
                        );
                    }
                }
            }
            Fixture::UnsupportedFieldStripping => {
                let request = expect_supported(case, actual);
                assert_eq!(model(&request), Some(default_model));
                assert_eq!(
                    tools_len(&request),
                    if matches!(profile.feature_policy.tools_policy, ToolsPolicy::Supported) {
                        1
                    } else {
                        0
                    }
                );
                assert_tool_choice(&request, profile.feature_policy.tool_choice_policy, "auto");
                assert_vendor_flag(
                    &request,
                    matches!(
                        profile.feature_policy.additional_params_policy,
                        AdditionalParamsPolicy::PassThrough
                    ),
                );
                assert!(
                    top_level(&request, "response_format").is_none(),
                    "tool-bearing turns should omit response_format: {request:?}"
                );
            }
            Fixture::ToolChoiceCoercion => {
                if matches!(
                    profile.feature_policy.tool_choice_policy,
                    ToolChoicePolicy::RejectRequired
                ) {
                    assert!(
                        matches!(actual, Outcome::RequestError),
                        "expected request rejection for {case:?}, got {actual:?}"
                    );
                    return;
                }

                let request = expect_supported(case, actual);
                match profile.feature_policy.tool_choice_policy {
                    ToolChoicePolicy::PassThrough => {
                        assert_eq!(tool_choice_text(&request), Some("required"));
                    }
                    ToolChoicePolicy::Unsupported => {
                        assert!(
                            tool_choice(&request).is_none(),
                            "expected tool_choice to be stripped: {request:?}"
                        );
                    }
                    ToolChoicePolicy::RejectRequired => unreachable!(),
                    ToolChoicePolicy::CoerceRequiredToAuto { steering_message } => {
                        assert_eq!(tool_choice_text(&request), Some("auto"));
                        let expected_message = format!("user:{steering_message}");
                        assert_eq!(
                            message_summaries(&request).last().map(String::as_str),
                            Some(expected_message.as_str()),
                        );
                    }
                }
            }
            Fixture::OutputSchemaHandling => {
                let request = expect_supported(case, actual);
                match profile.feature_policy.output_schema_policy {
                    OutputSchemaPolicy::NativeResponseFormat => {
                        assert_eq!(
                            top_level(&request, "response_format")
                                .and_then(|value| value.get("json_schema"))
                                .and_then(|value| value.get("name"))
                                .and_then(Value::as_str),
                            Some("fixture_schema"),
                        );
                    }
                    OutputSchemaPolicy::Unsupported => {
                        assert!(
                            top_level(&request, "response_format").is_none(),
                            "unexpected response_format: {request:?}"
                        );
                    }
                }
                assert_vendor_flag(
                    &request,
                    matches!(
                        profile.feature_policy.additional_params_policy,
                        AdditionalParamsPolicy::PassThrough
                    ),
                );
            }
            Fixture::AdditionalParamsPassthrough | Fixture::AdditionalParamsDrop => {
                let request = expect_supported(case, actual);
                assert_vendor_flag(
                    &request,
                    matches!(
                        profile.feature_policy.additional_params_policy,
                        AdditionalParamsPolicy::PassThrough
                    ),
                );
            }
        }
    }

    fn expect_supported(case: Fixture, actual: Outcome<Value>) -> Value {
        match actual {
            Outcome::Supported(request) => request,
            Outcome::RequestError => {
                panic!("expected supported request for {case:?}, got request rejection")
            }
        }
    }

    fn assert_vendor_flag(request: &Value, expected: bool) {
        if expected {
            assert_eq!(top_level(request, "vendor_flag"), Some(&json!(true)));
        } else {
            assert!(
                top_level(request, "vendor_flag").is_none(),
                "expected vendor_flag to be stripped: {request:?}"
            );
        }
    }

    fn assert_tool_choice(request: &Value, policy: ToolChoicePolicy, expected_pass_through: &str) {
        match policy {
            ToolChoicePolicy::PassThrough => {
                assert_eq!(tool_choice_text(request), Some(expected_pass_through));
            }
            ToolChoicePolicy::Unsupported => {
                assert!(
                    tool_choice(request).is_none(),
                    "expected tool_choice to be stripped: {request:?}"
                );
            }
            ToolChoicePolicy::RejectRequired => {
                assert_eq!(tool_choice_text(request), Some(expected_pass_through));
            }
            ToolChoicePolicy::CoerceRequiredToAuto { .. } => {
                assert_eq!(tool_choice_text(request), Some(expected_pass_through));
            }
        }
    }

    macro_rules! provider_request_conformance_tests {
        ($harness:ty) => {
            #[test]
            fn request_conformance_model_override_precedence() {
                crate::providers::openai::completion::request_conformance::assert_case::<$harness>(
                    crate::providers::openai::completion::request_conformance::Fixture::ModelOverridePrecedence,
                );
            }

            #[test]
            fn request_conformance_preamble_document_history_ordering() {
                crate::providers::openai::completion::request_conformance::assert_case::<$harness>(
                    crate::providers::openai::completion::request_conformance::Fixture::PreambleDocumentHistoryOrdering,
                );
            }

            #[test]
            fn request_conformance_empty_message_rejection() {
                crate::providers::openai::completion::request_conformance::assert_case::<$harness>(
                    crate::providers::openai::completion::request_conformance::Fixture::EmptyMessageRejection,
                );
            }

            #[test]
            fn request_conformance_unsupported_field_stripping() {
                crate::providers::openai::completion::request_conformance::assert_case::<$harness>(
                    crate::providers::openai::completion::request_conformance::Fixture::UnsupportedFieldStripping,
                );
            }

            #[test]
            fn request_conformance_tool_choice_coercion() {
                crate::providers::openai::completion::request_conformance::assert_case::<$harness>(
                    crate::providers::openai::completion::request_conformance::Fixture::ToolChoiceCoercion,
                );
            }

            #[test]
            fn request_conformance_output_schema_handling() {
                crate::providers::openai::completion::request_conformance::assert_case::<$harness>(
                    crate::providers::openai::completion::request_conformance::Fixture::OutputSchemaHandling,
                );
            }

            #[test]
            fn request_conformance_additional_params_passthrough() {
                crate::providers::openai::completion::request_conformance::assert_case::<$harness>(
                    crate::providers::openai::completion::request_conformance::Fixture::AdditionalParamsPassthrough,
                );
            }

            #[test]
            fn request_conformance_additional_params_drop() {
                crate::providers::openai::completion::request_conformance::assert_case::<$harness>(
                    crate::providers::openai::completion::request_conformance::Fixture::AdditionalParamsDrop,
                );
            }
        };
    }

    pub(crate) use provider_request_conformance_tests;
}

#[cfg(test)]
mod tests {
    use super::{
        AdditionalParamsPolicy, CompatibleChatProfile, CompatibleFeaturePolicy,
        CompatibleStreamingToolCall, OutputSchemaPolicy, ToolCallConflictPolicy, ToolChoicePolicy,
        ToolsPolicy, apply_compatible_tool_call_deltas, build_compatible_request_core,
        finalize_completed_streaming_tool_call, take_finalized_tool_calls,
    };
    use crate::OneOrMany;
    use crate::completion::{CompletionRequest, ToolDefinition};
    use crate::message::{Message, ToolChoice, UserContent};
    use crate::streaming::{RawStreamingChoice, RawStreamingToolCall};
    use std::collections::HashMap;

    #[test]
    fn requires_non_empty_messages_when_profile_demands_it() {
        let req = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::User {
                content: OneOrMany::one(UserContent::text("hello")),
            }),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let err = build_compatible_request_core(
            "test-model",
            req,
            CompatibleChatProfile::new("Example Provider").require_messages(),
            |preamble| preamble.to_owned(),
            None,
            |_| false,
            |_message| Ok(Vec::<String>::new()),
        )
        .expect_err("empty converted messages should fail");

        assert!(err.to_string().contains(
            "Example Provider request has no provider-compatible messages after conversion"
        ));
    }

    #[test]
    fn preserves_model_override_and_additional_params() {
        let req = CompletionRequest {
            model: Some("override-model".to_owned()),
            preamble: Some("system".to_owned()),
            chat_history: OneOrMany::one(Message::User {
                content: OneOrMany::one(UserContent::text("hello")),
            }),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: Some(0.5),
            max_tokens: Some(42),
            tool_choice: None,
            additional_params: Some(serde_json::json!({"foo":"bar"})),
            output_schema: None,
        };

        let result = build_compatible_request_core(
            "default-model",
            req,
            CompatibleChatProfile::new("Example Provider"),
            |preamble| preamble.to_owned(),
            None,
            |_| false,
            |message| Ok(vec![format!("{message:?}")]),
        )
        .expect("request conversion should succeed");

        assert_eq!(result.model, "override-model");
        assert_eq!(result.temperature, Some(0.5));
        assert_eq!(result.max_tokens, Some(42));
        assert_eq!(
            result.additional_params,
            Some(serde_json::json!({"foo":"bar"}))
        );
        assert_eq!(result.messages.len(), 2);
    }

    #[test]
    fn feature_policy_strips_unsupported_fields() {
        let mut req = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::User {
                content: OneOrMany::one(UserContent::text("hello")),
            }),
            documents: Vec::new(),
            tools: vec![ToolDefinition {
                name: "lookup_weather".to_owned(),
                description: "Lookup the weather".to_owned(),
                parameters: serde_json::json!({"type":"object"}),
            }],
            temperature: None,
            max_tokens: None,
            tool_choice: Some(ToolChoice::Auto),
            additional_params: Some(serde_json::json!({"foo":"bar"})),
            output_schema: Some(
                serde_json::from_value(serde_json::json!({
                    "title": "example",
                    "type": "object"
                }))
                .expect("schema should deserialize"),
            ),
        };

        CompatibleFeaturePolicy::default()
            .with_tools_policy(ToolsPolicy::Unsupported)
            .with_tool_choice_policy(ToolChoicePolicy::Unsupported)
            .with_output_schema_policy(OutputSchemaPolicy::Unsupported)
            .with_additional_params_policy(AdditionalParamsPolicy::Unsupported)
            .apply("Example Provider", &mut req)
            .expect("policy application should succeed");

        assert!(req.tools.is_empty());
        assert!(req.tool_choice.is_none());
        assert!(req.additional_params.is_none());
        assert!(req.output_schema.is_none());
    }

    #[test]
    fn tool_choice_policy_can_coerce_required_to_auto_with_steering_message() {
        let req = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::User {
                content: OneOrMany::one(UserContent::text("hello")),
            }),
            documents: Vec::new(),
            tools: vec![ToolDefinition {
                name: "lookup_weather".to_owned(),
                description: "Lookup the weather".to_owned(),
                parameters: serde_json::json!({"type":"object"}),
            }],
            temperature: None,
            max_tokens: None,
            tool_choice: Some(ToolChoice::Required),
            additional_params: None,
            output_schema: None,
        };

        let result = build_compatible_request_core(
            "test-model",
            req,
            CompatibleChatProfile::new("Example Provider")
                .coerce_required_tool_choice_to_auto("Use a tool."),
            |preamble| preamble.to_owned(),
            Some(|text| text.to_owned()),
            |_| false,
            |message| Ok(vec![format!("{message:?}")]),
        )
        .expect("request conversion should succeed");

        assert_eq!(result.tool_choice, Some(ToolChoice::Auto));
        assert_eq!(result.messages.last(), Some(&"Use a tool.".to_owned()));
    }

    #[test]
    fn native_response_format_merges_schema_into_additional_params() {
        let req = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::User {
                content: OneOrMany::one(UserContent::text("hello")),
            }),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: Some(serde_json::json!({"vendor_flag": true})),
            output_schema: Some(
                serde_json::from_value(serde_json::json!({
                    "title": "example",
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" }
                    }
                }))
                .expect("schema should deserialize"),
            ),
        };

        let result = build_compatible_request_core(
            "test-model",
            req,
            CompatibleChatProfile::new("Example Provider").native_response_format(),
            |preamble| preamble.to_owned(),
            None,
            |_| false,
            |message| Ok(vec![format!("{message:?}")]),
        )
        .expect("request conversion should succeed");

        let additional_params = result
            .additional_params
            .expect("response_format should be merged");
        assert_eq!(
            additional_params.get("vendor_flag"),
            Some(&serde_json::json!(true))
        );
        assert!(additional_params.get("response_format").is_some());
    }

    #[test]
    fn evicts_distinct_tool_calls_that_reuse_the_same_index() {
        let mut tool_calls = HashMap::from([(
            0,
            RawStreamingToolCall {
                id: "call_1".to_owned(),
                internal_call_id: "internal_1".to_owned(),
                call_id: None,
                name: "weather".to_owned(),
                arguments: serde_json::json!({"city":"Paris"}),
                signature: None,
                additional_params: None,
            },
        )]);

        let events = apply_compatible_tool_call_deltas::<()>(
            &mut tool_calls,
            [CompatibleStreamingToolCall {
                index: 0,
                id: Some("call_2"),
                name: Some("time"),
                arguments: Some("{"),
            }],
            ToolCallConflictPolicy::EvictDistinctIdAndName,
        );

        assert!(
            matches!(events.first(), Some(RawStreamingChoice::ToolCall(tool_call)) if tool_call.id == "call_1")
        );
        assert_eq!(
            tool_calls.get(&0).map(|tool_call| tool_call.id.as_str()),
            Some("call_2")
        );
        assert_eq!(
            tool_calls.get(&0).map(|tool_call| tool_call.name.as_str()),
            Some("time")
        );
    }

    #[test]
    fn finalizes_null_arguments_into_empty_objects() {
        let finalized = finalize_completed_streaming_tool_call(RawStreamingToolCall::empty());
        assert_eq!(finalized.arguments, serde_json::json!({}));
    }

    #[test]
    fn drains_finalized_tool_calls() {
        let mut tool_calls = HashMap::from([(0, RawStreamingToolCall::empty())]);

        let events = take_finalized_tool_calls::<()>(&mut tool_calls);

        assert!(tool_calls.is_empty());
        assert!(
            matches!(events.as_slice(), [RawStreamingChoice::ToolCall(tool_call)] if tool_call.arguments == serde_json::json!({}))
        );
    }
}
