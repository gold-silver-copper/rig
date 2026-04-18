use crate::completion::{CompletionError, CompletionRequest as CoreCompletionRequest};
use crate::message;

/// Shared unsupported-feature policy for request builders.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CompatibleFeaturePolicy {
    supports_output_schema: bool,
    supports_tools: bool,
    supports_tool_choice: bool,
    supports_additional_params: bool,
}

impl Default for CompatibleFeaturePolicy {
    fn default() -> Self {
        Self {
            supports_output_schema: false,
            supports_tools: true,
            supports_tool_choice: true,
            supports_additional_params: true,
        }
    }
}

impl CompatibleFeaturePolicy {
    pub(crate) const fn supports_output_schema(mut self) -> Self {
        self.supports_output_schema = true;
        self
    }

    pub(crate) const fn without_tools(mut self) -> Self {
        self.supports_tools = false;
        self
    }

    pub(crate) const fn without_tool_choice(mut self) -> Self {
        self.supports_tool_choice = false;
        self
    }

    pub(crate) const fn without_additional_params(mut self) -> Self {
        self.supports_additional_params = false;
        self
    }

    pub(crate) fn apply(self, provider_name: &'static str, req: &mut CoreCompletionRequest) {
        if req.output_schema.is_some() && !self.supports_output_schema {
            tracing::warn!(
                "Structured outputs currently not supported for {}",
                provider_name
            );
            req.output_schema = None;
        }

        if !req.tools.is_empty() && !self.supports_tools {
            tracing::warn!("WARNING: `tools` not supported on {}", provider_name);
            req.tools.clear();
        }

        if req.tool_choice.is_some() && !self.supports_tool_choice {
            tracing::warn!("WARNING: `tool_choice` not supported on {}", provider_name);
            req.tool_choice = None;
        }

        if req.additional_params.is_some() && !self.supports_additional_params {
            tracing::warn!(
                "WARNING: `additional_params` not supported on {}",
                provider_name
            );
            req.additional_params = None;
        }
    }
}

/// Shared request-shaping profile for OpenAI-compatible chat providers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CompatibleChatProfile {
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
                supports_output_schema: false,
                supports_tools: true,
                supports_tool_choice: true,
                supports_additional_params: true,
            },
        }
    }

    pub(crate) const fn require_messages(mut self) -> Self {
        self.require_messages = true;
        self
    }

    pub(crate) const fn supports_output_schema(mut self) -> Self {
        self.feature_policy = self.feature_policy.supports_output_schema();
        self
    }

    pub(crate) const fn without_tools(mut self) -> Self {
        self.feature_policy = self.feature_policy.without_tools();
        self
    }

    pub(crate) const fn without_tool_choice(mut self) -> Self {
        self.feature_policy = self.feature_policy.without_tool_choice();
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
    pub output_schema: Option<schemars::Schema>,
}

pub(crate) fn build_compatible_request_core<M, F>(
    default_model: &str,
    mut req: CoreCompletionRequest,
    profile: CompatibleChatProfile,
    system_message: impl Fn(&str) -> M,
    mut convert_message: F,
) -> Result<CompatibleRequestCore<M>, CompletionError>
where
    F: FnMut(message::Message) -> Result<Vec<M>, CompletionError>,
{
    profile
        .feature_policy
        .apply(profile.provider_name, &mut req);

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

    Ok(CompatibleRequestCore {
        model: request_model.unwrap_or_else(|| default_model.to_owned()),
        messages,
        temperature,
        max_tokens,
        tools,
        tool_choice,
        additional_params,
        output_schema,
    })
}

#[cfg(test)]
mod tests {
    use super::{CompatibleChatProfile, CompatibleFeaturePolicy, build_compatible_request_core};
    use crate::OneOrMany;
    use crate::completion::{CompletionRequest, ToolDefinition};
    use crate::message::{Message, ToolChoice, UserContent};

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
            |preamble| format!("system:{preamble}"),
            |_message| Ok(vec!["history".to_owned()]),
        )
        .expect("request conversion should succeed");

        assert_eq!(result.model, "override-model");
        assert_eq!(result.temperature, Some(0.5));
        assert_eq!(result.max_tokens, Some(42));
        assert_eq!(
            result.messages,
            vec!["system:system".to_owned(), "history".to_owned()]
        );
        assert_eq!(
            result.additional_params,
            Some(serde_json::json!({"foo":"bar"}))
        );
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
                name: "ping".to_owned(),
                description: "Ping tool".to_owned(),
                parameters: serde_json::json!({"type":"object"}),
            }],
            temperature: None,
            max_tokens: None,
            tool_choice: Some(ToolChoice::Required),
            additional_params: Some(serde_json::json!({"foo":"bar"})),
            output_schema: Some(
                serde_json::from_value(serde_json::json!({
                    "title": "Example",
                    "type": "object"
                }))
                .expect("schema should deserialize"),
            ),
        };

        CompatibleFeaturePolicy::default()
            .without_tools()
            .without_tool_choice()
            .without_additional_params()
            .apply("Example Provider", &mut req);

        assert!(req.tools.is_empty());
        assert!(req.tool_choice.is_none());
        assert!(req.additional_params.is_none());
        assert!(req.output_schema.is_none());
    }
}
