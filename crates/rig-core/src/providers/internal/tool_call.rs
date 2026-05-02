use crate::{
    message::{ToolCall, ToolFunction},
    model_event::ModelEvent,
};

#[derive(Debug, Clone)]
pub(crate) struct ProviderToolCall {
    pub(crate) id: String,
    pub(crate) internal_call_id: String,
    pub(crate) call_id: Option<String>,
    pub(crate) name: String,
    pub(crate) arguments: serde_json::Value,
    pub(crate) signature: Option<String>,
    pub(crate) additional_params: Option<serde_json::Value>,
}

impl ProviderToolCall {
    pub(crate) fn empty() -> Self {
        Self {
            id: String::new(),
            internal_call_id: nanoid::nanoid!(),
            call_id: None,
            name: String::new(),
            arguments: serde_json::Value::Null,
            signature: None,
            additional_params: None,
        }
    }

    pub(crate) fn new(id: String, name: String, arguments: serde_json::Value) -> Self {
        Self {
            id,
            internal_call_id: nanoid::nanoid!(),
            call_id: None,
            name,
            arguments,
            signature: None,
            additional_params: None,
        }
    }

    pub(crate) fn with_internal_call_id(mut self, internal_call_id: String) -> Self {
        self.internal_call_id = internal_call_id;
        self
    }

    pub(crate) fn with_call_id(mut self, call_id: String) -> Self {
        self.call_id = Some(call_id);
        self
    }

    pub(crate) fn with_signature(mut self, signature: Option<String>) -> Self {
        self.signature = signature;
        self
    }
}

impl From<ProviderToolCall> for ToolCall {
    fn from(tool_call: ProviderToolCall) -> Self {
        ToolCall {
            id: tool_call.id,
            call_id: tool_call.call_id,
            function: ToolFunction {
                name: tool_call.name,
                arguments: tool_call.arguments,
            },
            signature: tool_call.signature,
            additional_params: tool_call.additional_params,
        }
    }
}

impl<R> From<ProviderToolCall> for ModelEvent<R> {
    fn from(tool_call: ProviderToolCall) -> Self {
        let internal_call_id = tool_call.internal_call_id.clone();
        Self::ToolCallDone {
            tool_call: tool_call.into(),
            internal_call_id: Some(internal_call_id),
        }
    }
}
