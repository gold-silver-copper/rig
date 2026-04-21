use crate::{http_client, wasm_compat::WasmCompatSend};
use http::StatusCode;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VerifyProviderError {
    #[error("provider internal error: {message}")]
    InternalServerError { message: String },

    #[error("provider overloaded: {message}")]
    Overloaded { message: String },

    #[error("provider returned unexpected status {status}: {message}")]
    UnexpectedStatus { status: StatusCode, message: String },
}

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("invalid authentication")]
    InvalidAuthentication,
    #[error(transparent)]
    ProviderError(VerifyProviderError),
    #[error("http error: {0}")]
    HttpError(
        #[from]
        #[source]
        http_client::Error,
    ),
}

impl VerifyError {
    pub fn internal_server(message: impl Into<String>) -> Self {
        Self::ProviderError(VerifyProviderError::InternalServerError {
            message: message.into(),
        })
    }

    pub fn overloaded(message: impl Into<String>) -> Self {
        Self::ProviderError(VerifyProviderError::Overloaded {
            message: message.into(),
        })
    }

    pub fn unexpected_status(status: StatusCode, message: impl Into<String>) -> Self {
        Self::ProviderError(VerifyProviderError::UnexpectedStatus {
            status,
            message: message.into(),
        })
    }
}

/// A provider client that can verify the configuration.
/// Clone is required for conversions between client types.
pub trait VerifyClient {
    /// Verify the configuration.
    fn verify(&self) -> impl Future<Output = Result<(), VerifyError>> + WasmCompatSend;
}

#[cfg(test)]
mod tests {
    use super::{VerifyError, VerifyProviderError};
    use http::StatusCode;

    #[test]
    fn verify_error_constructors_capture_provider_context() {
        assert!(matches!(
            VerifyError::internal_server("server exploded"),
            VerifyError::ProviderError(VerifyProviderError::InternalServerError { message })
                if message == "server exploded"
        ));
        assert!(matches!(
            VerifyError::overloaded("too many requests"),
            VerifyError::ProviderError(VerifyProviderError::Overloaded { message })
                if message == "too many requests"
        ));
        assert!(matches!(
            VerifyError::unexpected_status(StatusCode::BAD_GATEWAY, "proxy"),
            VerifyError::ProviderError(VerifyProviderError::UnexpectedStatus {
                status,
                message,
            }) if status == StatusCode::BAD_GATEWAY && message == "proxy"
        ));
    }
}
