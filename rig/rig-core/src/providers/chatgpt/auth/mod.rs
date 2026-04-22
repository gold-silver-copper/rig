//! Shared ChatGPT authentication types and target-specific dispatch.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(target_family = "wasm")]
mod wasm;

#[cfg(not(target_family = "wasm"))]
use native as platform;
#[cfg(target_family = "wasm")]
use wasm as platform;

#[derive(Debug, Clone)]
pub struct DeviceCodePrompt {
    pub verification_uri: String,
    pub user_code: String,
}

#[derive(Clone, Default)]
pub struct DeviceCodeHandler(Option<Arc<dyn Fn(DeviceCodePrompt) + Send + Sync>>);

impl DeviceCodeHandler {
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(DeviceCodePrompt) + Send + Sync + 'static,
    {
        Self(Some(Arc::new(handler)))
    }
}

impl fmt::Debug for DeviceCodeHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_some() {
            f.write_str("DeviceCodeHandler(<callback>)")
        } else {
            f.write_str("DeviceCodeHandler(None)")
        }
    }
}

#[derive(Clone)]
pub enum AuthSource {
    AccessToken {
        access_token: String,
        account_id: Option<String>,
    },
    OAuth,
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AccessToken { .. } => f.write_str("AccessToken(<redacted>)"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

#[derive(Clone)]
pub struct Authenticator {
    source: AuthSource,
    platform: platform::PlatformAuthenticator,
    state_lock: Arc<Mutex<()>>,
}

impl fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Authenticator")
            .field("source", &self.source)
            .field("platform", &self.platform)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error(transparent)]
    Flow(AuthFlowError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
}

/// Structured ChatGPT authentication flow failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthFlowError {
    /// ChatGPT OAuth is unavailable on the current target.
    UnsupportedOnWasm,
    /// The device-code authorization window expired before the user finished it.
    DeviceAuthorizationTimedOut,
    /// The device-code endpoint returned an unexpected non-polling failure.
    DeviceAuthorizationFailed {
        status: reqwest::StatusCode,
        body: String,
    },
    /// Refresh-token exchange failed and did not qualify for reauthentication.
    TokenRefreshFailed {
        status: reqwest::StatusCode,
        error_code: Option<String>,
        error_description: Option<String>,
        response_body: Option<String>,
    },
}

impl std::fmt::Display for AuthFlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOnWasm => {
                f.write_str("ChatGPT OAuth is not supported on wasm targets")
            }
            Self::DeviceAuthorizationTimedOut => {
                f.write_str("Timed out waiting for ChatGPT device authorization")
            }
            Self::DeviceAuthorizationFailed { status, body } => {
                if body.trim().is_empty() {
                    write!(f, "ChatGPT device authorization failed: {status}")
                } else {
                    write!(f, "ChatGPT device authorization failed: {status} {body}")
                }
            }
            Self::TokenRefreshFailed {
                status,
                error_code,
                error_description,
                response_body,
            } => {
                write!(f, "ChatGPT token refresh failed: {status}")?;

                if let Some(error_code) = error_code.as_deref() {
                    write!(f, " {error_code}")?;
                }

                if let Some(description) = error_description
                    .as_deref()
                    .map(str::trim)
                    .filter(|description| !description.is_empty())
                {
                    write!(f, " ({description})")?;
                } else if error_code.is_none()
                    && let Some(body) = response_body
                        .as_deref()
                        .map(str::trim)
                        .filter(|body| !body.is_empty())
                {
                    write!(f, " {body}")?;
                }

                Ok(())
            }
        }
    }
}

impl std::error::Error for AuthFlowError {}

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub access_token: String,
    pub account_id: Option<String>,
}

impl Authenticator {
    pub fn new(
        source: AuthSource,
        auth_file: Option<PathBuf>,
        device_code_handler: DeviceCodeHandler,
    ) -> Self {
        Self {
            source,
            platform: platform::PlatformAuthenticator::new(auth_file, device_code_handler),
            state_lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn auth_context(&self) -> Result<AuthContext, AuthError> {
        match &self.source {
            AuthSource::AccessToken {
                access_token,
                account_id,
            } => Ok(AuthContext {
                access_token: access_token.clone(),
                account_id: account_id.clone(),
            }),
            AuthSource::OAuth => {
                let _guard = self.state_lock.lock().await;
                self.platform.auth_context_oauth().await
            }
        }
    }
}
