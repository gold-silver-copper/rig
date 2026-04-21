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
    ApiKey(String),
    GitHubAccessToken(String),
    OAuth,
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("ApiKey(<redacted>)"),
            Self::GitHubAccessToken(_) => f.write_str("GitHubAccessToken(<redacted>)"),
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

/// Structured GitHub Copilot authentication flow failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthFlowError {
    /// Copilot OAuth is unavailable on the current target.
    UnsupportedOnWasm,
    /// The device-code authorization window expired before the user finished it.
    DeviceAuthorizationTimedOut,
    /// GitHub reported that the device code expired before completion.
    DeviceAuthorizationExpired,
    /// GitHub reported that the user denied the device authorization request.
    DeviceAuthorizationDenied,
    /// GitHub returned a terminal OAuth device-code error.
    DeviceAuthorizationFailed {
        error_code: Option<String>,
        error_description: Option<String>,
    },
    /// The Copilot API key response was missing the issued token.
    MissingApiKeyToken,
}

impl std::fmt::Display for AuthFlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOnWasm => {
                f.write_str("GitHub Copilot OAuth is not supported on wasm targets")
            }
            Self::DeviceAuthorizationTimedOut => {
                f.write_str("Timed out waiting for GitHub Copilot device authorization")
            }
            Self::DeviceAuthorizationExpired => {
                f.write_str("GitHub device authorization expired before it completed")
            }
            Self::DeviceAuthorizationDenied => {
                f.write_str("GitHub device authorization was denied")
            }
            Self::DeviceAuthorizationFailed {
                error_code,
                error_description,
            } => match (
                error_code.as_deref(),
                error_description
                    .as_deref()
                    .map(str::trim)
                    .filter(|description| !description.is_empty()),
            ) {
                (Some(error_code), Some(description)) => write!(
                    f,
                    "GitHub device authorization failed: {error_code} ({description})"
                ),
                (Some(error_code), None) => {
                    write!(f, "GitHub device authorization failed: {error_code}")
                }
                (None, _) => f.write_str("GitHub device authorization failed: unknown error"),
            },
            Self::MissingApiKeyToken => {
                f.write_str("GitHub Copilot API key response did not include a token")
            }
        }
    }
}

impl std::error::Error for AuthFlowError {}

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub api_key: String,
    pub api_base: Option<String>,
}

impl Authenticator {
    pub fn new(
        source: AuthSource,
        access_token_file: Option<PathBuf>,
        api_key_file: Option<PathBuf>,
        device_code_handler: DeviceCodeHandler,
    ) -> Self {
        Self {
            source,
            platform: platform::PlatformAuthenticator::new(
                access_token_file,
                api_key_file,
                device_code_handler,
            ),
            state_lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn auth_context(&self) -> Result<AuthContext, AuthError> {
        match &self.source {
            AuthSource::ApiKey(api_key) => Ok(AuthContext {
                api_key: api_key.clone(),
                api_base: None,
            }),
            AuthSource::GitHubAccessToken(access_token) => {
                let _guard = self.state_lock.lock().await;
                self.platform
                    .auth_context_with_github_access_token(access_token)
                    .await
            }
            AuthSource::OAuth => {
                let _guard = self.state_lock.lock().await;
                self.platform.auth_context_oauth().await
            }
        }
    }
}
