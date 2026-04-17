use super::{
    client::{ApiErrorResponse, ApiResponse},
    completion::Usage,
};
use crate::embeddings::EmbeddingError;
use crate::http_client::HttpClientExt;
use crate::{embeddings, http_client};
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// OpenAI Embedding API
// ================================================================
#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<serde_json::Number>,
    pub index: usize,
}

#[doc(hidden)]
#[derive(Clone)]
pub struct GenericEmbeddingModel<Ext = super::OpenAIResponsesExt, H = reqwest::Client> {
    client: crate::client::Client<Ext, H>,
    pub model: String,
    pub encoding_format: Option<EncodingFormat>,
    pub user: Option<String>,
    ndims: usize,
}

/// The embedding model struct for OpenAI's Embeddings API.
///
/// This preserves the historical public generic shape where the first generic
/// parameter is the HTTP client type.
pub type EmbeddingModel<H = reqwest::Client> = GenericEmbeddingModel<super::OpenAIResponsesExt, H>;

fn embedding_metadata(identifier: &str) -> Option<crate::models::EmbeddingMetadata> {
    crate::models::openai::lookup(identifier).and_then(|model| model.embedding)
}

impl<Ext, H> embeddings::EmbeddingModel for GenericEmbeddingModel<Ext, H>
where
    crate::client::Client<Ext, H>: HttpClientExt + Clone + std::fmt::Debug + Send + 'static,
    Ext: crate::client::Provider + Clone + 'static,
    H: Clone + Default + std::fmt::Debug + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;

    type Client = crate::client::Client<Ext, H>;

    fn make(client: &Self::Client, model: impl Into<String>, ndims: Option<usize>) -> Self {
        let model = model.into();
        let dims = ndims
            .or_else(|| embedding_metadata(&model).and_then(|metadata| metadata.default_dimensions))
            .unwrap_or_default();

        Self::new(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let mut body = json!({
            "model": self.model,
            "input": documents,
        });

        let supports_dimensions_override = embedding_metadata(self.model.as_str())
            .and_then(|metadata| metadata.supports_dimensions_override)
            .unwrap_or(true);

        if self.ndims > 0 && supports_dimensions_override {
            body["dimensions"] = json!(self.ndims);
        }

        if let Some(encoding_format) = &self.encoding_format {
            body["encoding_format"] = json!(encoding_format);
        }

        if let Some(user) = &self.user {
            body["user"] = json!(user);
        }

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/embeddings")?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        if response.status().is_success() {
            let body: Vec<u8> = response.into_body().await?;
            let body: ApiResponse<EmbeddingResponse> = serde_json::from_slice(&body)?;

            match body {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "OpenAI embedding token usage: {:?}",
                        response.usage
                    );

                    if response.data.len() != documents.len() {
                        return Err(EmbeddingError::ResponseError(
                            "Response data length does not match input length".into(),
                        ));
                    }

                    Ok(response
                        .data
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding
                                .embedding
                                .into_iter()
                                .filter_map(|n| n.as_f64())
                                .collect(),
                        })
                        .collect())
                }
                ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
            }
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::ProviderError(text))
        }
    }
}

impl<Ext, H> GenericEmbeddingModel<Ext, H>
where
    Ext: crate::client::Provider,
{
    pub fn new(
        client: crate::client::Client<Ext, H>,
        model: impl Into<String>,
        ndims: usize,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: None,
            ndims,
            user: None,
        }
    }

    pub fn with_model(client: crate::client::Client<Ext, H>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: None,
            ndims,
            user: None,
        }
    }

    pub fn with_encoding_format(
        client: crate::client::Client<Ext, H>,
        model: &str,
        ndims: usize,
        encoding_format: EncodingFormat,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: Some(encoding_format),
            ndims,
            user: None,
        }
    }

    pub fn encoding_format(mut self, encoding_format: EncodingFormat) -> Self {
        self.encoding_format = Some(encoding_format);
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_uses_catalog_default_dimensions_for_known_models() {
        let client = crate::providers::openai::Client::new("test-key").expect("build client");
        let model = <EmbeddingModel as embeddings::EmbeddingModel>::make(
            &client,
            crate::models::openai::TEXT_EMBEDDING_3_LARGE,
            None,
        );

        assert_eq!(model.ndims, 3_072);
    }

    #[test]
    fn known_ada_models_disable_dimension_overrides() {
        let metadata = embedding_metadata(crate::models::openai::TEXT_EMBEDDING_ADA_002)
            .expect("ada-002 metadata should exist");

        assert_eq!(metadata.supports_dimensions_override, Some(false));
    }

    #[test]
    fn unknown_models_keep_zero_default_dimensions() {
        let client = crate::providers::openai::Client::new("test-key").expect("build client");
        let model = <EmbeddingModel as embeddings::EmbeddingModel>::make(
            &client,
            "future-embed-model",
            None,
        );

        assert_eq!(model.ndims, 0);
    }
}
