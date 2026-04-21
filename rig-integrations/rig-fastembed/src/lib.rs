#![cfg_attr(
    not(test),
    deny(
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unreachable,
        clippy::unwrap_used,
        clippy::indexing_slicing
    )
)]

use std::sync::Arc;

pub use fastembed::EmbeddingModel as FastembedModel;
use fastembed::{InitOptionsUserDefined, ModelInfo, TextEmbedding, UserDefinedEmbeddingModel};
use rig::embeddings::{self, EmbeddingError};

#[cfg(feature = "hf-hub")]
use fastembed::InitOptions;
#[cfg(feature = "hf-hub")]
use rig::{Embed, embeddings::EmbeddingsBuilder};

/// The `rig-fastembed` client.
///
/// Use this as your main entrypoint for any `rig-fastembed` functionality.
#[derive(Clone)]
pub struct Client;

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    /// Create a new `rig-fastembed` client.
    pub fn new() -> Self {
        Self
    }

    /// Create an embedding model from a known FastEmbed model identifier.
    ///
    /// # Example
    /// ```
    /// use rig_fastembed::{Client, FastembedModel};
    ///
    /// // Initialize the `rig-fastembed` client
    /// let fastembed_client = rig_fastembed::Client::new();
    ///
    /// let embedding_model = fastembed_client.embedding_model(&FastembedModel::AllMiniLML6V2Q)?;
    /// # Ok::<_, rig::embeddings::EmbeddingError>(())
    /// ```
    #[cfg(feature = "hf-hub")]
    pub fn embedding_model(
        &self,
        model: &FastembedModel,
    ) -> Result<EmbeddingModel, EmbeddingError> {
        let ndims = TextEmbedding::get_model_info(model)
            .map(|info| info.dim)
            .map_err(|error| EmbeddingError::InitializationError(error.to_string()))?;

        EmbeddingModel::new(model, ndims)
    }

    /// Create an embedding builder with the given embedding model.
    ///
    /// # Example
    /// ```no_run
    /// use rig_fastembed::{Client, FastembedModel};
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// // Initialize the Fastembed client
    /// let fastembed_client = Client::new();
    ///
    /// let embeddings = fastembed_client
    ///     .embeddings::<String>(&FastembedModel::AllMiniLML6V2Q)?
    ///     .documents(vec!["Hello, world!".to_string(), "Goodbye, world!".to_string()])?
    ///     .build()
    ///     .await?;
    /// # let _ = embeddings;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "hf-hub")]
    pub fn embeddings<D: Embed>(
        &self,
        model: &fastembed::EmbeddingModel,
    ) -> Result<EmbeddingsBuilder<EmbeddingModel, D>, EmbeddingError> {
        Ok(EmbeddingsBuilder::new(self.embedding_model(model)?))
    }
}

#[derive(Clone)]
enum EmbedderState {
    Ready(Arc<TextEmbedding>),
    InitializationError(String),
}

#[derive(Clone)]
pub struct EmbeddingModel {
    embedder: EmbedderState,
    pub model: Option<FastembedModel>,
    requested_model: String,
    ndims: usize,
}

impl EmbeddingModel {
    fn model_name(model: &FastembedModel) -> String {
        TextEmbedding::get_model_info(model)
            .map(|info| info.model_code.clone())
            .unwrap_or_else(|_| format!("{model:?}"))
    }

    fn initialization_failed(
        model: Option<FastembedModel>,
        requested_model: String,
        ndims: usize,
        error: impl Into<String>,
    ) -> Self {
        Self {
            embedder: EmbedderState::InitializationError(error.into()),
            model,
            requested_model,
            ndims,
        }
    }

    /// Return the model identifier originally requested by the caller.
    pub fn requested_model(&self) -> &str {
        &self.requested_model
    }

    /// Return whether this model is ready to serve embeddings.
    pub fn is_available(&self) -> bool {
        matches!(self.embedder, EmbedderState::Ready(_))
    }

    fn embedder(&self) -> Result<&Arc<TextEmbedding>, EmbeddingError> {
        match &self.embedder {
            EmbedderState::Ready(embedder) => Ok(embedder),
            EmbedderState::InitializationError(error) => {
                Err(EmbeddingError::InitializationError(format!(
                    "FastEmbed model `{}` is unavailable: {error}",
                    self.requested_model
                )))
            }
        }
    }

    #[cfg(feature = "hf-hub")]
    pub fn new(model: &fastembed::EmbeddingModel, ndims: usize) -> Result<Self, EmbeddingError> {
        let embedder = Arc::new(
            TextEmbedding::try_new(
                InitOptions::new(model.to_owned()).with_show_download_progress(true),
            )
            .map_err(|error| EmbeddingError::InitializationError(error.to_string()))?,
        );

        Ok(Self {
            embedder: EmbedderState::Ready(embedder),
            model: Some(model.to_owned()),
            requested_model: Self::model_name(model),
            ndims,
        })
    }

    pub fn new_from_user_defined(
        user_defined_model: UserDefinedEmbeddingModel,
        ndims: usize,
        model_info: &ModelInfo<FastembedModel>,
    ) -> Result<Self, EmbeddingError> {
        let fastembed_embedding_model = TextEmbedding::try_new_from_user_defined(
            user_defined_model,
            InitOptionsUserDefined::default(),
        )
        .map_err(|error| EmbeddingError::InitializationError(error.to_string()))?;

        let embedder = Arc::new(fastembed_embedding_model);

        Ok(Self {
            embedder: EmbedderState::Ready(embedder),
            model: Some(model_info.model.to_owned()),
            requested_model: model_info.model_code.clone(),
            ndims,
        })
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client;

    fn make(_client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        let requested_model = model.into();
        let parsed_model = FastembedModel::try_from(requested_model.clone());
        match parsed_model {
            Ok(model) => {
                let ndims = dims
                    .or_else(|| {
                        TextEmbedding::get_model_info(&model)
                            .ok()
                            .map(|info| info.dim)
                    })
                    .unwrap_or_default();

                #[cfg(feature = "hf-hub")]
                {
                    Self::new(&model, ndims).unwrap_or_else(|error| {
                        Self::initialization_failed(
                            Some(model),
                            requested_model,
                            ndims,
                            error.to_string(),
                        )
                    })
                }

                #[cfg(not(feature = "hf-hub"))]
                {
                    Self::initialization_failed(
                        Some(model),
                        requested_model,
                        ndims,
                        "FastEmbed support requires the `hf-hub` feature".to_string(),
                    )
                }
            }
            Err(error) => {
                Self::initialization_failed(None, requested_model, dims.unwrap_or_default(), error)
            }
        }
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents_as_strings: Vec<String> = documents.into_iter().collect();
        let embedder = self.embedder()?;

        let documents_as_vec = embedder
            .embed(documents_as_strings.clone(), None)
            .map_err(|err| EmbeddingError::ProviderError(err.to_string()))?;

        let docs = documents_as_strings
            .into_iter()
            .zip(documents_as_vec)
            .map(|(document, embedding)| embeddings::Embedding {
                document,
                vec: embedding.into_iter().map(|f| f as f64).collect(),
            })
            .collect::<Vec<embeddings::Embedding>>();

        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::{Client, EmbeddingModel};
    use rig::embeddings::{self, EmbeddingError};

    #[tokio::test]
    async fn invalid_model_make_returns_error_instead_of_panicking() {
        let model = <EmbeddingModel as embeddings::EmbeddingModel>::make(
            &Client::new(),
            "not-a-fastembed-model",
            None,
        );

        assert!(model.model.is_none());
        assert_eq!(model.requested_model(), "not-a-fastembed-model");
        assert!(!model.is_available());

        let error = embeddings::EmbeddingModel::embed_text(&model, "hello")
            .await
            .expect_err("invalid model should return an initialization error");

        assert!(matches!(error, EmbeddingError::InitializationError(_)));
    }

    #[test]
    fn invalid_model_with_explicit_dims_keeps_dims_without_fabricating_a_model() {
        let model = <EmbeddingModel as embeddings::EmbeddingModel>::make(
            &Client::new(),
            "not-a-fastembed-model",
            Some(384),
        );

        assert!(model.model.is_none());
        assert_eq!(embeddings::EmbeddingModel::ndims(&model), 384);
        assert_eq!(model.requested_model(), "not-a-fastembed-model");
    }
}
