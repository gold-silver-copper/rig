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
        let ndims = embedding_dimensions(model)?;
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
pub struct EmbeddingModel {
    embedder: Arc<TextEmbedding>,
    pub model: FastembedModel,
    ndims: usize,
}

impl EmbeddingModel {
    #[cfg(feature = "hf-hub")]
    pub fn new(model: &fastembed::EmbeddingModel, ndims: usize) -> Result<Self, EmbeddingError> {
        let embedder = Arc::new(
            TextEmbedding::try_new(
                InitOptions::new(model.to_owned()).with_show_download_progress(true),
            )
            .map_err(|error| EmbeddingError::initialization(error.to_string()))?,
        );

        Ok(Self {
            embedder,
            model: model.to_owned(),
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
        .map_err(|error| EmbeddingError::initialization(error.to_string()))?;

        let embedder = Arc::new(fastembed_embedding_model);

        Ok(Self {
            embedder,
            model: model_info.model.to_owned(),
            ndims,
        })
    }
}

fn embedding_dimensions(model: &FastembedModel) -> Result<usize, EmbeddingError> {
    TextEmbedding::get_model_info(model)
        .map(|info| info.dim)
        .map_err(|error| EmbeddingError::initialization(error.to_string()))
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client;

    fn make(
        _client: &Self::Client,
        model: impl Into<String>,
        dims: Option<usize>,
    ) -> Result<Self, EmbeddingError> {
        let requested_model = model.into();
        let model = FastembedModel::try_from(requested_model.clone()).map_err(|error| {
            EmbeddingError::initialization(format!(
                "FastEmbed model `{requested_model}` is unavailable: {error}"
            ))
        })?;
        let ndims = match dims {
            Some(ndims) => ndims,
            None => embedding_dimensions(&model)?,
        };

        #[cfg(feature = "hf-hub")]
        {
            Self::new(&model, ndims)
        }

        #[cfg(not(feature = "hf-hub"))]
        {
            let _ = ndims;
            Err(EmbeddingError::initialization(
                "FastEmbed support requires the `hf-hub` feature",
            ))
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

        let documents_as_vec = self
            .embedder
            .embed(documents_as_strings.clone(), None)
            .map_err(|err| EmbeddingError::provider(err.to_string()))?;

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
    async fn invalid_model_make_fails_during_construction() {
        let error = <EmbeddingModel as embeddings::EmbeddingModel>::make(
            &Client::new(),
            "not-a-fastembed-model",
            None,
        )
        .err()
        .expect("invalid model should fail during construction");

        assert!(matches!(error, EmbeddingError::InitializationError(_)));
    }

    #[test]
    fn invalid_model_with_explicit_dims_still_fails_construction() {
        let error = <EmbeddingModel as embeddings::EmbeddingModel>::make(
            &Client::new(),
            "not-a-fastembed-model",
            Some(384),
        )
        .err()
        .expect("explicit dims must not bypass model validation");

        assert!(matches!(error, EmbeddingError::InitializationError(_)));
    }
}
