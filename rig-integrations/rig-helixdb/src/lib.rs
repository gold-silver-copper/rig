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

use helix_rs::HelixDBClient;
use rig::{
    embeddings::EmbeddingModel,
    vector_store::{InsertDocuments, VectorStoreError, VectorStoreIndex, request::Filter},
};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug)]
enum HelixDbVectorStoreError {
    Query(String),
}

impl fmt::Display for HelixDbVectorStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HelixDbVectorStoreError::Query(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for HelixDbVectorStoreError {}

/// A client for easily carrying out Rig-related vector store operations.
///
/// If you are unsure what type to use for the client, `helix_rs::HelixDB` is the typical default.
///
/// Usage:
/// ```rust
/// let openai_model =
///     rig::providers::openai::Client::from_env().embedding_model("text-embedding-ada-002");
///
/// let helixdb_client = HelixDB::new(None, Some(6969), None);
/// let vector_store = HelixDBVectorStore::new(helixdb_client, openai_model.clone());
/// ```
pub struct HelixDBVectorStore<C, E> {
    client: C,
    model: E,
}

pub type HelixDBFilter = Filter<serde_json::Value>;

/// The result of a query. Only used internally as this is a representative type required for the relevant HelixDB query (`VectorSearch`).
#[derive(Deserialize, Serialize, Clone, Debug)]
struct QueryResult {
    id: String,
    score: f64,
    doc: String,
    json_payload: String,
}

/// An input query. Only used internally as this is a representative type required for the relevant HelixDB query (`VectorSearch`).
#[derive(Deserialize, Serialize, Clone, Debug)]
struct QueryInput {
    vector: Vec<f64>,
    limit: u64,
    threshold: f64,
}

impl QueryInput {
    /// Makes a new instance of `QueryInput`.
    pub(crate) fn new(vector: Vec<f64>, limit: u64, threshold: f64) -> Self {
        Self {
            vector,
            limit,
            threshold,
        }
    }
}

impl<C, E> HelixDBVectorStore<C, E>
where
    C: HelixDBClient + Send,
    E: EmbeddingModel,
{
    pub fn new(client: C, model: E) -> Self {
        Self { client, model }
    }

    pub fn client(&self) -> &C {
        &self.client
    }
}

impl<C, E> InsertDocuments for HelixDBVectorStore<C, E>
where
    C: HelixDBClient + Send + Sync,
    E: EmbeddingModel + Send + Sync,
{
    async fn insert_documents<Doc: Serialize + rig::Embed + Send>(
        &self,
        documents: Vec<(Doc, rig::OneOrMany<rig::embeddings::Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        #[derive(Serialize, Deserialize, Clone, Debug, Default)]
        struct QueryInput {
            vector: Vec<f64>,
            doc: String,
            json_payload: String,
        }

        #[derive(Serialize, Deserialize, Clone, Debug, Default)]
        struct QueryOutput {
            doc: String,
        }

        for (document, embeddings) in documents {
            let json_document = serde_json::to_value(&document)?;
            let json_document_as_string = serde_json::to_string(&json_document)?;

            for embedding in embeddings {
                let embedded_text = embedding.document;
                let vector: Vec<f64> = embedding.vec;

                let query = QueryInput {
                    vector,
                    doc: embedded_text,
                    json_payload: json_document_as_string.clone(),
                };

                self.client
                    .query::<QueryInput, QueryOutput>("InsertVector", &query)
                    .await
                    .map_err(|error| {
                        VectorStoreError::DatastoreError(Box::new(HelixDbVectorStoreError::Query(
                            error.to_string(),
                        )))
                    })?;
            }
        }
        Ok(())
    }
}

impl<C, E> VectorStoreIndex for HelixDBVectorStore<C, E>
where
    C: HelixDBClient + Send + Sync,
    E: EmbeddingModel + Send + Sync,
{
    type Filter = HelixDBFilter;

    async fn top_n<T: for<'a> serde::Deserialize<'a> + Send>(
        &self,
        req: rig::vector_store::VectorSearchRequest<HelixDBFilter>,
    ) -> Result<Vec<(f64, String, T)>, rig::vector_store::VectorStoreError> {
        let vector = self.model.embed_text(req.query()).await?.vec;

        let query_input =
            QueryInput::new(vector, req.samples(), req.threshold().unwrap_or_default());

        #[derive(Serialize, Deserialize, Debug)]
        struct VecResult {
            vec_docs: Vec<QueryResult>,
        }

        let result: VecResult = self
            .client
            .query::<QueryInput, VecResult>("VectorSearch", &query_input)
            .await
            .map_err(|error| {
                VectorStoreError::DatastoreError(Box::new(HelixDbVectorStoreError::Query(
                    error.to_string(),
                )))
            })?;

        let mut docs = Vec::new();
        for item in result.vec_docs {
            let score = -(item.score - 1.);
            if req.threshold().is_some_and(|threshold| score < threshold) {
                continue;
            }

            let payload: serde_json::Value = serde_json::from_str(&item.json_payload)?;
            if req
                .filter()
                .as_ref()
                .is_some_and(|filter| !filter.satisfies(&payload))
            {
                continue;
            }

            let doc: T = serde_json::from_value(payload)?;

            // HelixDB gives us the cosine distance, so we need to use `-(cosine_dist - 1)` to get the cosine similarity score.
            docs.push((score, item.id, doc));
        }

        Ok(docs)
    }

    async fn top_n_ids(
        &self,
        req: rig::vector_store::VectorSearchRequest<HelixDBFilter>,
    ) -> Result<Vec<(f64, String)>, rig::vector_store::VectorStoreError> {
        let vector = self.model.embed_text(req.query()).await?.vec;

        let query_input =
            QueryInput::new(vector, req.samples(), req.threshold().unwrap_or_default());

        #[derive(Serialize, Deserialize, Debug)]
        struct VecResult {
            vec_docs: Vec<QueryResult>,
        }

        let result: VecResult = self
            .client
            .query::<QueryInput, VecResult>("VectorSearch", &query_input)
            .await
            .map_err(|error| {
                VectorStoreError::DatastoreError(Box::new(HelixDbVectorStoreError::Query(
                    error.to_string(),
                )))
            })?;

        // HelixDB gives us the cosine distance, so we need to use `-(cosine_dist - 1)` to get the cosine similarity score.
        let mut docs = Vec::new();
        for item in result.vec_docs {
            let score = -(item.score - 1.);
            if req.threshold().is_some_and(|threshold| score < threshold) {
                continue;
            }

            if let Some(filter) = req.filter() {
                let payload: serde_json::Value = serde_json::from_str(&item.json_payload)?;
                if !filter.satisfies(&payload) {
                    continue;
                }
            }

            docs.push((score, item.id));
        }

        Ok(docs)
    }
}
