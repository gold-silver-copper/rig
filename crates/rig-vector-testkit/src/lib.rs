//! Shared vector-store conformance fixtures and assertions.
//!
//! This crate intentionally stays independent of any concrete vector-store
//! implementation. Backends provide their own storage setup and then run these
//! fixtures through the common [`VectorStoreIndex`](rig_core::vector_store::VectorStoreIndex)
//! API.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use anyhow::{Context, Result, bail, ensure};
use rig_core::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use rig_core::vector_store::VectorStoreIndex;
use rig_core::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

const FIXTURE_EXPECTED_SCORE_EPSILON: f64 = 1e-9;

/// Distance metric used by an ANN conformance fixture.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AnnMetric {
    /// Cosine similarity where larger scores are better.
    Cosine,
    /// L1 distance represented as a score where larger scores are better.
    L1,
    /// L2 distance represented as a score where larger scores are better.
    L2,
}

/// A small vector-store conformance fixture.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AnnFixture {
    name: String,
    metric: AnnMetric,
    dimensions: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<AnnFixtureSource>,
    documents: Vec<FixtureDocument>,
    queries: Vec<FixtureQuery>,
}

impl AnnFixture {
    /// Creates and validates a fixture.
    pub fn new(
        name: String,
        metric: AnnMetric,
        dimensions: usize,
        source: Option<AnnFixtureSource>,
        documents: Vec<FixtureDocument>,
        queries: Vec<FixtureQuery>,
    ) -> Result<Self> {
        let fixture = Self {
            name,
            metric,
            dimensions,
            source,
            documents,
            queries,
        };
        fixture.validate()?;
        Ok(fixture)
    }

    /// Parses and validates a fixture from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        let fixture = serde_json::from_str::<Self>(json).context("failed to parse ANN fixture")?;
        fixture.validate()?;
        Ok(fixture)
    }

    /// Fixture name used in assertion messages.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Fixture distance metric.
    pub fn metric(&self) -> AnnMetric {
        self.metric
    }

    /// Vector dimensionality for all documents and queries in the fixture.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Source metadata for generated fixtures.
    pub fn source(&self) -> Option<&AnnFixtureSource> {
        self.source.as_ref()
    }

    /// Documents to insert into the vector store under test.
    pub fn documents(&self) -> &[FixtureDocument] {
        &self.documents
    }

    /// Queries to run against the vector store under test.
    pub fn queries(&self) -> &[FixtureQuery] {
        &self.queries
    }

    /// Recomputes the exact expected neighbors for `query` from fixture vectors.
    pub fn expected_neighbors(&self, query: &FixtureQuery) -> Result<Vec<ExpectedNeighbor>> {
        compute_expected_neighbors(
            self.metric,
            self.documents(),
            &query.vector,
            query.top_k,
            query.threshold,
        )
        .with_context(|| {
            format!(
                "failed to compute expected neighbors for fixture '{}' query '{}'",
                self.name, query.id
            )
        })
    }

    /// Validates fixture structure and internal references.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.name.trim().is_empty(),
            "ANN fixture name must not be empty"
        );
        ensure!(
            self.dimensions > 0,
            "ANN fixture '{}' must have at least one dimension",
            self.name
        );
        ensure!(
            !self.documents.is_empty(),
            "ANN fixture '{}' must contain documents",
            self.name
        );
        ensure!(
            !self.queries.is_empty(),
            "ANN fixture '{}' must contain queries",
            self.name
        );

        let mut document_ids = HashSet::new();
        let mut embedding_texts = HashSet::new();
        for document in &self.documents {
            ensure_non_empty(&document.id, "document id", &self.name)?;
            ensure_non_empty(&document.text, "document text", &self.name)?;
            ensure!(
                document_ids.insert(document.id.as_str()),
                "ANN fixture '{}' has duplicate document id '{}'",
                self.name,
                document.id
            );
            ensure!(
                embedding_texts.insert(document.text.as_str()),
                "ANN fixture '{}' has duplicate embedding text '{}'",
                self.name,
                document.text
            );
            ensure_vector_dimensions(&document.vector, self.dimensions, &self.name, &document.id)?;
        }

        for query in &self.queries {
            ensure_non_empty(&query.id, "query id", &self.name)?;
            ensure_non_empty(&query.text, "query text", &self.name)?;
            ensure!(
                embedding_texts.insert(query.text.as_str()),
                "ANN fixture '{}' has duplicate embedding text '{}'",
                self.name,
                query.text
            );
            ensure!(
                query.top_k > 0,
                "ANN fixture '{}' query '{}' must request at least one sample",
                self.name,
                query.id
            );
            ensure!(
                !query.expected.is_empty(),
                "ANN fixture '{}' query '{}' must have expected neighbors",
                self.name,
                query.id
            );
            let expected_len = u64::try_from(query.expected.len())
                .context("expected neighbor count did not fit in u64")?;
            ensure!(
                expected_len <= query.top_k,
                "ANN fixture '{}' query '{}' has more expected neighbors than top_k",
                self.name,
                query.id
            );
            ensure_vector_dimensions(&query.vector, self.dimensions, &self.name, &query.id)?;

            let mut expected_ids = HashSet::new();
            for expected in &query.expected {
                ensure!(
                    document_ids.contains(expected.id.as_str()),
                    "ANN fixture '{}' query '{}' references unknown document id '{}'",
                    self.name,
                    query.id,
                    expected.id
                );
                ensure!(
                    expected_ids.insert(expected.id.as_str()),
                    "ANN fixture '{}' query '{}' has duplicate expected id '{}'",
                    self.name,
                    query.id,
                    expected.id
                );
                ensure!(
                    expected.score.is_finite(),
                    "ANN fixture '{}' query '{}' expected score for '{}' must be finite",
                    self.name,
                    query.id,
                    expected.id
                );
            }

            let oracle = self.expected_neighbors(query)?;
            ensure!(
                oracle.len() == query.expected.len(),
                "ANN fixture '{}' query '{}' has {} embedded expected neighbors but vector oracle produced {}",
                self.name,
                query.id,
                query.expected.len(),
                oracle.len()
            );
            for (expected, oracle) in query.expected.iter().zip(oracle.iter()) {
                ensure!(
                    expected.id == oracle.id,
                    "ANN fixture '{}' query '{}' expected id '{}' did not match vector oracle id '{}'",
                    self.name,
                    query.id,
                    expected.id,
                    oracle.id
                );
                ensure_score_close(
                    expected.score,
                    oracle.score,
                    FIXTURE_EXPECTED_SCORE_EPSILON,
                    &self.name,
                    &query.id,
                    &expected.id,
                    "fixture/vector oracle",
                )?;
            }
        }

        Ok(())
    }
}

/// Provenance metadata for a generated ANN fixture.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AnnFixtureSource {
    /// Source kind, for example `ann-benchmarks-hdf5`.
    pub kind: String,
    /// Source dataset name.
    pub dataset: String,
    /// Source URL, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Source dataset metric, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_metric: Option<String>,
    /// First train row used as a fixture document.
    pub train_start: usize,
    /// Number of train rows used as fixture documents.
    pub train_count: usize,
    /// First test row used as a fixture query.
    pub test_start: usize,
    /// Number of test rows used as fixture queries.
    pub test_count: usize,
    /// Number of expected neighbors generated for non-threshold queries.
    pub top_k: usize,
    /// Tool that generated the fixture.
    pub generated_by: String,
}

/// A document row in an ANN conformance fixture.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FixtureDocument {
    /// Stable document identifier.
    pub id: String,
    /// Text passed to the embedding model when this document is embedded.
    pub text: String,
    /// Precomputed vector for this document.
    pub vector: Vec<f64>,
}

/// A query in an ANN conformance fixture.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FixtureQuery {
    /// Stable query identifier.
    pub id: String,
    /// Text passed to the embedding model when the query is embedded.
    pub text: String,
    /// Precomputed query vector.
    pub vector: Vec<f64>,
    /// Number of nearest neighbors requested from the vector store.
    pub top_k: u64,
    /// Optional minimum score threshold for this query.
    pub threshold: Option<f64>,
    /// Expected neighbors after applying `top_k` and `threshold`.
    pub expected: Vec<ExpectedNeighbor>,
}

/// Expected nearest-neighbor result for a fixture query.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExpectedNeighbor {
    /// Expected document identifier.
    pub id: String,
    /// Expected backend-facing score.
    pub score: f64,
}

/// Computes exact expected neighbors from document vectors and a query vector.
///
/// The returned scores follow Rig's vector-store convention: larger values are
/// better for every metric.
pub fn compute_expected_neighbors(
    metric: AnnMetric,
    documents: &[FixtureDocument],
    query_vector: &[f64],
    top_k: u64,
    threshold: Option<f64>,
) -> Result<Vec<ExpectedNeighbor>> {
    ensure!(top_k > 0, "top_k must be greater than zero");
    ensure!(!documents.is_empty(), "documents must not be empty");
    ensure!(
        query_vector.iter().all(|value| value.is_finite()),
        "query vector contains a non-finite value"
    );

    let dimensions = query_vector.len();
    ensure!(dimensions > 0, "query vector must not be empty");
    let top_k = usize::try_from(top_k).context("top_k did not fit in usize")?;

    let mut scored = documents
        .iter()
        .map(|document| {
            ensure!(
                document.vector.len() == dimensions,
                "document '{}' has {} dimensions, expected {dimensions}",
                document.id,
                document.vector.len()
            );
            ensure!(
                document.vector.iter().all(|value| value.is_finite()),
                "document '{}' vector contains a non-finite value",
                document.id
            );
            Ok(ExpectedNeighbor {
                id: document.id.clone(),
                score: score_vector(metric, query_vector, &document.vector)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    scored.sort_by(compare_neighbor_score_desc);
    let expected = scored
        .into_iter()
        .filter(|neighbor| threshold.is_none_or(|threshold| neighbor.score >= threshold))
        .take(top_k)
        .collect::<Vec<_>>();

    ensure!(
        !expected.is_empty(),
        "vector oracle produced no expected neighbors"
    );

    Ok(expected)
}

/// Deterministic embedding model backed by an [`AnnFixture`].
#[derive(Clone, Debug, Default)]
pub struct FixtureEmbeddingModel {
    dimensions: usize,
    embeddings: HashMap<String, Vec<f64>>,
}

impl FixtureEmbeddingModel {
    /// Builds a deterministic embedding model from all fixture documents and queries.
    pub fn from_fixture(fixture: &AnnFixture) -> Result<Self> {
        fixture.validate()?;

        let mut embeddings = HashMap::new();
        for document in fixture.documents() {
            insert_embedding(
                &mut embeddings,
                document.text.clone(),
                document.vector.clone(),
                fixture.name(),
            )?;
        }
        for query in fixture.queries() {
            insert_embedding(
                &mut embeddings,
                query.text.clone(),
                query.vector.clone(),
                fixture.name(),
            )?;
        }

        Ok(Self {
            dimensions: fixture.dimensions(),
            embeddings,
        })
    }

    /// Vector dimensionality returned by this model.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl EmbeddingModel for FixtureEmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, dims: Option<usize>) -> Self {
        Self {
            dimensions: dims.unwrap_or_default(),
            embeddings: HashMap::new(),
        }
    }

    fn ndims(&self) -> usize {
        self.dimensions
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let texts = texts.into_iter().collect::<Vec<_>>();
        let mut output = Vec::with_capacity(texts.len());

        for text in texts {
            let vector = self.embeddings.get(&text).cloned().ok_or_else(|| {
                EmbeddingError::ResponseError(format!(
                    "ANN fixture embedding missing for text '{text}'"
                ))
            })?;
            if vector.len() != self.dimensions {
                return Err(EmbeddingError::ResponseError(format!(
                    "ANN fixture embedding for text '{text}' has {} dimensions, expected {}",
                    vector.len(),
                    self.dimensions
                )));
            }
            output.push(Embedding {
                document: text,
                vec: vector,
            });
        }

        Ok(output)
    }
}

/// Options for asserting vector-store conformance fixtures.
#[derive(Clone, Debug)]
pub struct AssertOptions {
    /// Maximum tolerated absolute difference between expected and actual scores.
    pub score_epsilon: f64,
    /// Minimum expected-id recall required for approximate backends.
    pub min_recall: f64,
    /// Whether returned IDs must match fixture order exactly.
    pub exact_order: bool,
}

impl AssertOptions {
    /// Requires exact ordered results and score agreement.
    pub fn exact() -> Self {
        Self::default()
    }

    /// Sets the score epsilon.
    pub fn score_epsilon(mut self, score_epsilon: f64) -> Self {
        self.score_epsilon = score_epsilon;
        self
    }

    /// Allows approximate backends to satisfy the fixture by recall.
    pub fn recall_at_least(mut self, min_recall: f64) -> Self {
        self.min_recall = min_recall;
        self.exact_order = false;
        self
    }
}

impl Default for AssertOptions {
    fn default() -> Self {
        Self {
            score_epsilon: 1e-5,
            min_recall: 1.0,
            exact_order: true,
        }
    }
}

/// Runs all queries in `fixture` against `index` and asserts conformance.
///
/// The document ID callback verifies that `top_n` returns the same identity as
/// `top_n_ids`, even when the backend deserializes full document payloads.
pub async fn assert_vector_store_fixture<I, F, D, DocId>(
    index: &I,
    fixture: &AnnFixture,
    options: AssertOptions,
    document_id: DocId,
) -> Result<()>
where
    I: VectorStoreIndex<Filter = F>,
    F: SearchFilter + WasmCompatSend + WasmCompatSync,
    D: DeserializeOwned + Debug + WasmCompatSend,
    DocId: Fn(&D) -> String,
{
    fixture.validate()?;
    ensure!(
        options.score_epsilon.is_finite() && options.score_epsilon >= 0.0,
        "score epsilon must be finite and non-negative"
    );
    ensure!(
        (0.0..=1.0).contains(&options.min_recall),
        "minimum recall must be between 0.0 and 1.0"
    );

    for query in fixture.queries() {
        let expected = fixture.expected_neighbors(query)?;
        let id_results = index
            .top_n_ids(request_for_query::<F>(query))
            .await
            .with_context(|| {
                format!(
                    "top_n_ids failed for fixture '{}' query '{}'",
                    fixture.name(),
                    query.id
                )
            })?;
        assert_neighbors(
            fixture,
            query,
            &expected,
            &id_results,
            &options,
            "top_n_ids",
        )?;

        let doc_results = index
            .top_n::<D>(request_for_query::<F>(query))
            .await
            .with_context(|| {
                format!(
                    "top_n failed for fixture '{}' query '{}'",
                    fixture.name(),
                    query.id
                )
            })?;

        ensure!(
            doc_results.len() == id_results.len(),
            "fixture '{}' query '{}' returned {} top_n rows but {} top_n_ids rows",
            fixture.name(),
            query.id,
            doc_results.len(),
            id_results.len()
        );

        for ((doc_score, doc_id, document), (id_score, id)) in
            doc_results.iter().zip(id_results.iter())
        {
            ensure!(
                doc_id == id,
                "fixture '{}' query '{}' top_n id '{}' did not match top_n_ids id '{}'",
                fixture.name(),
                query.id,
                doc_id,
                id
            );
            ensure_score_close(
                *doc_score,
                *id_score,
                options.score_epsilon,
                fixture.name(),
                &query.id,
                id,
                "top_n/top_n_ids",
            )?;
            let decoded_id = document_id(document);
            ensure!(
                decoded_id == *doc_id,
                "fixture '{}' query '{}' decoded document id '{}' did not match returned id '{}'",
                fixture.name(),
                query.id,
                decoded_id,
                doc_id
            );
        }
    }

    Ok(())
}

fn request_for_query<F>(query: &FixtureQuery) -> VectorSearchRequest<F>
where
    F: SearchFilter,
{
    let mut builder = VectorSearchRequest::<F>::builder()
        .query(query.text.clone())
        .samples(query.top_k);
    if let Some(threshold) = query.threshold {
        builder = builder.threshold(threshold);
    }
    builder.build()
}

fn assert_neighbors(
    fixture: &AnnFixture,
    query: &FixtureQuery,
    expected: &[ExpectedNeighbor],
    actual: &[(f64, String)],
    options: &AssertOptions,
    label: &str,
) -> Result<()> {
    let expected_ids = expected
        .iter()
        .map(|neighbor| neighbor.id.as_str())
        .collect::<Vec<_>>();
    let actual_ids = actual.iter().map(|(_, id)| id.as_str()).collect::<Vec<_>>();

    if options.exact_order {
        ensure!(
            actual.len() == expected.len(),
            "fixture '{}' query '{}' {label} returned {} rows, expected {}; expected ids: {:?}; actual ids: {:?}",
            fixture.name(),
            query.id,
            actual.len(),
            expected.len(),
            expected_ids,
            actual_ids
        );

        for (rank, ((actual_score, actual_id), expected)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            ensure!(
                actual_id == &expected.id,
                "fixture '{}' query '{}' {label} rank {rank} returned id '{}' where '{}' was expected; expected ids: {:?}; actual ids: {:?}",
                fixture.name(),
                query.id,
                actual_id,
                expected.id,
                expected_ids,
                actual_ids
            );
            let delta = (*actual_score - expected.score).abs();
            ensure!(
                actual_score.is_finite(),
                "fixture '{}' query '{}' {label} rank {rank} score for '{}' was not finite: {}",
                fixture.name(),
                query.id,
                expected.id,
                actual_score
            );
            ensure!(
                delta <= options.score_epsilon,
                "fixture '{}' query '{}' {label} rank {rank} score for '{}' was {}, expected {} within {}; delta {}; expected ids: {:?}; actual ids: {:?}",
                fixture.name(),
                query.id,
                expected.id,
                actual_score,
                expected.score,
                options.score_epsilon,
                delta,
                expected_ids,
                actual_ids
            );
        }
    } else {
        let hits = expected
            .iter()
            .filter(|expected| {
                actual
                    .iter()
                    .any(|(_, actual_id)| actual_id == &expected.id)
            })
            .count();
        let recall = hits as f64 / expected.len() as f64;
        ensure!(
            recall >= options.min_recall,
            "fixture '{}' query '{}' {label} recall {recall:.3} was below required {:.3}; expected ids: {:?}; actual ids: {:?}",
            fixture.name(),
            query.id,
            options.min_recall,
            expected_ids,
            actual_ids
        );

        for expected in expected {
            if let Some((actual_score, _)) = actual
                .iter()
                .find(|(_, actual_id)| actual_id == &expected.id)
            {
                ensure_score_close(
                    *actual_score,
                    expected.score,
                    options.score_epsilon,
                    fixture.name(),
                    &query.id,
                    &expected.id,
                    label,
                )?;
            }
        }
    }

    Ok(())
}

fn score_vector(metric: AnnMetric, query: &[f64], document: &[f64]) -> Result<f64> {
    ensure!(
        query.len() == document.len(),
        "query vector has {} dimensions but document vector has {}",
        query.len(),
        document.len()
    );
    let score = match metric {
        AnnMetric::Cosine => {
            let (dot, query_norm, document_norm) = query.iter().zip(document.iter()).fold(
                (0.0f32, 0.0f32, 0.0f32),
                |(dot, query_norm, document_norm), (query_value, document_value)| {
                    let query_value = *query_value as f32;
                    let document_value = *document_value as f32;
                    (
                        dot + query_value * document_value,
                        query_norm + query_value * query_value,
                        document_norm + document_value * document_value,
                    )
                },
            );
            f64::from(dot / (query_norm.sqrt() * document_norm.sqrt()))
        }
        AnnMetric::L1 => {
            let distance = query
                .iter()
                .zip(document.iter())
                .map(|(query_value, document_value)| {
                    ((*query_value as f32) - (*document_value as f32)).abs()
                })
                .sum::<f32>();
            -f64::from(distance)
        }
        AnnMetric::L2 => {
            let squared_distance = query
                .iter()
                .zip(document.iter())
                .map(|(query_value, document_value)| {
                    let delta = (*query_value as f32) - (*document_value as f32);
                    delta * delta
                })
                .sum::<f32>();
            -f64::from(squared_distance.sqrt())
        }
    };
    ensure!(score.is_finite(), "computed score was not finite");
    Ok(score)
}

fn compare_neighbor_score_desc(lhs: &ExpectedNeighbor, rhs: &ExpectedNeighbor) -> Ordering {
    rhs.score
        .partial_cmp(&lhs.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| lhs.id.cmp(&rhs.id))
}

fn ensure_score_close(
    actual: f64,
    expected: f64,
    epsilon: f64,
    fixture_name: &str,
    query_id: &str,
    document_id: &str,
    label: &str,
) -> Result<()> {
    ensure!(
        actual.is_finite(),
        "fixture '{fixture_name}' query '{query_id}' {label} score for '{document_id}' was not finite: {actual}"
    );
    ensure!(
        (actual - expected).abs() <= epsilon,
        "fixture '{fixture_name}' query '{query_id}' {label} score for '{document_id}' was {actual}, expected {expected} within {epsilon}"
    );
    Ok(())
}

fn ensure_non_empty(value: &str, field: &str, fixture_name: &str) -> Result<()> {
    ensure!(
        !value.trim().is_empty(),
        "ANN fixture '{fixture_name}' {field} must not be empty"
    );
    Ok(())
}

fn ensure_vector_dimensions(
    vector: &[f64],
    dimensions: usize,
    fixture_name: &str,
    label: &str,
) -> Result<()> {
    ensure!(
        vector.len() == dimensions,
        "ANN fixture '{fixture_name}' vector '{label}' has {} dimensions, expected {dimensions}",
        vector.len()
    );
    ensure!(
        vector.iter().all(|value| value.is_finite()),
        "ANN fixture '{fixture_name}' vector '{label}' contains a non-finite value"
    );
    Ok(())
}

fn insert_embedding(
    embeddings: &mut HashMap<String, Vec<f64>>,
    text: String,
    vector: Vec<f64>,
    fixture_name: &str,
) -> Result<()> {
    if let Some(existing) = embeddings.insert(text.clone(), vector.clone())
        && existing != vector
    {
        bail!("ANN fixture '{fixture_name}' has conflicting vectors for text '{text}'");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result, ensure};
    use rig_core::embeddings::{
        Embed, EmbedError, EmbeddingModel, EmbeddingsBuilder, TextEmbedder,
    };
    use rig_core::vector_store::VectorStoreError;
    use rig_core::vector_store::request::Filter;
    use serde::Deserialize;
    use serde_json::json;

    use super::*;

    #[test]
    fn validation_accepts_expected_values_that_match_vector_oracle() -> Result<()> {
        let fixture = AnnFixture::from_json(&simple_cosine_fixture_json(json!([
            {"id": "doc_a", "score": 1.0},
            {"id": "doc_b", "score": 0.0},
            {"id": "doc_c", "score": -1.0}
        ])))?;
        let query = fixture
            .queries()
            .first()
            .context("fixture should include a query")?;
        let expected = fixture.expected_neighbors(query)?;
        let expected_ids = expected
            .iter()
            .map(|neighbor| neighbor.id.as_str())
            .collect::<Vec<_>>();

        ensure!(
            expected_ids == ["doc_a", "doc_b", "doc_c"],
            "unexpected oracle order: {expected_ids:?}"
        );

        Ok(())
    }

    #[test]
    fn validation_rejects_expected_order_that_differs_from_vector_oracle() -> Result<()> {
        let result = AnnFixture::from_json(&simple_cosine_fixture_json(json!([
            {"id": "doc_b", "score": 0.0},
            {"id": "doc_a", "score": 1.0},
            {"id": "doc_c", "score": -1.0}
        ])));

        ensure!(
            result.is_err(),
            "fixture validation should reject wrong order"
        );

        Ok(())
    }

    #[test]
    fn validation_rejects_expected_score_that_differs_from_vector_oracle() -> Result<()> {
        let result = AnnFixture::from_json(&simple_cosine_fixture_json(json!([
            {"id": "doc_a", "score": 0.5},
            {"id": "doc_b", "score": 0.0},
            {"id": "doc_c", "score": -1.0}
        ])));

        ensure!(
            result.is_err(),
            "fixture validation should reject wrong score"
        );

        Ok(())
    }

    #[test]
    fn threshold_equality_is_included_in_vector_oracle() -> Result<()> {
        let fixture = AnnFixture::from_json(&threshold_equality_fixture_json())?;
        let query = fixture
            .queries()
            .first()
            .context("fixture should include a query")?;
        let expected = fixture.expected_neighbors(query)?;
        let expected_ids = expected
            .iter()
            .map(|neighbor| neighbor.id.as_str())
            .collect::<Vec<_>>();

        ensure!(
            expected_ids == ["doc_a", "doc_b"],
            "threshold equality should include doc_b: {expected_ids:?}"
        );

        Ok(())
    }

    #[test]
    fn equal_scores_are_ordered_by_document_id() -> Result<()> {
        let fixture = AnnFixture::from_json(&tie_order_fixture_json())?;
        let query = fixture
            .queries()
            .first()
            .context("fixture should include a query")?;
        let expected = fixture.expected_neighbors(query)?;
        let expected_ids = expected
            .iter()
            .map(|neighbor| neighbor.id.as_str())
            .collect::<Vec<_>>();

        ensure!(
            expected_ids == ["doc_a", "doc_b"],
            "equal scores should be ordered by document id: {expected_ids:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn fixture_embedding_model_rejects_missing_text() -> Result<()> {
        let fixture = AnnFixture::from_json(&simple_cosine_fixture_json(json!([
            {"id": "doc_a", "score": 1.0},
            {"id": "doc_b", "score": 0.0},
            {"id": "doc_c", "score": -1.0}
        ])))?;
        let model = FixtureEmbeddingModel::from_fixture(&fixture)?;
        let result = model.embed_text("missing fixture text").await;

        ensure!(
            result.is_err(),
            "fixture embedding model should reject missing text"
        );

        Ok(())
    }

    #[tokio::test]
    async fn fixture_embedding_model_can_build_document_embeddings() -> Result<()> {
        let fixture = AnnFixture::from_json(&simple_cosine_fixture_json(json!([
            {"id": "doc_a", "score": 1.0},
            {"id": "doc_b", "score": 0.0},
            {"id": "doc_c", "score": -1.0}
        ])))?;
        let model = FixtureEmbeddingModel::from_fixture(&fixture)?;
        let embeddings = EmbeddingsBuilder::new(model)
            .documents([
                EmbeddableTestDocument::new("doc:a"),
                EmbeddableTestDocument::new("doc:b"),
            ])?
            .build()
            .await?;
        let vectors_by_text = embeddings
            .into_iter()
            .map(|(document, embedding)| (document.text, embedding.first().vec))
            .collect::<HashMap<_, _>>();

        ensure!(
            vectors_by_text.get("doc:a") == Some(&vec![1.0, 0.0]),
            "unexpected embedding for doc:a: {vectors_by_text:?}"
        );
        ensure!(
            vectors_by_text.get("doc:b") == Some(&vec![0.0, 1.0]),
            "unexpected embedding for doc:b: {vectors_by_text:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn approximate_recall_mode_accepts_partial_expected_ids() -> Result<()> {
        let fixture = AnnFixture::from_json(&simple_cosine_fixture_json(json!([
            {"id": "doc_a", "score": 1.0},
            {"id": "doc_b", "score": 0.0},
            {"id": "doc_c", "score": -1.0}
        ])))?;
        let query = fixture
            .queries()
            .first()
            .context("fixture should include a query")?;
        let first_expected = fixture
            .expected_neighbors(query)?
            .into_iter()
            .next()
            .context("oracle should produce at least one neighbor")?;
        let index = StaticIndex {
            results: vec![(first_expected.score, first_expected.id)],
        };

        assert_vector_store_fixture(
            &index,
            &fixture,
            AssertOptions::exact().recall_at_least(0.3),
            |document: &TestDocument| document.id.clone(),
        )
        .await?;

        Ok(())
    }

    #[derive(Clone)]
    struct StaticIndex {
        results: Vec<(f64, String)>,
    }

    impl VectorStoreIndex for StaticIndex {
        type Filter = Filter<serde_json::Value>;

        fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
            &self,
            _req: VectorSearchRequest<Self::Filter>,
        ) -> impl std::future::Future<
            Output = std::result::Result<Vec<(f64, String, T)>, VectorStoreError>,
        > + WasmCompatSend {
            let results = self.results.clone();
            async move {
                results
                    .iter()
                    .map(|(score, id)| {
                        let document = serde_json::from_value(json!({ "id": id }))?;
                        Ok((*score, id.clone(), document))
                    })
                    .collect::<std::result::Result<Vec<_>, VectorStoreError>>()
            }
        }

        fn top_n_ids(
            &self,
            _req: VectorSearchRequest<Self::Filter>,
        ) -> impl std::future::Future<
            Output = std::result::Result<Vec<(f64, String)>, VectorStoreError>,
        > + WasmCompatSend {
            let results = self.results.clone();
            async move { Ok(results) }
        }
    }

    #[derive(Debug, Deserialize)]
    struct TestDocument {
        id: String,
    }

    struct EmbeddableTestDocument {
        text: String,
    }

    impl EmbeddableTestDocument {
        fn new(text: impl Into<String>) -> Self {
            Self { text: text.into() }
        }
    }

    impl Embed for EmbeddableTestDocument {
        fn embed(&self, embedder: &mut TextEmbedder) -> std::result::Result<(), EmbedError> {
            embedder.embed(self.text.clone());
            Ok(())
        }
    }

    fn simple_cosine_fixture_json(expected: serde_json::Value) -> String {
        json!({
            "name": "simple_cosine",
            "metric": "cosine",
            "dimensions": 2,
            "documents": [
                {"id": "doc_a", "text": "doc:a", "vector": [1.0, 0.0]},
                {"id": "doc_b", "text": "doc:b", "vector": [0.0, 1.0]},
                {"id": "doc_c", "text": "doc:c", "vector": [-1.0, 0.0]}
            ],
            "queries": [
                {
                    "id": "query",
                    "text": "query",
                    "vector": [1.0, 0.0],
                    "top_k": 3,
                    "threshold": null,
                    "expected": expected
                }
            ]
        })
        .to_string()
    }

    fn threshold_equality_fixture_json() -> String {
        json!({
            "name": "threshold_equality",
            "metric": "cosine",
            "dimensions": 2,
            "documents": [
                {"id": "doc_a", "text": "doc:a", "vector": [1.0, 0.0]},
                {"id": "doc_b", "text": "doc:b", "vector": [0.0, 1.0]},
                {"id": "doc_c", "text": "doc:c", "vector": [-1.0, 0.0]}
            ],
            "queries": [
                {
                    "id": "query",
                    "text": "query",
                    "vector": [1.0, 0.0],
                    "top_k": 3,
                    "threshold": 0.0,
                    "expected": [
                        {"id": "doc_a", "score": 1.0},
                        {"id": "doc_b", "score": 0.0}
                    ]
                }
            ]
        })
        .to_string()
    }

    fn tie_order_fixture_json() -> String {
        json!({
            "name": "tie_order",
            "metric": "cosine",
            "dimensions": 2,
            "documents": [
                {"id": "doc_b", "text": "doc:b", "vector": [1.0, 0.0]},
                {"id": "doc_a", "text": "doc:a", "vector": [1.0, 0.0]}
            ],
            "queries": [
                {
                    "id": "query",
                    "text": "query",
                    "vector": [1.0, 0.0],
                    "top_k": 2,
                    "threshold": null,
                    "expected": [
                        {"id": "doc_a", "score": 1.0},
                        {"id": "doc_b", "score": 1.0}
                    ]
                }
            ]
        })
        .to_string()
    }
}
