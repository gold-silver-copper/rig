use std::os::raw::c_char;
use std::sync::Once;

use anyhow::Result;
use rig_core::embeddings::{EmbedError, Embedding, EmbeddingsBuilder, TextEmbedder};
use rig_core::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig_core::vector_store::{InsertDocuments, VectorStoreIndex};
use rig_core::{Embed, OneOrMany};
use rig_sqlite::{
    Column, ColumnValue, SqliteDistanceMetric, SqliteSearchFilter, SqliteVectorIndex,
    SqliteVectorStore, SqliteVectorStoreTable,
};
use rig_vector_testkit::{
    AnnFixture, AnnMetric, AssertOptions, FixtureDocument, FixtureEmbeddingModel, FixtureQuery,
    assert_source_ground_truth_exact, assert_vector_store_fixture, compute_expected_neighbors,
};
use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
use serde::{Deserialize, Serialize};
use sqlite_vec::sqlite3_vec_init;
use tokio_rusqlite::Connection;

const FIXTURES: [&str; 4] = [
    include_str!(
        "../../rig-vector-testkit/fixtures/ann/benchmark_derived_ann_random_xs_20_angular_cosine.json"
    ),
    include_str!(
        "../../rig-vector-testkit/fixtures/ann/benchmark_derived_ann_random_xs_20_angular_l1.json"
    ),
    include_str!(
        "../../rig-vector-testkit/fixtures/ann/benchmark_derived_ann_random_xs_20_angular_l2.json"
    ),
    include_str!(
        "../../rig-vector-testkit/fixtures/ann/benchmark_derived_vibe_glove_200_cosine.json"
    ),
];
const METADATA_QUERY_TEXT: &str = "sqlite:metadata-filter:query";
#[derive(Clone, Debug, Deserialize, Serialize)]
struct AnnDocument {
    id: String,
    text: String,
}

impl Embed for AnnDocument {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.text.clone());
        Ok(())
    }
}

impl SqliteVectorStoreTable for AnnDocument {
    fn name() -> &'static str {
        "ann_documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("text", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("text", Box::new(self.text.clone())),
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct MetadataAnnDocument {
    id: String,
    text: String,
    category: String,
    priority: i64,
    rating: f64,
    published: bool,
}

impl Embed for MetadataAnnDocument {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.text.clone());
        Ok(())
    }
}

impl SqliteVectorStoreTable for MetadataAnnDocument {
    fn name() -> &'static str {
        "ann_metadata_documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("text", "TEXT"),
            Column::new("category", "TEXT").indexed(),
            Column::new("priority", "INTEGER").indexed(),
            Column::new("rating", "FLOAT").indexed(),
            Column::new("published", "BOOLEAN").indexed(),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("text", Box::new(self.text.clone())),
            ("category", Box::new(self.category.clone())),
            ("priority", Box::new(self.priority)),
            ("rating", Box::new(self.rating)),
            ("published", Box::new(self.published)),
        ]
    }
}

#[derive(Clone, Debug)]
struct MetadataSeed {
    id: &'static str,
    category: &'static str,
    priority: i64,
    rating: f64,
    published: bool,
    vector: Vec<f64>,
}

type SqliteExtensionFn =
    unsafe extern "C" fn(*mut sqlite3, *mut *mut c_char, *const sqlite3_api_routines) -> i32;

#[tokio::test]
async fn sqlite_matches_ann_conformance_fixtures() -> Result<()> {
    register_sqlite_vec_extension();

    for fixture_json in FIXTURES {
        let fixture = AnnFixture::from_json(fixture_json)?;
        let index = sqlite_index_for_fixture(&fixture).await?;

        assert_vector_store_fixture(
            &index,
            &fixture,
            AssertOptions::exact().score_epsilon(1e-4),
            |document: &AnnDocument| document.id.clone(),
        )
        .await?;
        if fixture_has_native_source_ground_truth(&fixture) {
            assert_source_ground_truth_exact(&index, &fixture, 1e-4, |document: &AnnDocument| {
                document.id.clone()
            })
            .await?;
        }
    }

    Ok(())
}

#[tokio::test]
async fn sqlite_insert_documents_matches_ann_conformance_fixture() -> Result<()> {
    register_sqlite_vec_extension();

    for fixture_json in FIXTURES {
        let fixture = AnnFixture::from_json(fixture_json)?;
        let index = sqlite_index_for_fixture_with_insert_documents(&fixture).await?;

        assert_vector_store_fixture(
            &index,
            &fixture,
            AssertOptions::exact().score_epsilon(1e-4),
            |document: &AnnDocument| document.id.clone(),
        )
        .await?;
    }

    Ok(())
}

#[tokio::test]
async fn sqlite_metadata_filters_are_applied_during_ann_candidate_search() -> Result<()> {
    register_sqlite_vec_extension();

    let fixture = metadata_fixture()?;
    let index = sqlite_metadata_index_for_fixture(&fixture).await?;

    let unfiltered_req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query(METADATA_QUERY_TEXT)
        .samples(2)
        .build();
    let unfiltered = index.top_n_ids(unfiltered_req).await?;
    assert_scored_ids(
        &unfiltered,
        &[("nearest-excluded", 1.0), ("second-excluded", 0.99)],
        "unfiltered top_n_ids",
    )?;

    let filter = SqliteSearchFilter::eq("category", serde_json::json!("docs"))
        .and(SqliteSearchFilter::between(
            "priority".to_string(),
            1_i64..=10_i64,
        ))
        .and(SqliteSearchFilter::gt("rating", serde_json::json!(0.8)))
        .and(SqliteSearchFilter::eq("published", serde_json::json!(true)));
    let filtered_req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query(METADATA_QUERY_TEXT)
        .samples(2)
        .filter(filter)
        .build();
    let expected = [("filtered-best", 0.8), ("filtered-second", 0.6)];

    let id_results = index.top_n_ids(filtered_req.clone()).await?;
    assert_scored_ids(&id_results, &expected, "filtered top_n_ids")?;

    let doc_results = index.top_n::<MetadataAnnDocument>(filtered_req).await?;
    let scored_doc_ids = doc_results
        .iter()
        .map(|(score, id, document)| {
            anyhow::ensure!(
                id == &document.id,
                "top_n returned id '{id}' but decoded document id was '{}'",
                document.id
            );
            anyhow::ensure!(
                document.category == "docs"
                    && (1..=10).contains(&document.priority)
                    && document.rating > 0.8
                    && document.published,
                "top_n returned document that does not satisfy metadata filter: {document:?}"
            );
            Ok((*score, id.clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    assert_scored_ids(&scored_doc_ids, &expected, "filtered top_n")?;

    Ok(())
}

fn fixture_has_native_source_ground_truth(fixture: &AnnFixture) -> bool {
    fixture.queries().iter().any(|query| {
        query
            .source_ground_truth
            .as_ref()
            .is_some_and(|ground_truth| {
                ground_truth.metric == fixture.metric()
                    && ground_truth
                        .neighbors
                        .iter()
                        .any(|neighbor| neighbor.id.is_some())
            })
    })
}

async fn sqlite_index_for_fixture(
    fixture: &AnnFixture,
) -> Result<SqliteVectorIndex<FixtureEmbeddingModel, AnnDocument>> {
    let conn = Connection::open(format!("file:{}?mode=memory", fixture.name())).await?;
    let model = FixtureEmbeddingModel::from_fixture(fixture)?;
    let vector_store =
        SqliteVectorStore::with_distance_metric(conn, &model, sqlite_metric(fixture.metric()))
            .await?;

    // These conformance fixtures inject precomputed embeddings so the test
    // covers SQLite query/index behavior, not the normal InsertDocuments path.
    vector_store.add_rows(rows_for_fixture(fixture)).await?;

    Ok(vector_store.index(model))
}

async fn sqlite_index_for_fixture_with_insert_documents(
    fixture: &AnnFixture,
) -> Result<SqliteVectorIndex<FixtureEmbeddingModel, AnnDocument>> {
    let conn = Connection::open(format!(
        "file:{}_insert_documents?mode=memory",
        fixture.name()
    ))
    .await?;
    let model = FixtureEmbeddingModel::from_fixture(fixture)?;
    let vector_store =
        SqliteVectorStore::with_distance_metric(conn, &model, sqlite_metric(fixture.metric()))
            .await?;
    let documents = documents_for_fixture(fixture);
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(documents)?
        .build()
        .await?;

    vector_store.insert_documents(embeddings).await?;

    Ok(vector_store.index(model))
}

async fn sqlite_metadata_index_for_fixture(
    fixture: &AnnFixture,
) -> Result<SqliteVectorIndex<FixtureEmbeddingModel, MetadataAnnDocument>> {
    let conn = Connection::open(format!("file:{}_metadata?mode=memory", fixture.name())).await?;
    let model = FixtureEmbeddingModel::from_fixture(fixture)?;
    let vector_store =
        SqliteVectorStore::with_distance_metric(conn, &model, SqliteDistanceMetric::Cosine).await?;

    vector_store.add_rows(metadata_rows()).await?;

    Ok(vector_store.index(model))
}

fn documents_for_fixture(fixture: &AnnFixture) -> Vec<AnnDocument> {
    fixture
        .documents()
        .iter()
        .map(|document| AnnDocument {
            id: document.id.clone(),
            text: document.text.clone(),
        })
        .collect()
}

fn metadata_fixture() -> Result<AnnFixture> {
    let documents = metadata_seeds()
        .iter()
        .map(|seed| FixtureDocument {
            id: seed.id.to_string(),
            text: seed.id.to_string(),
            vector: seed.vector.clone(),
        })
        .collect::<Vec<_>>();
    let query_vector = vec![1.0, 0.0];
    let expected =
        compute_expected_neighbors(AnnMetric::Cosine, &documents, &query_vector, 3, None)?;

    AnnFixture::new(
        "sqlite_metadata_filter_conformance".to_string(),
        AnnMetric::Cosine,
        2,
        None,
        documents,
        vec![FixtureQuery {
            id: "metadata-query".to_string(),
            text: METADATA_QUERY_TEXT.to_string(),
            vector: query_vector,
            top_k: 3,
            threshold: None,
            expected,
            source_ground_truth: None,
        }],
    )
}

fn metadata_rows() -> Vec<(MetadataAnnDocument, OneOrMany<Embedding>)> {
    metadata_seeds()
        .into_iter()
        .map(|seed| {
            (
                MetadataAnnDocument {
                    id: seed.id.to_string(),
                    text: seed.id.to_string(),
                    category: seed.category.to_string(),
                    priority: seed.priority,
                    rating: seed.rating,
                    published: seed.published,
                },
                OneOrMany::one(Embedding {
                    document: seed.id.to_string(),
                    vec: seed.vector,
                }),
            )
        })
        .collect()
}

fn metadata_seeds() -> Vec<MetadataSeed> {
    vec![
        MetadataSeed {
            id: "nearest-excluded",
            category: "misc",
            priority: 1,
            rating: 0.99,
            published: true,
            vector: vec![1.0, 0.0],
        },
        MetadataSeed {
            id: "second-excluded",
            category: "archive",
            priority: 2,
            rating: 0.99,
            published: true,
            vector: vec![0.99, 0.14106735979665894],
        },
        MetadataSeed {
            id: "wrong-priority",
            category: "docs",
            priority: 20,
            rating: 0.99,
            published: true,
            vector: vec![0.95, 0.31224989991991997],
        },
        MetadataSeed {
            id: "wrong-published",
            category: "docs",
            priority: 3,
            rating: 0.99,
            published: false,
            vector: vec![0.9, 0.4358898943540673],
        },
        MetadataSeed {
            id: "filtered-best",
            category: "docs",
            priority: 3,
            rating: 0.95,
            published: true,
            vector: vec![0.8, 0.6],
        },
        MetadataSeed {
            id: "wrong-rating",
            category: "docs",
            priority: 4,
            rating: 0.7,
            published: true,
            vector: vec![0.7, 0.714142842854285],
        },
        MetadataSeed {
            id: "filtered-second",
            category: "docs",
            priority: 8,
            rating: 0.9,
            published: true,
            vector: vec![0.6, 0.8],
        },
    ]
}

fn rows_for_fixture(fixture: &AnnFixture) -> Vec<(AnnDocument, OneOrMany<Embedding>)> {
    fixture
        .documents()
        .iter()
        .map(|document| {
            (
                AnnDocument {
                    id: document.id.clone(),
                    text: document.text.clone(),
                },
                OneOrMany::one(Embedding {
                    document: document.text.clone(),
                    vec: document.vector.clone(),
                }),
            )
        })
        .collect()
}

fn assert_scored_ids(
    actual: &[(f64, String)],
    expected: &[(&str, f64)],
    context: &str,
) -> Result<()> {
    anyhow::ensure!(
        actual.len() == expected.len(),
        "{context} returned {} rows, expected {}; actual={actual:?}, expected={expected:?}",
        actual.len(),
        expected.len()
    );
    for ((actual_score, actual_id), (expected_id, expected_score)) in actual.iter().zip(expected) {
        anyhow::ensure!(
            actual_id == expected_id,
            "{context} returned id '{actual_id}', expected '{expected_id}'; actual={actual:?}"
        );
        anyhow::ensure!(
            actual_score.is_finite(),
            "{context} score for '{actual_id}' was not finite: {actual_score}"
        );
        anyhow::ensure!(
            (actual_score - expected_score).abs() <= 1e-4,
            "{context} score for '{actual_id}' was {actual_score}, expected {expected_score}; actual={actual:?}"
        );
    }
    Ok(())
}

fn sqlite_metric(metric: AnnMetric) -> SqliteDistanceMetric {
    match metric {
        AnnMetric::Cosine => SqliteDistanceMetric::Cosine,
        AnnMetric::L1 => SqliteDistanceMetric::L1,
        AnnMetric::L2 => SqliteDistanceMetric::L2,
    }
}

fn register_sqlite_vec_extension() {
    static REGISTER_SQLITE_VEC: Once = Once::new();

    REGISTER_SQLITE_VEC.call_once(|| unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute::<*const (), SqliteExtensionFn>(
            sqlite3_vec_init as *const (),
        )));
    });
}
