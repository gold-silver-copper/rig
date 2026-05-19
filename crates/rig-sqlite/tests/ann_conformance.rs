use std::os::raw::c_char;
use std::sync::Once;

use anyhow::Result;
use rig_core::embeddings::{EmbedError, Embedding, EmbeddingsBuilder, TextEmbedder};
use rig_core::vector_store::InsertDocuments;
use rig_core::{Embed, OneOrMany};
use rig_sqlite::{
    Column, ColumnValue, SqliteDistanceMetric, SqliteVectorIndex, SqliteVectorStore,
    SqliteVectorStoreTable,
};
use rig_vector_testkit::{
    AnnFixture, AnnMetric, AssertOptions, FixtureEmbeddingModel, assert_source_ground_truth_recall,
    assert_vector_store_fixture,
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
            assert_source_ground_truth_recall(&index, &fixture, 1.0).await?;
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
