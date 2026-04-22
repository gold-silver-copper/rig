//! Llamafile embeddings smoke tests.

use anyhow::Result;
#[cfg(feature = "derive")]
use rig::Embed;
use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingModel;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

use super::support;

#[cfg(feature = "derive")]
#[derive(Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn embeddings_smoke() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let client = support::client()?;
    let model = client.embedding_model(support::model_name())?;

    let embeddings = model
        .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
        .await?;

    assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    Ok(())
}

#[cfg(feature = "derive")]
#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080 and --features derive"]
async fn derive_document_embeddings() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let client = support::client()?;
    let embeddings = client
        .embeddings(support::model_name())?
        .document(Greetings {
            message: "Hello, world!".to_string(),
        })?
        .document(Greetings {
            message: "Goodbye, world!".to_string(),
        })?
        .build()
        .await?;

    assert_eq!(embeddings.len(), 2);
    for (_document, embeddings_for_document) in embeddings {
        let mut dims = None;
        for embedding in embeddings_for_document {
            assert!(
                !embedding.vec.is_empty(),
                "expected each embedding vector to be non-empty"
            );

            match dims {
                Some(expected_dims) => assert_eq!(embedding.vec.len(), expected_dims),
                None => dims = Some(embedding.vec.len()),
            }
        }
    }
    Ok(())
}
