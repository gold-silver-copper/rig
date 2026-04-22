use anyhow::{Context, Result, bail};
use rig::client::EmbeddingsClient;
use rig::providers::openai;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed,
    embeddings::EmbeddingsBuilder,
    vector_store::{InsertDocuments, VectorStoreIndex},
};
use rig_scylladb::{ScyllaDbVectorStore, create_session};
use serde::{Deserialize, Serialize};
use serde_json::json;
use testcontainers::{
    GenericImage,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

const SCYLLA_PORT: u16 = 9042;

fn create_embedding_vector(index: usize) -> Vec<f64> {
    let mut vec = vec![0.0; 1536];
    if let Some(slot) = vec.get_mut(index) {
        *slot = 1.0;
    }
    vec
}

#[derive(Embed, Clone, Serialize, Deserialize, Debug, PartialEq)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::test]
#[ignore = "requires Docker and ScyllaDB container"]
async fn vector_search_test() {
    assert_ok(async {
        let container = start_container().await?;

        let host = container
            .get_host()
            .await
            .context("failed to resolve scylladb container host")?
            .to_string();
        let port = container
            .get_host_port_ipv4(SCYLLA_PORT)
            .await
            .context("failed to resolve scylladb container port")?;

    println!("Container started on host:port {host}:{port}");

    // Wait for ScyllaDB to be ready and retry connection
    println!("🔌 Attempting to connect to ScyllaDB at {host}:{port}...");
        let session = {
            let mut retries = 0;
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

                match create_session(&format!("{host}:{port}")).await {
                    Ok(session) => {
                        println!("✅ Successfully connected to ScyllaDB!");
                        break session;
                    }
                    Err(e) => {
                        retries += 1;
                        if retries >= 15 {
                            bail!("failed to connect to ScyllaDB after {retries} retries: {e:?}");
                        }
                        println!(
                            "🔄 Connection attempt {retries} failed, retrying in 5 seconds... (attempt {retries}/15): {e}"
                        );
                    }
                }
            }
        };

    println!("Connected to ScyllaDB");

    // Init fake openai service
    let openai_mock = create_openai_mock_service().await;
        let openai_client = openai::Client::builder()
            .api_key("TEST")
            .base_url(openai_mock.base_url())
            .build()
            .context("failed to build mock openai client")?;

        let model = openai_client
            .embedding_model(openai::TEXT_EMBEDDING_ADA_002)
            .context("embedding model should build")?;

    // Create test documents with mocked embeddings
    let words = vec![
        Word {
            id: "doc0".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        Word {
            id: "doc1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        Word {
            id: "doc2".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }
    ];

        let documents = EmbeddingsBuilder::new(model.clone())
            .documents(words)?
            .build()
            .await?;

    // Create ScyllaDB vector store
        let vector_store = ScyllaDbVectorStore::new(
            model.clone(),
            session,
            "test_keyspace",
            "test_words",
            1536,
        )
        .await?;

    // Insert documents into vector store
        vector_store.insert_documents(documents).await?;

    println!("Documents inserted successfully");
    let query = "What is a glarb?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build();

    // Test vector search
        let results = vector_store.top_n::<Word>(req.clone()).await?;

    assert_eq!(
        results.len(),
        1,
        "Expected one result, got {}",
        results.len()
    );

        let (distance, id, doc) = results
            .first()
            .cloned()
            .context("expected one vector search result")?;
        println!("Distance: {distance}, id: {id}, document: {doc:?}");

    assert_eq!(doc.id, "doc1");
    assert!(doc.definition.contains("glarb-glarb"));

    // Test top_n_ids
        let id_results = vector_store.top_n_ids(req).await?;

    assert_eq!(
        id_results.len(),
        1,
        "Expected one (id) result, got {}",
        id_results.len()
    );

        let (id_distance, result_id) = id_results
            .first()
            .cloned()
            .context("expected one vector search id result")?;
        println!("Distance: {id_distance}, id: {result_id}");

    assert_eq!(result_id, id);

    let query = "What is a linglingdong?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build();

    // Test with different query
        let results2 = vector_store.top_n::<Word>(req).await?;

        assert_eq!(results2.len(), 1);
        let (_, _, doc2) = results2
            .first()
            .context("expected one linglingdong result")?;
        assert_eq!(doc2.id, "doc2");
        assert!(doc2.definition.contains("linglingdong"));

        println!("✅ ScyllaDB integration test completed successfully!");

        Ok(())
    }
    .await);
}

async fn start_container() -> Result<testcontainers::ContainerAsync<GenericImage>> {
    use std::time::Duration;
    use testcontainers::ImageExt;

    println!("🚀 Starting ScyllaDB container (this may take 2-5 minutes)...");

    // Setup a local ScyllaDB container for testing. NOTE: docker service must be running.
    // ScyllaDB takes a long time to start, so we:
    // 1. Use a smaller/faster image configuration
    // 2. Use developer mode for faster startup
    // 3. Set a generous timeout
    // 4. Use multiple wait strategies for better reliability
    let container = GenericImage::new("scylladb/scylla", "5.4") // Use older, more stable version
        .with_wait_for(WaitFor::seconds(60)) // Wait 60 seconds then check port
        .with_exposed_port(SCYLLA_PORT.tcp())
        .with_env_var("SCYLLA_ARGS", "--smp 1 --memory 512M --overprovisioned 1 --skip-wait-for-gossip-to-settle 0 --developer-mode 1")
        .with_startup_timeout(Duration::from_secs(300)) // 5 minutes timeout
        .start()
        .await
        .context("failed to start ScyllaDB container")?;

    println!("✅ ScyllaDB container started successfully!");
    Ok(container)
}

async fn create_openai_mock_service() -> httpmock::MockServer {
    let server = httpmock::MockServer::start();

    // Mock for multiple documents (integration test)
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .json_body(json!({
                "input": [
                    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets",
                    "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
                    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans."
                ],
                "model": "text-embedding-ada-002",
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "object": "list",
                "data": [
                  {
                    "object": "embedding",
                    "embedding": create_embedding_vector(0),
                    "index": 0
                  },
                  {
                    "object": "embedding",
                    "embedding": create_embedding_vector(1),
                    "index": 1
                  },
                  {
                    "object": "embedding",
                    "embedding": create_embedding_vector(2),
                    "index": 2
                  }
                ],
                "model": "text-embedding-ada-002",
                "usage": {
                  "prompt_tokens": 8,
                  "total_tokens": 8
                }
            }));
    });

    // Mock for single document (unit test)
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .json_body(json!({
                "input": ["Test definition"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "object": "list",
                "data": [
                  {
                    "object": "embedding",
                    "embedding": create_embedding_vector(0),
                    "index": 0
                  }
                ],
                "model": "text-embedding-ada-002",
                "usage": {
                  "prompt_tokens": 8,
                  "total_tokens": 8
                }
            }));
    });

    // Mock for search queries
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .json_body(json!({
                "input": ["What is a glarb?"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "object": "list",
                "data": [
                  {
                    "object": "embedding",
                    "embedding": create_embedding_vector(1), // Match doc1
                    "index": 0
                  }
                ],
                "model": "text-embedding-ada-002",
                "usage": {
                  "prompt_tokens": 8,
                  "total_tokens": 8
                }
            }));
    });

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .json_body(json!({
                "input": ["What is a linglingdong?"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "object": "list",
                "data": [
                  {
                    "object": "embedding",
                    "embedding": create_embedding_vector(2), // Match doc2
                    "index": 0
                  }
                ],
                "model": "text-embedding-ada-002",
                "usage": {
                  "prompt_tokens": 8,
                  "total_tokens": 8
                }
            }));
    });

    server
}

#[tokio::test]
async fn test_mock_server_setup() {
    assert_ok(
        async {
            let server = create_openai_mock_service().await;
            let openai_client = openai::Client::builder()
                .api_key("TEST")
                .base_url(server.base_url())
                .build()
                .context("failed to build mock openai client")?;

            let model = openai_client
                .embedding_model(openai::TEXT_EMBEDDING_ADA_002)
                .context("embedding model should build")?;

            // Test that we can create embeddings with the mock
            let words = vec![Word {
                id: "test1".to_string(),
                definition: "Test definition".to_string(),
            }];

            let embeddings = EmbeddingsBuilder::new(model)
                .documents(words)?
                .build()
                .await?;
            assert_eq!(embeddings.len(), 1);

            println!("✅ Mock server test passed!");

            Ok(())
        }
        .await,
    );
}

fn assert_ok(result: Result<()>) {
    assert!(result.is_ok(), "{result:?}");
}
