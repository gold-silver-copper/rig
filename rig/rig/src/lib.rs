#![cfg_attr(docsrs, feature(doc_cfg))]
//! Umbrella crate for Rig.
//!
//! This crate re-exports `rig-core` and exposes optional integration crates behind
//! feature flags.

pub use rig_core::*;

/// Optional Rig integrations.
pub mod integrations {
    #[cfg(feature = "bedrock")]
    #[cfg_attr(docsrs, doc(cfg(feature = "bedrock")))]
    pub use rig_bedrock as bedrock;

    #[cfg(feature = "fastembed")]
    #[cfg_attr(docsrs, doc(cfg(feature = "fastembed")))]
    pub use rig_fastembed as fastembed;

    #[cfg(feature = "gemini-grpc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gemini-grpc")))]
    pub use rig_gemini_grpc as gemini_grpc;

    #[cfg(feature = "helixdb")]
    #[cfg_attr(docsrs, doc(cfg(feature = "helixdb")))]
    pub use rig_helixdb as helixdb;

    #[cfg(feature = "lancedb")]
    #[cfg_attr(docsrs, doc(cfg(feature = "lancedb")))]
    pub use rig_lancedb as lancedb;

    #[cfg(feature = "milvus")]
    #[cfg_attr(docsrs, doc(cfg(feature = "milvus")))]
    pub use rig_milvus as milvus;

    #[cfg(feature = "mongodb")]
    #[cfg_attr(docsrs, doc(cfg(feature = "mongodb")))]
    pub use rig_mongodb as mongodb;

    #[cfg(feature = "neo4j")]
    #[cfg_attr(docsrs, doc(cfg(feature = "neo4j")))]
    pub use rig_neo4j as neo4j;

    #[cfg(feature = "postgres")]
    #[cfg_attr(docsrs, doc(cfg(feature = "postgres")))]
    pub use rig_postgres as postgres;

    #[cfg(feature = "qdrant")]
    #[cfg_attr(docsrs, doc(cfg(feature = "qdrant")))]
    pub use rig_qdrant as qdrant;

    #[cfg(feature = "s3vectors")]
    #[cfg_attr(docsrs, doc(cfg(feature = "s3vectors")))]
    pub use rig_s3vectors as s3vectors;

    #[cfg(feature = "scylladb")]
    #[cfg_attr(docsrs, doc(cfg(feature = "scylladb")))]
    pub use rig_scylladb as scylladb;

    #[cfg(feature = "sqlite")]
    #[cfg_attr(docsrs, doc(cfg(feature = "sqlite")))]
    pub use rig_sqlite as sqlite;

    #[cfg(feature = "surrealdb")]
    #[cfg_attr(docsrs, doc(cfg(feature = "surrealdb")))]
    pub use rig_surrealdb as surrealdb;

    #[cfg(feature = "vectorize")]
    #[cfg_attr(docsrs, doc(cfg(feature = "vectorize")))]
    pub use rig_vectorize as vectorize;

    #[cfg(feature = "vertexai")]
    #[cfg_attr(docsrs, doc(cfg(feature = "vertexai")))]
    pub use rig_vertexai as vertexai;
}
