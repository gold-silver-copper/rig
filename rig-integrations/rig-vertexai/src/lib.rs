#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unreachable,
        clippy::unwrap_used
    )
)]
pub mod client;
pub mod completion;
pub(crate) mod types;

pub use client::{Client, ClientBuilder, VertexAiClientError};
