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

pub mod client;
pub mod completion;
pub(crate) mod types;

pub use client::{Client, ClientBuilder};
