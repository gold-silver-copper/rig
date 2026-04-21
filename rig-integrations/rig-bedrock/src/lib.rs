#![cfg_attr(
    not(test),
    deny(
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unreachable,
        clippy::unwrap_used
    )
)]

pub mod client;
pub mod completion;
pub mod embedding;
pub mod image;
pub mod streaming;
pub mod types;
