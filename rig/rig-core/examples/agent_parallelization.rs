use rig::prelude::*;

use rig::providers::openai;
use rig::providers::openai::client::Client;

use rig::{
    pipeline::{self, TryOp, agent_ops},
    try_parallel,
};

use schemars::JsonSchema;

#[derive(serde::Deserialize, JsonSchema, serde::Serialize)]
struct DocumentScore {
    /// The score of the document
    score: f32,
}
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = Client::from_env()?;

    let manipulation_agent = openai_client
        .extractor::<DocumentScore>(openai::GPT_4)
        .preamble(
            "
            Your role is to score a user's statement on how manipulative it sounds between 0 and 1.
        ",
        )
        .build();

    let depression_agent = openai_client
        .extractor::<DocumentScore>(openai::GPT_4)
        .preamble(
            "
            Your role is to score a user's statement on how depressive it sounds between 0 and 1.
        ",
        )
        .build();

    let intelligent_agent = openai_client
        .extractor::<DocumentScore>(openai::GPT_4)
        .preamble(
            "
            Your role is to score a user's statement on how intelligent it sounds between 0 and 1.
        ",
        )
        .build();

    let statement = "I hate swimming. The water always gets in my eyes.";
    let (manip_score, dep_score, int_score) = pipeline::new()
        .chain(try_parallel!(
            agent_ops::extract(manipulation_agent),
            agent_ops::extract(depression_agent),
            agent_ops::extract(intelligent_agent)
        ))
        .try_call(statement)
        .await?;

    let response = format!(
        "
        Original statement: {statement}
        Manipulation sentiment score: {}
        Depression sentiment score: {}
        Intelligence sentiment score: {}
        ",
        manip_score.score, dep_score.score, int_score.score
    );

    println!("Pipeline run: {response:?}");

    Ok(())
}
