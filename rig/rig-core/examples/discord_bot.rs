use rig_core::integrations::discord_bot::DiscordExt;
use rig_core::prelude::*;
use rig_core::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let discord_bot_token = std::env::var("DISCORD_BOT_TOKEN")?;
    // Create OpenAI client
    let client = rig_core::providers::openai::Client::from_env()?;

    // Create agent with a single context prompt
    let mut discord_bot = client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful assistant.")
        .build()
        .into_discord_bot(&discord_bot_token)
        .await?;

    discord_bot.start().await?;

    Ok(())
}
