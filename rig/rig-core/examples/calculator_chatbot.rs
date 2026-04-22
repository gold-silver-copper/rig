use anyhow::Result;
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::prelude::*;
use rig::providers::openai;
use rig::{
    completion::ToolDefinition,
    embeddings::EmbeddingsBuilder,
    providers::openai::Client,
    tool::{Tool, ToolEmbedding, ToolSet},
    vector_store::in_memory_store::InMemoryVectorStore,
};

use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Debug, thiserror::Error)]
#[error("Init error")]
struct InitError;

fn arithmetic_tool_definition(
    name: &str,
    description: &str,
    x_description: &str,
    y_description: &str,
) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": x_description
                },
                "y": {
                    "type": "number",
                    "description": y_description
                }
            },
            "required": ["x", "y"]
        }),
    }
}

#[derive(Deserialize, Serialize)]
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        arithmetic_tool_definition(
            "add",
            "Add x and y together",
            "The first number to add",
            "The second number to add",
        )
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x + args.y;
        Ok(result)
    }
}

impl ToolEmbedding for Add {
    type InitError = InitError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Add)
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec!["Add x and y together".into()]
    }

    fn context(&self) -> Self::Context {}
}

#[derive(Deserialize, Serialize)]
struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        arithmetic_tool_definition(
            "subtract",
            "Subtract y from x (i.e.: x - y)",
            "The number to subtract from",
            "The number to subtract",
        )
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x - args.y;
        Ok(result)
    }
}

impl ToolEmbedding for Subtract {
    type InitError = InitError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Subtract)
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec!["Subtract y from x (i.e.: x - y)".into()]
    }

    fn context(&self) -> Self::Context {}
}

struct Multiply;

impl Tool for Multiply {
    const NAME: &'static str = "multiply";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        arithmetic_tool_definition(
            "multiply",
            "Compute the product of x and y (i.e.: x * y)",
            "The first factor in the product",
            "The second factor in the product",
        )
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x * args.y;
        Ok(result)
    }
}

impl ToolEmbedding for Multiply {
    type InitError = InitError;
    type Context = ();
    type State = ();
    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Multiply)
    }
    fn embedding_docs(&self) -> Vec<String> {
        vec!["Compute the product of x and y (i.e.: x * y)".into()]
    }
    fn context(&self) -> Self::Context {}
}

struct Divide;

impl Tool for Divide {
    const NAME: &'static str = "divide";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;
    async fn definition(&self, _prompt: String) -> ToolDefinition {
        arithmetic_tool_definition(
            "divide",
            "Compute the Quotient of x and y (i.e.: x / y). Useful for ratios.",
            "The Dividend of the division. The number being divided",
            "The Divisor of the division. The number by which the dividend is being divided",
        )
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x / args.y;
        Ok(result)
    }
}

impl ToolEmbedding for Divide {
    type InitError = InitError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Divide)
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec!["Compute the Quotient of x and y (i.e.: x / y). Useful for ratios.".into()]
    }

    fn context(&self) -> Self::Context {}
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = Client::from_env()?;

    // Create dynamic tools embeddings
    let toolset = ToolSet::builder()
        .dynamic_tool(Add)
        .dynamic_tool(Subtract)
        .dynamic_tool(Multiply)
        .dynamic_tool(Divide)
        .build();
    let embedding_model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002)?;
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(toolset.schemas()?)?
        .build()
        .await?;

    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone());
    let index = vector_store.index(embedding_model);

    // Create RAG agent with a single context prompt and a dynamic tool source
    let calculator_rag = openai_client
        .agent(openai::GPT_4)
        .preamble(
            "You are an assistant here to help the user select which tool is most appropriate to perform arithmetic operations.
            Follow these instructions closely.
            1. Consider the user's request carefully and identify the core elements of the request.
            2. Select which tool among those made available to you is appropriate given the context.
            3. This is very important: never perform the operation yourself and never give me the direct result.
            Always respond with the name of the tool that should be used and the appropriate inputs
            in the following format:
            Tool: <tool name>
            Inputs: <list of inputs>
            "
        )
        // Add a dynamic tool source with a sample rate of 1 (i.e.: only
        // 1 additional tool will be added to prompts)
        .dynamic_tools(4, index, toolset)
        .build();

    // Create a CLI chatbot from the agent
    let chatbot = ChatBotBuilder::new().agent(calculator_rag).build();

    chatbot.run().await?;

    Ok(())
}
