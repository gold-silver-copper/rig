use anyhow::Result;
use rig::{
    Embed,
    embeddings::{self, TextEmbedder, embed::EmbedError},
};
use serde::Serialize;

#[test]
fn test_custom_embed() -> Result<()> {
    #[derive(Embed)]
    struct WordDefinition {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        word: String,
        #[embed(embed_with = "custom_embedding_function")]
        definition: Definition,
    }

    #[derive(Serialize, Clone)]
    struct Definition {
        word: String,
        link: String,
        speech: String,
    }

    fn custom_embedding_function(
        embedder: &mut TextEmbedder,
        definition: Definition,
    ) -> Result<(), EmbedError> {
        embedder.embed(serde_json::to_string(&definition).map_err(EmbedError::new)?);

        Ok(())
    }

    let definition = WordDefinition {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: Definition {
            speech: "noun".to_string(),
            word: "a building in which people live; residence for human beings.".to_string(),
            link: "https://www.dictionary.com/browse/house".to_string(),
        },
    };

    assert_eq!(
        embeddings::to_texts(definition)?,
            vec!["{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()]
        );
    Ok(())
}

#[test]
fn test_custom_and_basic_embed() -> Result<()> {
    #[derive(Embed)]
    struct WordDefinition {
        #[allow(dead_code)]
        id: String,
        #[embed]
        word: String,
        #[embed(embed_with = "custom_embedding_function")]
        definition: Definition,
    }

    #[derive(Serialize, Clone)]
    struct Definition {
        word: String,
        link: String,
        speech: String,
    }

    fn custom_embedding_function(
        embedder: &mut TextEmbedder,
        definition: Definition,
    ) -> Result<(), EmbedError> {
        embedder.embed(serde_json::to_string(&definition).map_err(EmbedError::new)?);

        Ok(())
    }

    let definition = WordDefinition {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: Definition {
            speech: "noun".to_string(),
            word: "a building in which people live; residence for human beings.".to_string(),
            link: "https://www.dictionary.com/browse/house".to_string(),
        },
    };

    let texts = embeddings::to_texts(definition)?;

    assert_eq!(
        texts,
        vec![
            "house".to_string(),
            "{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()
        ]
    );
    Ok(())
}

#[test]
fn test_single_embed() -> Result<()> {
    #[derive(Embed)]
    struct WordDefinition {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        word: String,
        #[embed]
        definition: String,
    }

    let definition = "a building in which people live; residence for human beings.".to_string();

    let word_definition = WordDefinition {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: definition.clone(),
    };

    assert_eq!(embeddings::to_texts(word_definition)?, vec![definition]);
    Ok(())
}

#[test]
fn test_embed_vec_non_string() -> Result<()> {
    #[derive(Embed)]
    struct Company {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        company: String,
        #[embed]
        employee_ages: Vec<i32>,
    }

    let company = Company {
        id: "doc1".to_string(),
        company: "Google".to_string(),
        employee_ages: vec![25, 30, 35, 40],
    };

    assert_eq!(
        embeddings::to_texts(company)?,
        vec![
            "25".to_string(),
            "30".to_string(),
            "35".to_string(),
            "40".to_string()
        ]
    );
    Ok(())
}

#[test]
fn test_embed_vec_string() -> Result<()> {
    #[derive(Embed)]
    struct Company {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        company: String,
        #[embed]
        employee_names: Vec<String>,
    }

    let company = Company {
        id: "doc1".to_string(),
        company: "Google".to_string(),
        employee_names: vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
            "David".to_string(),
        ],
    };

    assert_eq!(
        embeddings::to_texts(company)?,
        vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
            "David".to_string()
        ]
    );
    Ok(())
}

#[test]
fn test_multiple_embed_tags() -> Result<()> {
    #[derive(Embed)]
    struct Company {
        #[allow(dead_code)]
        id: String,
        #[embed]
        company: String,
        #[embed]
        employee_ages: Vec<i32>,
    }

    let company = Company {
        id: "doc1".to_string(),
        company: "Google".to_string(),
        employee_ages: vec![25, 30, 35, 40],
    };

    assert_eq!(
        embeddings::to_texts(company)?,
        vec![
            "Google".to_string(),
            "25".to_string(),
            "30".to_string(),
            "35".to_string(),
            "40".to_string()
        ]
    );
    Ok(())
}
