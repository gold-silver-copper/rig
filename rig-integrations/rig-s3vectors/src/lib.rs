#[macro_use]
mod document;

use aws_sdk_s3vectors::{
    Client,
    types::{PutInputVector, VectorData},
};
use aws_smithy_types::Document;
use rig::{
    embeddings::EmbeddingModel,
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fmt};
use uuid::Uuid;

#[derive(Debug)]
enum S3VectorStoreError {
    InvalidFloat,
    MissingDistance,
    MissingMetadata,
}

impl fmt::Display for S3VectorStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            S3VectorStoreError::InvalidFloat => {
                write!(f, "S3Vectors returned a non-finite floating-point value")
            }
            S3VectorStoreError::MissingDistance => {
                write!(f, "S3Vectors response did not include a vector distance")
            }
            S3VectorStoreError::MissingMetadata => {
                write!(f, "S3Vectors response did not include vector metadata")
            }
        }
    }
}

impl std::error::Error for S3VectorStoreError {}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRecord {
    document: serde_json::Value,
    embedded_text: String,
}

// NOTE: Cannot be used in dynamic store due to aws_smithy_types::Document not impl'ing Serialize or Deserialize
#[derive(Clone, Debug)]
pub struct S3SearchFilter(aws_smithy_types::Document);

impl SearchFilter for S3SearchFilter {
    type Value = aws_smithy_types::Document;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        let key = key.as_ref().to_owned();
        Self(document!({ key: { "$eq": value } }))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        let key = key.as_ref().to_owned();
        Self(document!({ key: { "$gt": value } }))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        let key = key.as_ref().to_owned();
        Self(document!({ key: { "$lt": value } }))
    }

    fn and(self, rhs: Self) -> Self {
        Self(document!({ "$and": [ self.0, rhs.0 ]}))
    }

    fn or(self, rhs: Self) -> Self {
        Self(document!({ "$or": [ self.0, rhs.0 ]}))
    }
}

impl S3SearchFilter {
    pub fn inner(&self) -> &aws_smithy_types::Document {
        &self.0
    }

    pub fn into_inner(self) -> aws_smithy_types::Document {
        self.0
    }

    pub fn gte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(document!({ key: { "$gte": value } }))
    }

    pub fn lte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(document!({ key: { "$lte": value } }))
    }

    pub fn exists(key: String) -> Self {
        Self(document!({ "$exists": { key: true } }))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(document!({ "$not": self.0 }))
    }
}

pub struct S3VectorsVectorStore<M> {
    embedding_model: M,
    client: Client,
    bucket_name: String,
    index_name: String,
}

impl<M> S3VectorsVectorStore<M>
where
    M: EmbeddingModel,
{
    pub fn new(
        embedding_model: M,
        client: aws_sdk_s3vectors::Client,
        bucket_name: &str,
        index_name: &str,
    ) -> Self {
        Self {
            embedding_model,
            client,
            bucket_name: bucket_name.to_string(),
            index_name: index_name.to_string(),
        }
    }

    pub fn bucket_name(&self) -> &str {
        &self.bucket_name
    }

    pub fn set_bucket_name(&mut self, bucket_name: &str) {
        self.bucket_name = bucket_name.to_string();
    }

    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    pub fn set_index_name(&mut self, index_name: &str) {
        self.index_name = index_name.to_string();
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
}

impl<M> InsertDocuments for S3VectorsVectorStore<M>
where
    M: EmbeddingModel,
{
    async fn insert_documents<Doc: serde::Serialize + rig::Embed + Send>(
        &self,
        documents: Vec<(Doc, rig::OneOrMany<rig::embeddings::Embedding>)>,
    ) -> Result<(), rig::vector_store::VectorStoreError> {
        let docs: Vec<PutInputVector> = documents
            .into_iter()
            .map(|x| {
                let json_value = serde_json::to_value(&x.0).map_err(VectorStoreError::JsonError)?;

                x.1.into_iter()
                    .map(|y| {
                        let document = CreateRecord {
                            document: json_value.clone(),
                            embedded_text: y.document,
                        };
                        let document =
                            serde_json::to_value(&document).map_err(VectorStoreError::JsonError)?;
                        let document = json_value_to_document(&document)?;
                        let vec = y.vec.into_iter().map(|item| item as f32).collect();
                        PutInputVector::builder()
                            .metadata(document)
                            .data(VectorData::Float32(vec))
                            .key(Uuid::new_v4())
                            .build()
                            .map_err(|x| {
                                VectorStoreError::DatastoreError(
                                    format!("Couldn't build vector input: {x}").into(),
                                )
                            })
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<PutInputVector>>, VectorStoreError>>()
            .map_err(|x| {
                VectorStoreError::DatastoreError(
                    format!("Could not build vector store data: {x}").into(),
                )
            })?
            .into_iter()
            .flatten()
            .collect();

        self.client
            .put_vectors()
            .vector_bucket_name(self.bucket_name())
            .set_vectors(Some(docs))
            .set_index_name(Some(self.index_name.clone()))
            .send()
            .await
            .map_err(|x| {
                VectorStoreError::DatastoreError(
                    format!("Error while submitting document insertion request: {x}").into(),
                )
            })?;

        Ok(())
    }
}

fn json_value_to_document(value: &Value) -> Result<Document, VectorStoreError> {
    match value {
        Value::Null => Ok(Document::Null),
        Value::Bool(b) => Ok(Document::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Document::Number(aws_smithy_types::Number::NegInt(i)))
            } else if let Some(u) = n.as_u64() {
                Ok(Document::Number(aws_smithy_types::Number::PosInt(u)))
            } else if let Some(f) = n.as_f64() {
                Ok(Document::Number(aws_smithy_types::Number::Float(f)))
            } else {
                Err(VectorStoreError::DatastoreError(Box::new(
                    serde_json::Error::io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unsupported JSON number for S3Vectors metadata: {n}"),
                    )),
                )))
            }
        }
        Value::String(s) => Ok(Document::String(s.clone())),
        Value::Array(arr) => Ok(Document::Array(
            arr.iter()
                .map(json_value_to_document)
                .collect::<Result<_, _>>()?,
        )),
        Value::Object(obj) => Ok(Document::Object(
            obj.iter()
                .map(|(k, v)| Ok((k.clone(), json_value_to_document(v)?)))
                .collect::<Result<HashMap<_, _>, VectorStoreError>>()?,
        )),
    }
}

fn document_to_json_value(value: &Document) -> Result<Value, VectorStoreError> {
    match value {
        Document::Null => Ok(Value::Null),
        Document::Bool(b) => Ok(Value::Bool(*b)),
        Document::Number(n) => {
            let res = match n {
                aws_smithy_types::Number::Float(f) => {
                    serde_json::Number::from_f64(*f).ok_or_else(|| {
                        VectorStoreError::DatastoreError(Box::new(S3VectorStoreError::InvalidFloat))
                    })?
                }
                aws_smithy_types::Number::NegInt(i) => serde_json::Number::from(*i),
                aws_smithy_types::Number::PosInt(u) => serde_json::Number::from(*u),
            };

            Ok(serde_json::Value::Number(res))
        }
        Document::String(s) => Ok(Value::String(s.clone())),
        Document::Array(arr) => Ok(Value::Array(
            arr.iter()
                .map(document_to_json_value)
                .collect::<Result<_, _>>()?,
        )),
        Document::Object(obj) => {
            let res = obj
                .iter()
                .map(|(k, v)| Ok((k.clone(), document_to_json_value(v)?)))
                .collect::<Result<serde_json::Map<String, serde_json::Value>, VectorStoreError>>(
                )?;

            Ok(serde_json::Value::Object(res))
        }
    }
}

impl<M> VectorStoreIndex for S3VectorsVectorStore<M>
where
    M: EmbeddingModel,
{
    type Filter = S3SearchFilter;

    async fn top_n<T: for<'a> serde::Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<S3SearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        if req.samples() > i32::MAX as u64 {
            return Err(VectorStoreError::DatastoreError(format!("The number of samples to return with the `rig` AWS S3Vectors integration cannot be higher than {}", i32::MAX).into()));
        }

        let embedding = self
            .embedding_model
            .embed_text(req.query())
            .await?
            .vec
            .into_iter()
            .map(|x| x as f32)
            .collect();

        let mut query_builder = self
            .client
            .query_vectors()
            .query_vector(VectorData::Float32(embedding))
            .top_k(req.samples() as i32)
            .return_distance(true)
            .return_metadata(true)
            .vector_bucket_name(self.bucket_name())
            .index_name(self.index_name());

        if let Some(filter) = req.filter() {
            query_builder = query_builder.filter(filter.inner().clone())
        }

        let query = query_builder
            .send()
            .await
            .map_err(|error| VectorStoreError::DatastoreError(Box::new(error)))?;

        let res: Vec<(f64, String, T)> = query
            .vectors
            .into_iter()
            .map(|vector| {
                let distance = vector.distance.ok_or_else(|| {
                    VectorStoreError::DatastoreError(Box::new(S3VectorStoreError::MissingDistance))
                })? as f64;
                if req
                    .threshold()
                    .is_some_and(|threshold| distance < threshold)
                {
                    return Ok(None);
                }

                let metadata = vector.metadata.ok_or_else(|| {
                    VectorStoreError::DatastoreError(Box::new(S3VectorStoreError::MissingMetadata))
                })?;
                let value = document_to_json_value(&metadata)?;
                let metadata =
                    serde_json::from_value(value).map_err(VectorStoreError::JsonError)?;

                Ok(Some((distance, vector.key, metadata)))
            })
            .collect::<Result<Vec<_>, VectorStoreError>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(res)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<S3SearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        if req.samples() > i32::MAX as u64 {
            return Err(VectorStoreError::DatastoreError(format!("The number of samples to return with the `rig` AWS S3Vectors integration cannot be higher than {}", i32::MAX).into()));
        }

        let embedding = self
            .embedding_model
            .embed_text(req.query())
            .await?
            .vec
            .into_iter()
            .map(|x| x as f32)
            .collect();

        let mut query_builder = self
            .client
            .query_vectors()
            .query_vector(VectorData::Float32(embedding))
            .top_k(req.samples() as i32)
            .return_distance(true)
            .vector_bucket_name(self.bucket_name())
            .index_name(self.index_name());

        if let Some(filter) = req.filter() {
            query_builder = query_builder.filter(filter.inner().clone())
        }

        let query = query_builder
            .send()
            .await
            .map_err(|error| VectorStoreError::DatastoreError(Box::new(error)))?;

        let res: Vec<(f64, String)> = query
            .vectors
            .into_iter()
            .map(|vector| {
                let distance = vector.distance.ok_or_else(|| {
                    VectorStoreError::DatastoreError(Box::new(S3VectorStoreError::MissingDistance))
                })? as f64;
                if req
                    .threshold()
                    .is_some_and(|threshold| distance < threshold)
                {
                    return Ok(None);
                }

                Ok(Some((distance, vector.key)))
            })
            .collect::<Result<Vec<_>, VectorStoreError>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(res)
    }
}
