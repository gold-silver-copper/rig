use super::{Client, client::ApiResponse};
use crate::http_client::HttpClientExt;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::models::ImageGenerationResponseFormatMode;
use crate::{http_client, image_generation};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Deserialize;
use serde_json::json;

// ================================================================
// OpenAI Image Generation API
// ================================================================

#[derive(Debug, Deserialize)]
pub struct ImageGenerationData {
    pub b64_json: String,
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationResponse {
    pub created: i32,
    pub data: Vec<ImageGenerationData>,
}

impl TryFrom<ImageGenerationResponse>
    for image_generation::ImageGenerationResponse<ImageGenerationResponse>
{
    type Error = ImageGenerationError;

    fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
        let b64_json = value.data[0].b64_json.clone();

        let bytes = BASE64_STANDARD
            .decode(&b64_json)
            .expect("Failed to decode b64");

        Ok(image_generation::ImageGenerationResponse {
            image: bytes,
            response: value,
        })
    }
}

#[derive(Clone)]
pub struct ImageGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: dall-e-2)
    pub model: String,
}

impl<T> ImageGenerationModel<T> {
    pub(crate) fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

fn request_payload(model: &str, generation_request: &ImageGenerationRequest) -> serde_json::Value {
    let mut request = json!({
        "model": model,
        "prompt": generation_request.prompt,
        "size": format!("{}x{}", generation_request.width, generation_request.height),
    });

    let response_format_mode = crate::models::openai::lookup(model)
        .and_then(|model| model.image_generation)
        .and_then(|metadata| metadata.response_format_mode);

    if !matches!(
        response_format_mode,
        Some(ImageGenerationResponseFormatMode::OmitField)
    ) {
        merge_inplace(
            &mut request,
            json!({
                "response_format": "b64_json"
            }),
        );
    }

    request
}

impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = ImageGenerationResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
    {
        let request = request_payload(self.model.as_str(), &generation_request);
        let body = serde_json::to_vec(&request)?;

        let request = self
            .client
            .post("/images/generations")?
            .body(body)
            .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

        let response = self.client.send(request).await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = http_client::text(response).await?;

            return Err(ImageGenerationError::ProviderError(format!(
                "{}: {}",
                status, text,
            )));
        }

        let text = http_client::text(response).await?;

        match serde_json::from_str::<ApiResponse<ImageGenerationResponse>>(&text)? {
            ApiResponse::Ok(response) => response.try_into(),
            ApiResponse::Err(err) => Err(ImageGenerationError::ProviderError(err.message)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_request(model: &str) -> serde_json::Value {
        request_payload(
            model,
            &ImageGenerationRequest {
                prompt: "draw a lighthouse".into(),
                width: 512,
                height: 512,
                additional_params: None,
            },
        )
    }

    #[test]
    fn request_omits_response_format_for_catalogued_gpt_image_models() {
        let request = test_request(crate::models::openai::GPT_IMAGE_1_5);
        assert_eq!(request.get("response_format"), None);
    }

    #[test]
    fn request_keeps_response_format_for_dalle_models() {
        let request = test_request(crate::models::openai::DALL_E_3);
        assert_eq!(request.get("response_format"), Some(&json!("b64_json")));
    }

    #[test]
    fn request_keeps_response_format_for_unknown_models() {
        let request = test_request("future-image-model");
        assert_eq!(request.get("response_format"), Some(&json!("b64_json")));
    }
}
