//! The JSON body extractor every endpoint uses.
//!
//! Two things it does that `axum::Json` does not.
//!
//! **It answers in the API's own shape.** `axum::Json`'s rejection is a 422 with a
//! `text/plain` body, so a client that reads `error.code` off every response gets a parse
//! failure instead of an error -- for the most ordinary mistake there is, a malformed body.
//! This returns the same `{success, error: {code, message}}` envelope as everything else, with
//! `INVALID_INPUT` and a 400.
//!
//! **It refuses unknown fields**, together with `#[serde(deny_unknown_fields)]` on the request
//! types. serde's default is to ignore them, which made every misspelled or invented field a
//! silent no-op: `mode` on an insert, `top_k` and `alpha` on a hybrid search (both in this
//! crate's own Swagger example at the time), `named_vectors` on a table that then built no
//! named index. Each was accepted, dropped, and left the caller watching a knob that changed
//! nothing. serde's message names the field and lists the ones that exist, which is the whole
//! value of the change, so it is passed through as the message.

use crate::api::response::{ApiError, ApiResponse};
use axum::{
    async_trait,
    extract::{
        rejection::{JsonRejection, QueryRejection},
        FromRequest, FromRequestParts, Query,
    },
    http::{request::Parts, Request},
    Json,
};
use serde::de::DeserializeOwned;

/// `Json<T>`, with unknown fields refused and rejections in the API's error envelope.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApiJson<T>(pub T);

#[async_trait]
impl<S, B, T> FromRequest<S, B> for ApiJson<T>
where
    T: DeserializeOwned,
    B: axum::body::HttpBody + Send + 'static,
    B::Data: Send,
    B::Error: Into<axum::BoxError>,
    S: Send + Sync,
{
    type Rejection = ApiResponse;

    async fn from_request(req: Request<B>, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(ApiJson(value)),
            Err(rejection) => Err(rejection_response(rejection)),
        }
    }
}

/// `Query<T>`, with unknown parameters refused and rejections in the API's error envelope.
///
/// Same reasoning as `ApiJson`, and the same silence before it: `?limitt=1` was accepted and
/// dropped, so a caller who misspelled `limit` got the default page of 50 while believing they
/// had asked for one row. A query string is the easiest place in an HTTP API to make a typo and
/// the least likely place to notice one.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApiQuery<T>(pub T);

#[async_trait]
impl<S, T> FromRequestParts<S> for ApiQuery<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiResponse;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match Query::<T>::from_request_parts(parts, state).await {
            Ok(Query(value)) => Ok(ApiQuery(value)),
            Err(rejection) => Err(query_rejection_response(rejection)),
        }
    }
}

fn query_rejection_response(rejection: QueryRejection) -> ApiResponse {
    ApiResponse::error(ApiError::new("INVALID_INPUT", &rejection.body_text()))
}

fn rejection_response(rejection: JsonRejection) -> ApiResponse {
    // `body_text` is what serde said: "unknown field `mode`, expected one of `records`, ..." or
    // "missing field `vector`", with the line and column. Naming the field is the point.
    ApiResponse::error(ApiError::new("INVALID_INPUT", &rejection.body_text()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, routing::post, Router};
    use serde::Deserialize;
    use tower::ServiceExt;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Body2 {
        wanted: u32,
    }

    async fn handler(ApiJson(body): ApiJson<Body2>) -> String {
        body.wanted.to_string()
    }

    async fn post_body(body: &str) -> (u16, String) {
        let app = Router::new().route("/", post(handler));
        let request = Request::post("/")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let bytes = hyper::body::to_bytes(response.into_body()).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn a_good_body_is_extracted() {
        assert_eq!(post_body(r#"{"wanted": 7}"#).await, (200, "7".to_string()));
    }

    #[tokio::test]
    async fn an_unknown_field_is_a_400_naming_it() {
        let (status, body) = post_body(r#"{"wanted": 7, "wnated": 8}"#).await;
        assert_eq!(status, 400);
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["success"], false);
        assert_eq!(json["error"]["code"], "INVALID_INPUT");
        let message = json["error"]["message"].as_str().unwrap();
        assert!(message.contains("wnated"), "{message}");
        assert!(message.contains("wanted"), "{message}");
    }

    #[tokio::test]
    async fn a_malformed_body_is_an_error_object_not_plain_text() {
        // axum's own rejection here is a 422 with a text/plain body, which is what every client
        // reading `error.code` chokes on.
        let (status, body) = post_body("{not json").await;
        assert_eq!(status, 400);
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["error"]["code"], "INVALID_INPUT");
    }

    #[tokio::test]
    async fn a_missing_field_says_which() {
        let (_, body) = post_body("{}").await;
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("wanted"));
    }
}
