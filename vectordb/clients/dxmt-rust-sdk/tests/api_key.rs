//! The key reaches the server as `api-key`, on every request, and never reaches a log.
//!
//! A one-shot server on a local socket stands in for the real one, so this needs neither a
//! running DXMT nor a mock-server dependency: what is under test is the header on the wire.

use dxmt_client::Client;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;

/// Accept one request, answer it with an empty table list, and return its header lines.
fn one_request() -> (String, std::thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut headers = Vec::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line.trim().is_empty() {
                break;
            }
            headers.push(line.trim().to_ascii_lowercase());
        }
        let body = r#"{"success":true,"data":{"tables":[]},"error":null}"#;
        let mut stream = stream;
        write!(
            stream,
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        headers
    });
    (url, handle)
}

#[tokio::test]
async fn the_key_is_sent_as_the_api_key_header() {
    let (url, server) = one_request();
    let db = Client::new(url).unwrap().with_api_key("s3cret");
    db.list_tables().await.unwrap();
    let headers = server.join().unwrap();
    assert!(headers.contains(&"api-key: s3cret".to_string()), "{headers:?}");
}

#[tokio::test]
async fn without_a_key_no_header_is_sent() {
    let (url, server) = one_request();
    Client::new(url).unwrap().list_tables().await.unwrap();
    let headers = server.join().unwrap();
    assert!(!headers.iter().any(|h| h.starts_with("api-key:")), "{headers:?}");
}

#[test]
fn debug_output_does_not_contain_the_key() {
    let db = Client::new("http://127.0.0.1:9").unwrap().with_api_key("s3cret");
    let printed = format!("{db:?}");
    assert!(!printed.contains("s3cret"), "{printed}");
    assert!(printed.contains("<redacted>"), "{printed}");
}
