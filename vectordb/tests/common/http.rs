//! A minimal HTTP/1.1 client over `std::net`, for tests that talk to a spawned server.
//!
//! These tests used to shell out to `curl` and parse what it printed, each file with its own
//! copy of the parsing. On Windows that broke four different ways, none of them about the
//! server: `-o /dev/stdout` names a file that does not exist there, a `-w "\n%{http_code}"`
//! trailer comes back with a `\r` in it, a JSON body passed as an argument hits the
//! command-line length limit, and a Windows path pasted into a hand-built JSON string is an
//! invalid escape. One client in Rust has none of those, and needs no `curl` on the machine.
//!
//! Include it with `#[path = "../common/http.rs"] mod http;`.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// A response: the status, the headers as sent (names lower-cased), and the raw body.
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Reply {
    /// The first header named `name`, case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v.as_str())
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// Send one request to `127.0.0.1:port` and read the whole response. `None` when nothing is
/// listening or the connection fails, as `curl` exiting non-zero used to mean.
///
/// A body is sent as JSON unless `headers` names a content type.
pub fn request(
    port: u16,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
    timeout: Duration,
) -> Option<Reply> {
    let mut stream = TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), timeout).ok()?;
    stream.set_read_timeout(Some(timeout)).ok()?;
    stream.set_write_timeout(Some(timeout)).ok()?;

    let mut head =
        format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    if let Some(body) = body {
        if !headers
            .iter()
            .any(|(n, _)| n.eq_ignore_ascii_case("content-type"))
        {
            head.push_str("Content-Type: application/json\r\n");
        }
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).ok()?;
    if let Some(body) = body {
        stream.write_all(body).ok()?;
    }

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).ok()?;
    parse(&raw)
}

fn parse(raw: &[u8]) -> Option<Reply> {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&raw[..split]).ok()?;
    let rest = &raw[split + 4..];
    let mut lines = head.split("\r\n");
    let status: u16 = lines.next()?.split_whitespace().nth(1)?.parse().ok()?;
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_ascii_lowercase(), value.trim().to_string()))
        })
        .collect();
    let chunked = headers
        .iter()
        .any(|(n, v)| n == "transfer-encoding" && v.eq_ignore_ascii_case("chunked"));
    let body = if chunked {
        dechunk(rest)?
    } else if let Some(len) = headers
        .iter()
        .find(|(n, _)| n == "content-length")
        .and_then(|(_, v)| v.parse::<usize>().ok())
    {
        rest.get(..len)?.to_vec()
    } else {
        rest.to_vec()
    };
    Some(Reply {
        status,
        headers,
        body,
    })
}

fn dechunk(mut rest: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let line_end = rest.windows(2).position(|w| w == b"\r\n")?;
        let size_text = std::str::from_utf8(&rest[..line_end]).ok()?;
        let size = usize::from_str_radix(size_text.split(';').next()?.trim(), 16).ok()?;
        rest = &rest[line_end + 2..];
        if size == 0 {
            return Some(out);
        }
        out.extend_from_slice(rest.get(..size)?);
        rest = rest.get(size + 2..)?;
    }
}

/// `request` for the common case: a JSON body as text, the body back as text.
pub fn call(
    port: u16,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> Option<(u16, String)> {
    request(
        port,
        method,
        path,
        headers,
        body.map(str::as_bytes),
        Duration::from_secs(60),
    )
    .map(|r| (r.status, r.text()))
}
