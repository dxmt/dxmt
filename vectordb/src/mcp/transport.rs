//! MCP transport layer for different connection types

use async_trait::async_trait;
use futures::{Sink, SinkExt, Stream, StreamExt};
use tokio::net::{TcpStream, UnixStream};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Transport trait for MCP communication
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a message
    async fn send(&mut self, message: &str) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Receive a message
    async fn receive(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>>;
    
    /// Check if transport is connected
    fn is_connected(&self) -> bool;
    
    /// Close the transport
    async fn close(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}

type WsSink = Box<dyn Sink<axum::extract::ws::Message, Error = axum::Error> + Send + Sync + Unpin>;
type WsStream =
    Box<dyn Stream<Item = Result<axum::extract::ws::Message, axum::Error>> + Send + Sync + Unpin>;

/// WebSocket transport implementation
pub struct WebSocketTransport {
    sender: Option<Arc<Mutex<WsSink>>>,
    receiver: Option<Arc<Mutex<WsStream>>>,
    connected: bool,
}

impl WebSocketTransport {
    pub fn new(
        sender: Box<dyn Sink<axum::extract::ws::Message, Error = axum::Error> + Send + Sync + Unpin>,
        receiver: Box<dyn Stream<Item = Result<axum::extract::ws::Message, axum::Error>> + Send + Sync + Unpin>,
    ) -> Self {
        Self {
            sender: Some(Arc::new(Mutex::new(sender))),
            receiver: Some(Arc::new(Mutex::new(receiver))),
            connected: true,
        }
    }
}

#[async_trait]
impl McpTransport for WebSocketTransport {
    async fn send(&mut self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(sender_arc) = &self.sender {
            let mut sender = sender_arc.lock().await;
            sender.send(axum::extract::ws::Message::Text(message.to_string())).await?;
            Ok(())
        } else {
            Err("WebSocket sender not available".into())
        }
    }
    
    async fn receive(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        if let Some(receiver_arc) = &self.receiver {
            let mut receiver = receiver_arc.lock().await;
            if let Some(result) = receiver.next().await {
                match result {
                    Ok(msg) => {
                        if let Ok(text) = msg.to_text() {
                            Ok(Some(text.to_string()))
                        } else {
                            Ok(None)
                        }
                    }
                    Err(e) => Err(e.into()),
                }
            } else {
                self.connected = false;
                Ok(None)
            }
        } else {
            Err("WebSocket receiver not available".into())
        }
    }
    
    fn is_connected(&self) -> bool {
        self.connected
    }
    
    async fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.connected = false;
        self.sender = None;
        self.receiver = None;
        Ok(())
    }
}

/// Standard I/O transport for CLI integration
pub struct StdioTransport {
    connected: bool,
}

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl StdioTransport {
    pub fn new() -> Self {
        Self { connected: true }
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send(&mut self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", message);
        Ok(())
    }
    
    async fn receive(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        use tokio::io::{AsyncBufReadExt, BufReader};
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();
        
        match reader.read_line(&mut line).await {
            Ok(0) => {
                self.connected = false;
                Ok(None)
            }
            Ok(_) => Ok(Some(line.trim().to_string())),
            Err(e) => Err(e.into()),
        }
    }
    
    fn is_connected(&self) -> bool {
        self.connected
    }
    
    async fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.connected = false;
        Ok(())
    }
}

/// TCP transport for network connections
pub struct TcpTransport {
    stream: Option<TcpStream>,
    buffer: Vec<u8>,
}

impl TcpTransport {
    pub async fn connect(addr: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = TcpStream::connect(addr).await?;
        Ok(Self {
            stream: Some(stream),
            buffer: Vec::with_capacity(4096),
        })
    }
    
    pub fn from_stream(stream: TcpStream) -> Self {
        Self {
            stream: Some(stream),
            buffer: Vec::with_capacity(4096),
        }
    }
}

#[async_trait]
impl McpTransport for TcpTransport {
    async fn send(&mut self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        use tokio::io::AsyncWriteExt;
        
        if let Some(stream) = &mut self.stream {
            let frame = format!("{}\n", message);
            stream.write_all(frame.as_bytes()).await?;
            stream.flush().await?;
            Ok(())
        } else {
            Err("TCP stream not connected".into())
        }
    }
    
    async fn receive(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        use tokio::io::AsyncReadExt;
        
        if let Some(stream) = &mut self.stream {
            self.buffer.clear();
            let mut byte = [0u8; 1];
            
            loop {
                match stream.read(&mut byte).await {
                    Ok(0) => return Ok(None), // Connection closed
                    Ok(_) => {
                        if byte[0] == b'\n' {
                            break;
                        }
                        self.buffer.push(byte[0]);
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            
            String::from_utf8(self.buffer.clone())
                .map(Some)
                .map_err(|e| e.into())
        } else {
            Err("TCP stream not connected".into())
        }
    }
    
    fn is_connected(&self) -> bool {
        self.stream.is_some()
    }
    
    async fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut stream) = self.stream.take() {
            use tokio::io::AsyncWriteExt;
            stream.shutdown().await?;
        }
        Ok(())
    }
}

/// Unix domain socket transport
#[cfg(unix)]
pub struct UnixSocketTransport {
    stream: Option<UnixStream>,
    buffer: Vec<u8>,
}

#[cfg(unix)]
impl UnixSocketTransport {
    pub async fn connect(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = UnixStream::connect(path).await?;
        Ok(Self {
            stream: Some(stream),
            buffer: Vec::with_capacity(4096),
        })
    }
}

#[cfg(unix)]
#[async_trait]
impl McpTransport for UnixSocketTransport {
    async fn send(&mut self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        use tokio::io::AsyncWriteExt;
        
        if let Some(stream) = &mut self.stream {
            let frame = format!("{}\n", message);
            stream.write_all(frame.as_bytes()).await?;
            stream.flush().await?;
            Ok(())
        } else {
            Err("Unix socket not connected".into())
        }
    }
    
    async fn receive(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        use tokio::io::AsyncReadExt;
        
        if let Some(stream) = &mut self.stream {
            self.buffer.clear();
            let mut byte = [0u8; 1];
            
            loop {
                match stream.read(&mut byte).await {
                    Ok(0) => return Ok(None),
                    Ok(_) => {
                        if byte[0] == b'\n' {
                            break;
                        }
                        self.buffer.push(byte[0]);
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            
            String::from_utf8(self.buffer.clone())
                .map(Some)
                .map_err(|e| e.into())
        } else {
            Err("Unix socket not connected".into())
        }
    }
    
    fn is_connected(&self) -> bool {
        self.stream.is_some()
    }
    
    async fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut stream) = self.stream.take() {
            use tokio::io::AsyncWriteExt;
            stream.shutdown().await?;
        }
        Ok(())
    }
}

/// Transport factory for creating appropriate transport
pub struct TransportFactory;

impl TransportFactory {
    /// Create transport from connection string
    pub async fn create(connection_string: &str) -> Result<Box<dyn McpTransport>, Box<dyn std::error::Error>> {
        if connection_string.starts_with("ws://") || connection_string.starts_with("wss://") {
            Err("WebSocket transport requires WebSocket connection".into())
        } else if connection_string.starts_with("tcp://") {
            let addr = connection_string.strip_prefix("tcp://").unwrap();
            Ok(Box::new(TcpTransport::connect(addr).await?))
        } else if connection_string.starts_with("unix://") {
            #[cfg(unix)]
            {
                let path = connection_string.strip_prefix("unix://").unwrap();
                Ok(Box::new(UnixSocketTransport::connect(path).await?))
            }
            #[cfg(not(unix))]
            {
                Err("Unix socket not supported on this platform".into())
            }
        } else if connection_string == "stdio" {
            Ok(Box::new(StdioTransport::new()))
        } else {
            Err(format!("Unknown transport type: {}", connection_string).into())
        }
    }
}

/// Message framing for transport protocols
pub struct MessageFramer {
    max_message_size: usize,
}

impl MessageFramer {
    pub fn new(max_message_size: usize) -> Self {
        Self { max_message_size }
    }
    
    /// Frame a message with length prefix
    pub fn frame(&self, message: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let bytes = message.as_bytes();
        if bytes.len() > self.max_message_size {
            return Err("Message too large".into());
        }
        
        let mut framed = Vec::with_capacity(bytes.len() + 8);
        framed.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        framed.extend_from_slice(bytes);
        Ok(framed)
    }
    
    /// Unframe a message
    pub fn unframe(&self, data: &[u8]) -> Result<(String, usize), Box<dyn std::error::Error>> {
        if data.len() < 8 {
            return Err("Incomplete frame header".into());
        }
        
        let mut size_bytes = [0u8; 8];
        size_bytes.copy_from_slice(&data[0..8]);
        let message_size = u64::from_be_bytes(size_bytes) as usize;
        
        if message_size > self.max_message_size {
            return Err("Message size exceeds maximum".into());
        }
        
        let total_size = 8 + message_size;
        if data.len() < total_size {
            return Err("Incomplete message data".into());
        }
        
        let message = String::from_utf8(data[8..total_size].to_vec())?;
        Ok((message, total_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_message_framing() {
        let framer = MessageFramer::new(1024);
        
        let message = "Hello, MCP!";
        let framed = framer.frame(message).unwrap();
        
        let (unframed, size) = framer.unframe(&framed).unwrap();
        assert_eq!(unframed, message);
        assert_eq!(size, framed.len());
    }
    
    #[test]
    fn test_message_too_large() {
        let framer = MessageFramer::new(10);
        let message = "This message is too large";
        
        assert!(framer.frame(message).is_err());
    }
}