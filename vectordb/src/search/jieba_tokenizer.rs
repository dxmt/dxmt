//! Custom Jieba Tokenizer for Tantivy 0.25
//!
//! This module provides a simple wrapper around jieba-rs to integrate
//! Chinese word segmentation with Tantivy's full-text search engine.

use jieba_rs::Jieba;
use std::sync::Arc;
use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

/// Jieba tokenizer for Chinese text
#[derive(Clone)]
pub struct JiebaTokenizer {
    jieba: Arc<Jieba>,
}

impl JiebaTokenizer {
    /// Create a new Jieba tokenizer with default dictionary
    pub fn new() -> Self {
        Self {
            jieba: Arc::new(Jieba::new()),
        }
    }

    /// Create a tokenizer with a custom Jieba instance
    pub fn with_jieba(jieba: Jieba) -> Self {
        Self {
            jieba: Arc::new(jieba),
        }
    }
}

impl Default for JiebaTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for JiebaTokenizer {
    type TokenStream<'a> = JiebaTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        JiebaTokenStream {
            jieba: self.jieba.clone(),
            text,
            tokens: Vec::new(),
            index: 0,
            token: Token {
                offset_from: 0,
                offset_to: 0,
                position: 0,
                text: String::new(),
                position_length: 1,
            },
        }
    }
}

/// Token stream for Jieba tokenizer
pub struct JiebaTokenStream<'a> {
    jieba: Arc<Jieba>,
    text: &'a str,
    tokens: Vec<(String, usize, usize)>,
    index: usize,
    token: Token,
}

impl<'a> JiebaTokenStream<'a> {
    fn ensure_tokens(&mut self) {
        if self.tokens.is_empty() && !self.text.is_empty() {
            // Perform word segmentation
            let words = self.jieba.cut(self.text, false);

            // Calculate byte offsets
            let mut byte_offset = 0;
            for word in words {
                let word_bytes = word.len();
                if !word.trim().is_empty() {
                    self.tokens.push((
                        word.to_string(),
                        byte_offset,
                        byte_offset + word_bytes,
                    ));
                }
                byte_offset += word_bytes;
            }
        }
    }
}

impl<'a> TokenStream for JiebaTokenStream<'a> {
    fn advance(&mut self) -> bool {
        self.ensure_tokens();

        if self.index < self.tokens.len() {
            self.index += 1;

            // Update the token with current data
            if self.index > 0 && self.index <= self.tokens.len() {
                let (text, start, end) = &self.tokens[self.index - 1];
                self.token.text = text.clone();
                self.token.offset_from = *start;
                self.token.offset_to = *end;
                self.token.position = self.index - 1;
                self.token.position_length = 1;
            }

            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jieba_tokenizer() {
        let mut tokenizer = JiebaTokenizer::new();
        let mut stream = tokenizer.token_stream("中文分词测试");

        let mut tokens = Vec::new();
        while stream.advance() {
            tokens.push(stream.token().text.clone());
        }

        assert!(!tokens.is_empty());
        println!("Tokens: {:?}", tokens);
    }

    #[test]
    fn test_mixed_text() {
        let mut tokenizer = JiebaTokenizer::new();
        let mut stream = tokenizer.token_stream("这是machine learning测试");

        let mut tokens = Vec::new();
        while stream.advance() {
            tokens.push(stream.token().text.clone());
        }

        assert!(!tokens.is_empty());
        println!("Mixed tokens: {:?}", tokens);
    }
}
