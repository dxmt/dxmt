//! VectorQL Parser - Converts SQL-like queries to AST

use super::ast::*;
use super::VectorQLError;
use std::collections::HashMap;

/// SQL Parser for VectorQL
pub struct Parser {
    lexer: Lexer,
}

impl Parser {
    pub fn new() -> Self {
        Self {
            lexer: Lexer::new(),
        }
    }
    
    /// Parse a VectorQL query string into AST
    pub fn parse(&self, query: &str) -> Result<Statement, VectorQLError> {
        let tokens = self.lexer.tokenize(query)?;
        let mut parser_state = ParserState::new(tokens);
        self.parse_statement(&mut parser_state)
    }
    
    fn parse_statement(&self, state: &mut ParserState) -> Result<Statement, VectorQLError> {
        match state.peek()? {
            Token::Keyword(Keyword::Select) => {
                Ok(Statement::Select(self.parse_select(state)?))
            }
            Token::Keyword(Keyword::Insert) => {
                Ok(Statement::Insert(self.parse_insert(state)?))
            }
            Token::Keyword(Keyword::Update) => {
                Ok(Statement::Update(self.parse_update(state)?))
            }
            Token::Keyword(Keyword::Delete) => {
                Ok(Statement::Delete(self.parse_delete(state)?))
            }
            Token::Keyword(Keyword::Create) => {
                self.parse_create(state)
            }
            Token::Keyword(Keyword::Explain) => {
                state.consume(Token::Keyword(Keyword::Explain))?;
                Ok(Statement::Explain(Box::new(self.parse_statement(state)?)))
            }
            _ => Err(VectorQLError::ParseError("Expected statement keyword".to_string())),
        }
    }
    
    fn parse_select(&self, state: &mut ParserState) -> Result<SelectStatement, VectorQLError> {
        state.consume(Token::Keyword(Keyword::Select))?;
        
        // Parse projection
        let projection = self.parse_projection(state)?;
        
        // Parse FROM clause
        state.consume(Token::Keyword(Keyword::From))?;
        let from = self.parse_table_ref(state)?;
        
        // Parse WHERE clause
        let where_clause = if state.peek_keyword(Keyword::Where) {
            state.consume(Token::Keyword(Keyword::Where))?;
            Some(self.parse_expression(state)?)
        } else {
            None
        };
        
        // Parse GROUP BY
        let group_by = if state.peek_keyword(Keyword::Group) {
            state.consume(Token::Keyword(Keyword::Group))?;
            state.consume(Token::Keyword(Keyword::By))?;
            self.parse_expression_list(state)?
        } else {
            Vec::new()
        };
        
        // Parse HAVING
        let having = if state.peek_keyword(Keyword::Having) {
            state.consume(Token::Keyword(Keyword::Having))?;
            Some(self.parse_expression(state)?)
        } else {
            None
        };
        
        // Parse ORDER BY
        let order_by = if state.peek_keyword(Keyword::Order) {
            state.consume(Token::Keyword(Keyword::Order))?;
            state.consume(Token::Keyword(Keyword::By))?;
            self.parse_order_by_list(state)?
        } else {
            Vec::new()
        };
        
        // Parse LIMIT
        let limit = if state.peek_keyword(Keyword::Limit) {
            state.consume(Token::Keyword(Keyword::Limit))?;
            Some(state.consume_number()? as usize)
        } else {
            None
        };
        
        // Parse OFFSET
        let offset = if state.peek_keyword(Keyword::Offset) {
            state.consume(Token::Keyword(Keyword::Offset))?;
            Some(state.consume_number()? as usize)
        } else {
            None
        };
        
        Ok(SelectStatement {
            projection,
            from,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
        })
    }
    
    fn parse_projection(&self, state: &mut ParserState) -> Result<Vec<SelectItem>, VectorQLError> {
        let mut items = Vec::new();
        
        loop {
            if state.peek_token(Token::Star) {
                state.consume(Token::Star)?;
                items.push(SelectItem::Wildcard);
            } else {
                let expr = self.parse_expression(state)?;
                let alias = if state.peek_keyword(Keyword::As) {
                    state.consume(Token::Keyword(Keyword::As))?;
                    Some(state.consume_identifier()?)
                } else if state.peek_identifier() {
                    Some(state.consume_identifier()?)
                } else {
                    None
                };
                items.push(SelectItem::Expression { expr, alias });
            }
            
            if !state.peek_token(Token::Comma) {
                break;
            }
            state.consume(Token::Comma)?;
        }
        
        Ok(items)
    }
    
    fn parse_expression(&self, state: &mut ParserState) -> Result<Expression, VectorQLError> {
        self.parse_or(state)
    }
    
    fn parse_or(&self, state: &mut ParserState) -> Result<Expression, VectorQLError> {
        let mut left = self.parse_and(state)?;
        
        while state.peek_keyword(Keyword::Or) {
            state.consume(Token::Keyword(Keyword::Or))?;
            let right = self.parse_and(state)?;
            left = Expression::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Or,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    fn parse_and(&self, state: &mut ParserState) -> Result<Expression, VectorQLError> {
        let mut left = self.parse_comparison(state)?;
        
        while state.peek_keyword(Keyword::And) {
            state.consume(Token::Keyword(Keyword::And))?;
            let right = self.parse_comparison(state)?;
            left = Expression::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    fn parse_comparison(&self, state: &mut ParserState) -> Result<Expression, VectorQLError> {
        let mut expr = self.parse_primary(state)?;
        
        // Check for vector functions
        if state.peek_identifier() {
            let func_name = state.peek_identifier_value();
            match func_name.to_uppercase().as_str() {
                "SIMILARITY" => {
                    state.consume_identifier()?;
                    state.consume(Token::LeftParen)?;
                    let vector_col = Box::new(expr);
                    state.consume(Token::Comma)?;
                    let query_vector = Box::new(self.parse_expression(state)?);
                    state.consume(Token::RightParen)?;
                    
                    expr = Expression::Similarity {
                        vector_col,
                        query_vector,
                    };
                }
                "KNN" => {
                    state.consume_identifier()?;
                    state.consume(Token::LeftParen)?;
                    let vector_col = Box::new(expr);
                    state.consume(Token::Comma)?;
                    let query_vector = Box::new(self.parse_expression(state)?);
                    state.consume(Token::Comma)?;
                    state.consume_identifier()?; // "k"
                    state.consume(Token::Eq)?;
                    let k = state.consume_number()? as usize;
                    state.consume(Token::RightParen)?;
                    
                    expr = Expression::KNN {
                        vector_col,
                        query_vector,
                        k,
                    };
                }
                _ => {}
            }
        }
        
        // Check for comparison operators
        if let Some(op) = self.parse_comparison_op(state) {
            let right = self.parse_primary(state)?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    fn parse_primary(&self, state: &mut ParserState) -> Result<Expression, VectorQLError> {
        match state.peek()? {
            Token::Null => {
                state.consume(Token::Null)?;
                Ok(Expression::Null)
            }
            Token::Number(n) => {
                let num = *n;
                state.advance();
                if num.fract() == 0.0 {
                    Ok(Expression::Int(num as i64))
                } else {
                    Ok(Expression::Float(num))
                }
            }
            Token::String(s) => {
                let string = s.clone();
                state.advance();
                Ok(Expression::String(string))
            }
            Token::Identifier(name) => {
                let ident = name.clone();
                state.advance();
                
                // Check for function call
                if state.peek_token(Token::LeftParen) {
                    state.consume(Token::LeftParen)?;
                    
                    // Special handling for EMBED function
                    if ident.to_uppercase() == "EMBED" {
                        let text = Box::new(self.parse_expression(state)?);
                        let model = if state.peek_token(Token::Comma) {
                            state.consume(Token::Comma)?;
                            Some(state.consume_string()?)
                        } else {
                            None
                        };
                        state.consume(Token::RightParen)?;
                        return Ok(Expression::Embed { text, model });
                    }
                    
                    // Regular function
                    let args = self.parse_expression_list(state)?;
                    state.consume(Token::RightParen)?;
                    Ok(Expression::Function { name: ident, args })
                } else {
                    Ok(Expression::Column(ident))
                }
            }
            Token::LeftBracket => {
                // Parse vector literal
                state.consume(Token::LeftBracket)?;
                let mut values = Vec::new();
                
                while !state.peek_token(Token::RightBracket) {
                    values.push(state.consume_number()? as f32);
                    if !state.peek_token(Token::Comma) {
                        break;
                    }
                    state.consume(Token::Comma)?;
                }
                
                state.consume(Token::RightBracket)?;
                Ok(Expression::Vector(values))
            }
            Token::LeftParen => {
                state.consume(Token::LeftParen)?;
                let expr = self.parse_expression(state)?;
                state.consume(Token::RightParen)?;
                Ok(expr)
            }
            _ => Err(VectorQLError::ParseError("Expected expression".to_string())),
        }
    }
    
    fn parse_comparison_op(&self, state: &mut ParserState) -> Option<BinaryOperator> {
        let op = match state.peek().ok()? {
            Token::Eq => Some(BinaryOperator::Eq),
            Token::Ne => Some(BinaryOperator::Ne),
            Token::Lt => Some(BinaryOperator::Lt),
            Token::Le => Some(BinaryOperator::Le),
            Token::Gt => Some(BinaryOperator::Gt),
            Token::Ge => Some(BinaryOperator::Ge),
            _ => None,
        };
        
        if op.is_some() {
            state.advance();
        }
        
        op
    }
    
    fn parse_expression_list(&self, state: &mut ParserState) -> Result<Vec<Expression>, VectorQLError> {
        let mut exprs = Vec::new();
        
        if !state.is_end() {
            exprs.push(self.parse_expression(state)?);
            
            while state.peek_token(Token::Comma) {
                state.consume(Token::Comma)?;
                exprs.push(self.parse_expression(state)?);
            }
        }
        
        Ok(exprs)
    }
    
    fn parse_order_by_list(&self, state: &mut ParserState) -> Result<Vec<OrderByItem>, VectorQLError> {
        let mut items = Vec::new();
        
        loop {
            let expr = self.parse_expression(state)?;
            let asc = if state.peek_keyword(Keyword::Desc) {
                state.consume(Token::Keyword(Keyword::Desc))?;
                false
            } else {
                if state.peek_keyword(Keyword::Asc) {
                    state.consume(Token::Keyword(Keyword::Asc))?;
                }
                true
            };
            
            items.push(OrderByItem {
                expr,
                asc,
                nulls_first: None,
            });
            
            if !state.peek_token(Token::Comma) {
                break;
            }
            state.consume(Token::Comma)?;
        }
        
        Ok(items)
    }
    
    fn parse_table_ref(&self, state: &mut ParserState) -> Result<TableRef, VectorQLError> {
        let name = state.consume_identifier()?;
        let alias = if state.peek_keyword(Keyword::As) {
            state.consume(Token::Keyword(Keyword::As))?;
            Some(state.consume_identifier()?)
        } else if state.peek_identifier() {
            Some(state.consume_identifier()?)
        } else {
            None
        };
        
        Ok(TableRef {
            name,
            alias,
            join: None,
        })
    }
    
    fn parse_insert(&self, state: &mut ParserState) -> Result<InsertStatement, VectorQLError> {
        state.consume(Token::Keyword(Keyword::Insert))?;
        state.consume(Token::Keyword(Keyword::Into))?;
        
        let table = state.consume_identifier()?;
        
        // Parse column list
        let columns = if state.peek_token(Token::LeftParen) {
            state.consume(Token::LeftParen)?;
            let cols = self.parse_identifier_list(state)?;
            state.consume(Token::RightParen)?;
            cols
        } else {
            Vec::new()
        };
        
        state.consume(Token::Keyword(Keyword::Values))?;
        
        // Parse values
        let mut values = Vec::new();
        loop {
            state.consume(Token::LeftParen)?;
            let row = self.parse_expression_list(state)?;
            state.consume(Token::RightParen)?;
            values.push(row);
            
            if !state.peek_token(Token::Comma) {
                break;
            }
            state.consume(Token::Comma)?;
        }
        
        Ok(InsertStatement {
            table,
            columns,
            values,
            on_conflict: None,
        })
    }
    
    fn parse_update(&self, state: &mut ParserState) -> Result<UpdateStatement, VectorQLError> {
        state.consume(Token::Keyword(Keyword::Update))?;
        let table = state.consume_identifier()?;
        
        state.consume(Token::Keyword(Keyword::Set))?;
        let assignments = self.parse_assignments(state)?;
        
        let where_clause = if state.peek_keyword(Keyword::Where) {
            state.consume(Token::Keyword(Keyword::Where))?;
            Some(self.parse_expression(state)?)
        } else {
            None
        };
        
        Ok(UpdateStatement {
            table,
            assignments,
            where_clause,
        })
    }
    
    fn parse_delete(&self, state: &mut ParserState) -> Result<DeleteStatement, VectorQLError> {
        state.consume(Token::Keyword(Keyword::Delete))?;
        state.consume(Token::Keyword(Keyword::From))?;
        let table = state.consume_identifier()?;
        
        let where_clause = if state.peek_keyword(Keyword::Where) {
            state.consume(Token::Keyword(Keyword::Where))?;
            Some(self.parse_expression(state)?)
        } else {
            None
        };
        
        Ok(DeleteStatement {
            table,
            where_clause,
        })
    }
    
    fn parse_create(&self, state: &mut ParserState) -> Result<Statement, VectorQLError> {
        state.consume(Token::Keyword(Keyword::Create))?;
        
        if state.peek_keyword(Keyword::Index) {
            Ok(Statement::CreateIndex(self.parse_create_index(state)?))
        } else if state.peek_keyword(Keyword::Materialized) {
            state.consume(Token::Keyword(Keyword::Materialized))?;
            state.consume(Token::Keyword(Keyword::View))?;
            Ok(Statement::CreateView(self.parse_create_view(state, true)?))
        } else if state.peek_keyword(Keyword::View) {
            state.consume(Token::Keyword(Keyword::View))?;
            Ok(Statement::CreateView(self.parse_create_view(state, false)?))
        } else {
            Err(VectorQLError::ParseError("Expected INDEX or VIEW after CREATE".to_string()))
        }
    }
    
    fn parse_create_index(&self, state: &mut ParserState) -> Result<CreateIndexStatement, VectorQLError> {
        state.consume(Token::Keyword(Keyword::Index))?;
        let name = state.consume_identifier()?;
        
        state.consume(Token::Keyword(Keyword::On))?;
        let table = state.consume_identifier()?;
        
        state.consume(Token::Keyword(Keyword::Using))?;
        let index_type = self.parse_index_type(state)?;
        
        state.consume(Token::LeftParen)?;
        let column = state.consume_identifier()?;
        state.consume(Token::RightParen)?;
        
        let options = if state.peek_keyword(Keyword::With) {
            state.consume(Token::Keyword(Keyword::With))?;
            self.parse_options(state)?
        } else {
            HashMap::new()
        };
        
        Ok(CreateIndexStatement {
            name,
            table,
            column,
            index_type,
            options,
        })
    }
    
    fn parse_create_view(&self, state: &mut ParserState, materialized: bool) -> Result<CreateViewStatement, VectorQLError> {
        let name = state.consume_identifier()?;
        state.consume(Token::Keyword(Keyword::As))?;
        let query = self.parse_select(state)?;
        
        Ok(CreateViewStatement {
            name,
            query,
            materialized,
            refresh_policy: None,
        })
    }
    
    fn parse_index_type(&self, state: &mut ParserState) -> Result<IndexType, VectorQLError> {
        let type_name = state.consume_identifier()?;
        match type_name.to_uppercase().as_str() {
            "HNSW" => Ok(IndexType::HNSW),
            "IVF" => Ok(IndexType::IVF),
            "IVFPQ" => Ok(IndexType::IVFPQ),
            "DISKANN" => Ok(IndexType::DiskANN),
            "SCANN" => Ok(IndexType::ScaNN),
            "FLAT" => Ok(IndexType::Flat),
            _ => Err(VectorQLError::ParseError(format!("Unknown index type: {}", type_name))),
        }
    }
    
    fn parse_assignments(&self, state: &mut ParserState) -> Result<Vec<Assignment>, VectorQLError> {
        let mut assignments = Vec::new();
        
        loop {
            let column = state.consume_identifier()?;
            state.consume(Token::Eq)?;
            let value = self.parse_expression(state)?;
            
            assignments.push(Assignment { column, value });
            
            if !state.peek_token(Token::Comma) {
                break;
            }
            state.consume(Token::Comma)?;
        }
        
        Ok(assignments)
    }
    
    fn parse_identifier_list(&self, state: &mut ParserState) -> Result<Vec<String>, VectorQLError> {
        let mut identifiers = Vec::new();
        
        if !state.is_end() {
            identifiers.push(state.consume_identifier()?);
            
            while state.peek_token(Token::Comma) {
                state.consume(Token::Comma)?;
                identifiers.push(state.consume_identifier()?);
            }
        }
        
        Ok(identifiers)
    }
    
    fn parse_options(&self, state: &mut ParserState) -> Result<HashMap<String, serde_json::Value>, VectorQLError> {
        let mut options = HashMap::new();
        
        state.consume(Token::LeftParen)?;
        
        while !state.peek_token(Token::RightParen) {
            let key = state.consume_identifier()?;
            state.consume(Token::Eq)?;
            
            let value = if let Ok(num) = state.consume_number() {
                serde_json::Value::Number(serde_json::Number::from_f64(num).unwrap())
            } else if let Ok(s) = state.consume_string() {
                serde_json::Value::String(s)
            } else {
                return Err(VectorQLError::ParseError("Expected value in options".to_string()));
            };
            
            options.insert(key, value);
            
            if !state.peek_token(Token::Comma) {
                break;
            }
            state.consume(Token::Comma)?;
        }
        
        state.consume(Token::RightParen)?;
        
        Ok(options)
    }
}

/// Lexer for tokenizing VectorQL queries
struct Lexer;

impl Lexer {
    fn new() -> Self {
        Self
    }
    
    fn tokenize(&self, query: &str) -> Result<Vec<Token>, VectorQLError> {
        // Simplified tokenization - in production, use a proper lexer
        let mut tokens = Vec::new();
        let mut chars = query.chars().peekable();
        
        while let Some(&ch) = chars.peek() {
            match ch {
                ' ' | '\t' | '\n' | '\r' => {
                    chars.next();
                }
                '(' => {
                    tokens.push(Token::LeftParen);
                    chars.next();
                }
                ')' => {
                    tokens.push(Token::RightParen);
                    chars.next();
                }
                '[' => {
                    tokens.push(Token::LeftBracket);
                    chars.next();
                }
                ']' => {
                    tokens.push(Token::RightBracket);
                    chars.next();
                }
                ',' => {
                    tokens.push(Token::Comma);
                    chars.next();
                }
                '*' => {
                    tokens.push(Token::Star);
                    chars.next();
                }
                '=' => {
                    chars.next();
                    tokens.push(Token::Eq);
                }
                '>' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Ge);
                    } else {
                        tokens.push(Token::Gt);
                    }
                }
                '<' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Le);
                    } else if chars.peek() == Some(&'>') {
                        chars.next();
                        tokens.push(Token::Ne);
                    } else {
                        tokens.push(Token::Lt);
                    }
                }
                '!' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Ne);
                    }
                }
                '\'' | '"' => {
                    let quote = ch;
                    chars.next();
                    let mut string = String::new();
                    while let Some(&ch) = chars.peek() {
                        if ch == quote {
                            chars.next();
                            break;
                        }
                        string.push(ch);
                        chars.next();
                    }
                    tokens.push(Token::String(string));
                }
                '0'..='9' | '.' | '-' => {
                    let mut number = String::new();
                    while let Some(&ch) = chars.peek() {
                        if ch.is_numeric() || ch == '.' || (ch == '-' && number.is_empty()) {
                            number.push(ch);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if let Ok(n) = number.parse::<f64>() {
                        tokens.push(Token::Number(n));
                    }
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    let mut ident = String::new();
                    while let Some(&ch) = chars.peek() {
                        if ch.is_alphanumeric() || ch == '_' {
                            ident.push(ch);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    
                    // Check for keywords
                    let token = match ident.to_uppercase().as_str() {
                        "SELECT" => Token::Keyword(Keyword::Select),
                        "FROM" => Token::Keyword(Keyword::From),
                        "WHERE" => Token::Keyword(Keyword::Where),
                        "GROUP" => Token::Keyword(Keyword::Group),
                        "BY" => Token::Keyword(Keyword::By),
                        "HAVING" => Token::Keyword(Keyword::Having),
                        "ORDER" => Token::Keyword(Keyword::Order),
                        "LIMIT" => Token::Keyword(Keyword::Limit),
                        "OFFSET" => Token::Keyword(Keyword::Offset),
                        "INSERT" => Token::Keyword(Keyword::Insert),
                        "INTO" => Token::Keyword(Keyword::Into),
                        "VALUES" => Token::Keyword(Keyword::Values),
                        "UPDATE" => Token::Keyword(Keyword::Update),
                        "SET" => Token::Keyword(Keyword::Set),
                        "DELETE" => Token::Keyword(Keyword::Delete),
                        "CREATE" => Token::Keyword(Keyword::Create),
                        "INDEX" => Token::Keyword(Keyword::Index),
                        "VIEW" => Token::Keyword(Keyword::View),
                        "MATERIALIZED" => Token::Keyword(Keyword::Materialized),
                        "ON" => Token::Keyword(Keyword::On),
                        "USING" => Token::Keyword(Keyword::Using),
                        "WITH" => Token::Keyword(Keyword::With),
                        "AS" => Token::Keyword(Keyword::As),
                        "AND" => Token::Keyword(Keyword::And),
                        "OR" => Token::Keyword(Keyword::Or),
                        "NOT" => Token::Keyword(Keyword::Not),
                        "NULL" => Token::Null,
                        "ASC" => Token::Keyword(Keyword::Asc),
                        "DESC" => Token::Keyword(Keyword::Desc),
                        "EXPLAIN" => Token::Keyword(Keyword::Explain),
                        _ => Token::Identifier(ident),
                    };
                    tokens.push(token);
                }
                _ => {
                    chars.next();
                }
            }
        }
        
        Ok(tokens)
    }
}

/// Parser state for tracking position in token stream
struct ParserState {
    tokens: Vec<Token>,
    position: usize,
}

impl ParserState {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }
    
    fn peek(&self) -> Result<&Token, VectorQLError> {
        self.tokens.get(self.position)
            .ok_or_else(|| VectorQLError::ParseError("Unexpected end of query".to_string()))
    }
    
    fn advance(&mut self) {
        self.position += 1;
    }
    
    fn consume(&mut self, expected: Token) -> Result<(), VectorQLError> {
        if self.peek()? == &expected {
            self.advance();
            Ok(())
        } else {
            Err(VectorQLError::ParseError(
                format!("Expected {:?}, got {:?}", expected, self.peek()?)
            ))
        }
    }
    
    fn consume_identifier(&mut self) -> Result<String, VectorQLError> {
        match self.peek()? {
            Token::Identifier(name) => {
                let ident = name.clone();
                self.advance();
                Ok(ident)
            }
            _ => Err(VectorQLError::ParseError("Expected identifier".to_string())),
        }
    }
    
    fn consume_string(&mut self) -> Result<String, VectorQLError> {
        match self.peek()? {
            Token::String(s) => {
                let string = s.clone();
                self.advance();
                Ok(string)
            }
            _ => Err(VectorQLError::ParseError("Expected string".to_string())),
        }
    }
    
    fn consume_number(&mut self) -> Result<f64, VectorQLError> {
        match self.peek()? {
            Token::Number(n) => {
                let num = *n;
                self.advance();
                Ok(num)
            }
            _ => Err(VectorQLError::ParseError("Expected number".to_string())),
        }
    }
    
    fn peek_token(&self, token: Token) -> bool {
        self.peek().map(|t| t == &token).unwrap_or(false)
    }
    
    fn peek_keyword(&self, keyword: Keyword) -> bool {
        self.peek().map(|t| t == &Token::Keyword(keyword)).unwrap_or(false)
    }
    
    fn peek_identifier(&self) -> bool {
        matches!(self.peek(), Ok(Token::Identifier(_)))
    }
    
    fn peek_identifier_value(&self) -> String {
        match self.peek() {
            Ok(Token::Identifier(name)) => name.clone(),
            _ => String::new(),
        }
    }
    
    fn is_end(&self) -> bool {
        self.position >= self.tokens.len()
    }
}

/// Token types
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum Token {
    // Keywords
    Keyword(Keyword),
    
    // Literals
    Null,
    Number(f64),
    String(String),
    Identifier(String),
    
    // Operators
    Eq, Ne, Lt, Le, Gt, Ge,
    Plus, Minus, Star, Slash,
    
    // Punctuation
    LeftParen, RightParen,
    LeftBracket, RightBracket,
    Comma, Semicolon,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum Keyword {
    Select, From, Where, Group, By, Having, Order, Limit, Offset,
    Insert, Into, Values, Update, Set, Delete,
    Create, Index, View, Materialized, On, Using, With,
    As, And, Or, Not, In, Between, Like,
    Asc, Desc, Explain,
}