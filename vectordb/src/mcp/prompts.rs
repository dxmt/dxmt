//! MCP prompt templates for AI assistant interactions

use super::*;

/// Prompt template registry
pub struct PromptRegistry {
    prompts: Vec<Prompt>,
}

impl PromptRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            prompts: Vec::new(),
        };
        registry.register_default_prompts();
        registry
    }
    
    /// Register all default prompts
    fn register_default_prompts(&mut self) {
        // RAG prompts
        self.register_rag_prompts();
        
        // Analysis prompts
        self.register_analysis_prompts();
        
        // Data management prompts
        self.register_data_prompts();
        
        // Optimization prompts
        self.register_optimization_prompts();
        
        // Troubleshooting prompts
        self.register_troubleshooting_prompts();
    }
    
    /// Register RAG-related prompts
    fn register_rag_prompts(&mut self) {
        self.prompts.push(Prompt {
            name: "rag_basic".to_string(),
            description: Some("Basic RAG pipeline".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "question".to_string(),
                    description: Some("User question".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "table".to_string(),
                    description: Some("Table to search".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "top_k".to_string(),
                    description: Some("Number of documents to retrieve".to_string()),
                    required: Some(false),
                },
            ]),
        });
        
        self.prompts.push(Prompt {
            name: "rag_with_reranking".to_string(),
            description: Some("RAG with reranking step".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "question".to_string(),
                    description: Some("User question".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "table".to_string(),
                    description: Some("Table to search".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "initial_k".to_string(),
                    description: Some("Initial retrieval count".to_string()),
                    required: Some(false),
                },
                PromptArgument {
                    name: "final_k".to_string(),
                    description: Some("Final document count after reranking".to_string()),
                    required: Some(false),
                },
            ]),
        });
        
        self.prompts.push(Prompt {
            name: "conversational_rag".to_string(),
            description: Some("RAG with conversation history".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "question".to_string(),
                    description: Some("Current question".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "history".to_string(),
                    description: Some("Conversation history".to_string()),
                    required: Some(false),
                },
                PromptArgument {
                    name: "table".to_string(),
                    description: Some("Table to search".to_string()),
                    required: Some(true),
                },
            ]),
        });
    }
    
    /// Register analysis prompts
    fn register_analysis_prompts(&mut self) {
        self.prompts.push(Prompt {
            name: "semantic_clustering".to_string(),
            description: Some("Cluster and analyze semantic groups".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "table".to_string(),
                    description: Some("Table to analyze".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "num_clusters".to_string(),
                    description: Some("Target number of clusters".to_string()),
                    required: Some(false),
                },
                PromptArgument {
                    name: "summary_type".to_string(),
                    description: Some("Type of summary to generate".to_string()),
                    required: Some(false),
                },
            ]),
        });
        
        self.prompts.push(Prompt {
            name: "similarity_matrix".to_string(),
            description: Some("Generate similarity matrix analysis".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "items".to_string(),
                    description: Some("Items to compare".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "metric".to_string(),
                    description: Some("Distance metric".to_string()),
                    required: Some(false),
                },
                PromptArgument {
                    name: "threshold".to_string(),
                    description: Some("Similarity threshold".to_string()),
                    required: Some(false),
                },
            ]),
        });
        
        self.prompts.push(Prompt {
            name: "outlier_detection".to_string(),
            description: Some("Detect and analyze outliers".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "table".to_string(),
                    description: Some("Table to analyze".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "contamination".to_string(),
                    description: Some("Expected outlier percentage".to_string()),
                    required: Some(false),
                },
            ]),
        });
    }
    
    /// Register data management prompts
    fn register_data_prompts(&mut self) {
        self.prompts.push(Prompt {
            name: "data_import_guide".to_string(),
            description: Some("Guide for importing data".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "data_type".to_string(),
                    description: Some("Type of data to import".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "source_format".to_string(),
                    description: Some("Format of source data".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "size_estimate".to_string(),
                    description: Some("Estimated data size".to_string()),
                    required: Some(false),
                },
            ]),
        });
        
        self.prompts.push(Prompt {
            name: "embedding_selection".to_string(),
            description: Some("Choose optimal embedding model".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "use_case".to_string(),
                    description: Some("Your use case".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "language".to_string(),
                    description: Some("Primary language".to_string()),
                    required: Some(false),
                },
                PromptArgument {
                    name: "performance_priority".to_string(),
                    description: Some("Speed vs accuracy preference".to_string()),
                    required: Some(false),
                },
            ]),
        });
        
        self.prompts.push(Prompt {
            name: "data_quality_check".to_string(),
            description: Some("Analyze data quality".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "table".to_string(),
                    description: Some("Table to check".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "checks".to_string(),
                    description: Some("Quality checks to perform".to_string()),
                    required: Some(false),
                },
            ]),
        });
    }
    
    /// Register optimization prompts
    fn register_optimization_prompts(&mut self) {
        self.prompts.push(Prompt {
            name: "index_optimization".to_string(),
            description: Some("Optimize index configuration".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "table".to_string(),
                    description: Some("Table to optimize".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "query_patterns".to_string(),
                    description: Some("Common query patterns".to_string()),
                    required: Some(false),
                },
                PromptArgument {
                    name: "optimization_goal".to_string(),
                    description: Some("Speed, accuracy, or balanced".to_string()),
                    required: Some(false),
                },
            ]),
        });
        
        self.prompts.push(Prompt {
            name: "performance_tuning".to_string(),
            description: Some("Tune system performance".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "workload_type".to_string(),
                    description: Some("Type of workload".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "current_metrics".to_string(),
                    description: Some("Current performance metrics".to_string()),
                    required: Some(false),
                },
                PromptArgument {
                    name: "target_metrics".to_string(),
                    description: Some("Target performance goals".to_string()),
                    required: Some(false),
                },
            ]),
        });
    }
    
    /// Register troubleshooting prompts
    fn register_troubleshooting_prompts(&mut self) {
        self.prompts.push(Prompt {
            name: "debug_search_issues".to_string(),
            description: Some("Debug search quality issues".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "problem_description".to_string(),
                    description: Some("Describe the issue".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "example_query".to_string(),
                    description: Some("Example problematic query".to_string()),
                    required: Some(false),
                },
                PromptArgument {
                    name: "expected_results".to_string(),
                    description: Some("What results were expected".to_string()),
                    required: Some(false),
                },
            ]),
        });
        
        self.prompts.push(Prompt {
            name: "performance_diagnosis".to_string(),
            description: Some("Diagnose performance problems".to_string()),
            arguments: Some(vec![
                PromptArgument {
                    name: "symptoms".to_string(),
                    description: Some("Performance symptoms".to_string()),
                    required: Some(true),
                },
                PromptArgument {
                    name: "when_started".to_string(),
                    description: Some("When issue started".to_string()),
                    required: Some(false),
                },
                PromptArgument {
                    name: "recent_changes".to_string(),
                    description: Some("Recent system changes".to_string()),
                    required: Some(false),
                },
            ]),
        });
    }
    
    /// Get all prompts
    pub fn get_prompts(&self) -> Vec<Prompt> {
        self.prompts.clone()
    }
    
    /// Get prompt by name
    pub fn get_prompt(&self, name: &str) -> Option<&Prompt> {
        self.prompts.iter().find(|p| p.name == name)
    }
    
    /// Generate prompt messages
    pub fn generate_prompt_messages(
        &self,
        name: &str,
        args: Option<HashMap<String, String>>,
    ) -> GetPromptResult {
        match name {
            "rag_basic" => self.generate_rag_basic(args),
            "rag_with_reranking" => self.generate_rag_reranking(args),
            "conversational_rag" => self.generate_conversational_rag(args),
            "semantic_clustering" => self.generate_semantic_clustering(args),
            "similarity_matrix" => self.generate_similarity_matrix(args),
            "outlier_detection" => self.generate_outlier_detection(args),
            "data_import_guide" => self.generate_import_guide(args),
            "embedding_selection" => self.generate_embedding_selection(args),
            "index_optimization" => self.generate_index_optimization(args),
            "debug_search_issues" => self.generate_debug_search(args),
            _ => self.generate_default_prompt(name, args),
        }
    }
    
    fn generate_rag_basic(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let question = args.as_ref()
            .and_then(|a| a.get("question"))
            .map(|s| s.as_str())
            .unwrap_or("[User question]");
        let table = args.as_ref()
            .and_then(|a| a.get("table"))
            .map(|s| s.as_str())
            .unwrap_or("documents");
        let top_k = args.as_ref()
            .and_then(|a| a.get("top_k"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(5);
        
        GetPromptResult {
            description: Some("Basic RAG pipeline execution".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        format!(
                            "You are an AI assistant with access to a vector database. \
                             Follow these steps to answer the user's question:\n\
                             1. Use the 'vector_search' tool to search the '{}' table\n\
                             2. Retrieve the top {} most relevant documents\n\
                             3. Analyze the retrieved context\n\
                             4. Generate a comprehensive answer based on the context\n\
                             5. Cite specific sources when making claims\n\
                             If the context doesn't contain relevant information, say so clearly.",
                            table, top_k
                        )
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(question.to_string()),
                },
            ],
        }
    }
    
    fn generate_rag_reranking(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let question = args.as_ref()
            .and_then(|a| a.get("question"))
            .map(|s| s.as_str())
            .unwrap_or("[User question]");
        let table = args.as_ref()
            .and_then(|a| a.get("table"))
            .map(|s| s.as_str())
            .unwrap_or("documents");
        let initial_k = args.as_ref()
            .and_then(|a| a.get("initial_k"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(20);
        let final_k = args.as_ref()
            .and_then(|a| a.get("final_k"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(5);
        
        GetPromptResult {
            description: Some("RAG with reranking for improved relevance".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        format!(
                            "You are an AI assistant performing RAG with reranking. Follow this process:\n\
                             \n\
                             1. Initial Retrieval:\n\
                                - Search the '{}' table for {} relevant documents\n\
                                - Cast a wider net to ensure coverage\n\
                             \n\
                             2. Reranking:\n\
                                - Analyze each retrieved document for relevance\n\
                                - Score based on:\n\
                                  • Direct relevance to the question\n\
                                  • Information quality\n\
                                  • Recency (if applicable)\n\
                                  • Source credibility\n\
                                - Select the top {} documents\n\
                             \n\
                             3. Answer Generation:\n\
                                - Synthesize information from reranked documents\n\
                                - Prioritize the most relevant sources\n\
                                - Provide citations\n\
                             \n\
                             Be explicit about your reranking rationale.",
                            table, initial_k, final_k
                        )
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(question.to_string()),
                },
            ],
        }
    }
    
    fn generate_conversational_rag(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let question = args.as_ref()
            .and_then(|a| a.get("question"))
            .map(|s| s.as_str())
            .unwrap_or("[Current question]");
        let history = args.as_ref()
            .and_then(|a| a.get("history"))
            .map(|s| s.as_str())
            .unwrap_or("");
        let table = args.as_ref()
            .and_then(|a| a.get("table"))
            .map(|s| s.as_str())
            .unwrap_or("documents");
        
        GetPromptResult {
            description: Some("Conversational RAG with context awareness".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        format!(
                            "You are a conversational AI assistant with vector database access. \n\
                             \n\
                             Conversation Management:\n\
                             1. Consider the conversation history for context\n\
                             2. Resolve pronouns and references from previous messages\n\
                             3. Maintain consistency with previous answers\n\
                             \n\
                             Search Strategy:\n\
                             1. Reformulate the query considering conversation context\n\
                             2. Search the '{}' table with the reformulated query\n\
                             3. Filter results considering what was already discussed\n\
                             \n\
                             Response Guidelines:\n\
                             - Build upon previous responses\n\
                             - Avoid repeating information unless asked\n\
                             - Acknowledge when changing or correcting previous statements\n\
                             - Maintain a conversational tone",
                            table
                        )
                    ),
                },
                PromptMessage {
                    role: MessageRole::Assistant,
                    content: MessageContent::Text(
                        if history.is_empty() {
                            "I'm ready to help you. What would you like to know?".to_string()
                        } else {
                            format!("Previous conversation:\n{}", history)
                        }
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(question.to_string()),
                },
            ],
        }
    }
    
    fn generate_semantic_clustering(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let table = args.as_ref()
            .and_then(|a| a.get("table"))
            .map(|s| s.as_str())
            .unwrap_or("vectors");
        let num_clusters = args.as_ref()
            .and_then(|a| a.get("num_clusters"))
            .and_then(|s| s.parse::<usize>().ok());
        
        GetPromptResult {
            description: Some("Perform semantic clustering analysis".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are an AI assistant specialized in semantic analysis. \
                         Use the vector database tools to perform clustering and provide insights.".to_string()
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(
                        format!(
                            "Perform semantic clustering on the '{}' table:\n\
                             1. Use the 'cluster_vectors' tool{}\n\
                             2. Analyze the resulting clusters\n\
                             3. Identify key themes for each cluster\n\
                             4. Find representative examples\n\
                             5. Suggest meaningful labels\n\
                             6. Identify any outliers or anomalies",
                            table,
                            num_clusters.map(|n| format!(" with {} clusters", n)).unwrap_or_default()
                        )
                    ),
                },
            ],
        }
    }
    
    fn generate_similarity_matrix(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let items = args.as_ref()
            .and_then(|a| a.get("items"))
            .map(|s| s.as_str())
            .unwrap_or("[items to compare]");
        
        GetPromptResult {
            description: Some("Generate and analyze similarity matrix".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are analyzing similarities between items using vector embeddings.".to_string()
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(
                        format!(
                            "Create a similarity analysis for: {}\n\
                             1. Generate or retrieve embeddings for each item\n\
                             2. Calculate pairwise similarities\n\
                             3. Create a similarity matrix\n\
                             4. Identify highly similar pairs\n\
                             5. Find dissimilar items\n\
                             6. Detect similarity clusters\n\
                             7. Provide insights and patterns",
                            items
                        )
                    ),
                },
            ],
        }
    }
    
    fn generate_outlier_detection(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let table = args.as_ref()
            .and_then(|a| a.get("table"))
            .map(|s| s.as_str())
            .unwrap_or("vectors");
        
        GetPromptResult {
            description: Some("Detect and analyze outliers".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are performing outlier detection analysis on vector data.".to_string()
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(
                        format!(
                            "Detect outliers in the '{}' table:\n\
                             1. Use the 'detect_anomalies' tool\n\
                             2. Analyze detected outliers\n\
                             3. Categorize outlier types\n\
                             4. Investigate potential causes\n\
                             5. Assess if outliers are errors or legitimate edge cases\n\
                             6. Provide recommendations for handling",
                            table
                        )
                    ),
                },
            ],
        }
    }
    
    fn generate_import_guide(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let data_type = args.as_ref()
            .and_then(|a| a.get("data_type"))
            .map(|s| s.as_str())
            .unwrap_or("text documents");
        let source_format = args.as_ref()
            .and_then(|a| a.get("source_format"))
            .map(|s| s.as_str())
            .unwrap_or("JSON");
        
        GetPromptResult {
            description: Some("Guide for data import".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are guiding a user through data import into the vector database.".to_string()
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(
                        format!(
                            "I need to import {} in {} format. Please guide me through:\n\
                             1. Data preparation and cleaning\n\
                             2. Choosing the right embedding model\n\
                             3. Table configuration (dimension, metric, index)\n\
                             4. Batch processing strategy\n\
                             5. Metadata extraction\n\
                             6. Import validation\n\
                             7. Post-import optimization",
                            data_type, source_format
                        )
                    ),
                },
            ],
        }
    }
    
    fn generate_embedding_selection(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let use_case = args.as_ref()
            .and_then(|a| a.get("use_case"))
            .map(|s| s.as_str())
            .unwrap_or("semantic search");
        
        GetPromptResult {
            description: Some("Select optimal embedding model".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are helping select the best embedding model for the use case.".to_string()
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(
                        format!(
                            "Help me choose an embedding model for {}:\n\
                             1. List available models with their characteristics\n\
                             2. Compare dimension sizes and performance\n\
                             3. Consider language support requirements\n\
                             4. Evaluate speed vs accuracy tradeoffs\n\
                             5. Recommend the best model\n\
                             6. Provide configuration settings\n\
                             7. Suggest evaluation metrics",
                            use_case
                        )
                    ),
                },
            ],
        }
    }
    
    fn generate_index_optimization(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let table = args.as_ref()
            .and_then(|a| a.get("table"))
            .map(|s| s.as_str())
            .unwrap_or("vectors");
        
        GetPromptResult {
            description: Some("Optimize index configuration".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are optimizing vector index configuration for performance.".to_string()
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(
                        format!(
                            "Optimize the index for table '{}':\n\
                             1. Analyze current index statistics\n\
                             2. Review query patterns and performance\n\
                             3. Compare index algorithms (HNSW, IVF, etc.)\n\
                             4. Tune index parameters\n\
                             5. Test different configurations\n\
                             6. Measure improvements\n\
                             7. Provide final recommendations",
                            table
                        )
                    ),
                },
            ],
        }
    }
    
    fn generate_debug_search(&self, args: Option<HashMap<String, String>>) -> GetPromptResult {
        let problem = args.as_ref()
            .and_then(|a| a.get("problem_description"))
            .map(|s| s.as_str())
            .unwrap_or("Search results are not relevant");
        
        GetPromptResult {
            description: Some("Debug search quality issues".to_string()),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are debugging vector search quality issues.".to_string()
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(
                        format!(
                            "Debug this search issue: {}\n\
                             \n\
                             Diagnostic steps:\n\
                             1. Verify embedding model consistency\n\
                             2. Check vector normalization\n\
                             3. Review distance metric selection\n\
                             4. Analyze vector distribution\n\
                             5. Test with known good examples\n\
                             6. Check for data quality issues\n\
                             7. Review index parameters\n\
                             8. Compare different search methods\n\
                             \n\
                             Provide diagnosis and solutions.",
                            problem
                        )
                    ),
                },
            ],
        }
    }
    
    fn generate_default_prompt(
        &self,
        name: &str,
        args: Option<HashMap<String, String>>,
    ) -> GetPromptResult {
        GetPromptResult {
            description: Some(format!("Prompt: {}", name)),
            messages: vec![
                PromptMessage {
                    role: MessageRole::System,
                    content: MessageContent::Text(
                        "You are an AI assistant with access to a vector database.".to_string()
                    ),
                },
                PromptMessage {
                    role: MessageRole::User,
                    content: MessageContent::Text(
                        format!("Execute the '{}' prompt with arguments: {:?}", name, args)
                    ),
                },
            ],
        }
    }
}