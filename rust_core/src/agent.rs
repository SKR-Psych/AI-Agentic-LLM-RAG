use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2, Array3, ArrayD};
use ndarray_rand::rand_distr::{Normal, Distribution};
use ndarray_rand::RandomExt;
use thiserror::Error;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::time::{SystemTime, UNIX_EPOCH};

// Re-export memory types
pub use crate::memory::{MemorySystem, MemoryType, MemoryItem, MemoryError};

/// Type alias for embeddings
pub type Embedding = Vec<f32>;

/// Maximum number of conversation turns to keep in context
const MAX_CONVERSATION_HISTORY: usize = 10;

/// Maximum number of memories to retrieve per search
const MAX_MEMORIES_TO_RETRIEVE: usize = 5;

/// Minimum similarity score for memory retrieval
const MEMORY_SIMILARITY_THRESHOLD: f32 = 0.75;

/// Error type for agent operations
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    #[error("Training error: {0}")]
    TrainingError(String),
    
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Tool error: {0}")]
    ToolError(String),
    
    #[error("Invalid input: {0}")]
    ValidationError(String),
    
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    
    #[error("Authentication error: {0}")]
    AuthenticationError(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
    
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    
    #[error(transparent)]
    MemorySystemError(#[from] MemoryError),
}

/// Result type for agent operations
pub type AgentResult<T> = Result<T, AgentError>;

/// Represents the role of a message in a conversation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MessageRole {
    /// System message that sets the behavior of the assistant
    System,
    /// User message that contains the user's input
    User,
    /// Assistant message that contains the model's response
    Assistant,
    /// Function message that contains the result of a function call
    Function,
    /// Tool message that contains the result of a tool execution
    Tool,
}

/// Represents a message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique identifier for the message
    pub id: String,
    /// Role of the message sender
    pub role: MessageRole,
    /// Content of the message
    pub content: String,
    /// Optional name of the sender (for function/tool calls)
    pub name: Option<String>,
    /// Optional function call information
    pub function_call: Option<FunctionCall>,
    /// Timestamp when the message was created
    pub timestamp: u64,
    /// Optional metadata for the message
    pub metadata: serde_json::Value,
}

impl Message {
    /// Create a new message with the given role and content
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        Self {
            id: Uuid::new_v4().to_string(),
            role,
            content: content.into(),
            name: None,
            function_call: None,
            timestamp: now,
            metadata: serde_json::json!({}),
        }
    }
    
    /// Create a new system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }
    
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }
    
    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }
    
    /// Create a new function message
    pub fn function(name: impl Into<String>, content: impl Into<String>) -> Self {
        let mut msg = Self::new(MessageRole::Function, content);
        msg.name = Some(name.into());
        msg
    }
    
    /// Create a new tool message
    pub fn tool(name: impl Into<String>, content: impl Into<String>) -> Self {
        let mut msg = Self::new(MessageRole::Tool, content);
        msg.name = Some(name.into());
        msg
    }
    
    /// Add metadata to the message
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
    
    /// Set the function call for this message
    pub fn with_function_call(mut self, function_call: FunctionCall) -> Self {
        self.function_call = Some(function_call);
        self
    }
}

/// Represents a function call in a message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function being called
    pub name: String,
    /// Arguments to pass to the function (as a JSON string)
    pub arguments: String,
    /// Optional ID for tracking the function call
    pub call_id: Option<String>,
}

impl FunctionCall {
    /// Create a new function call
    pub fn new(name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: arguments.into(),
            call_id: Some(Uuid::new_v4().to_string()),
        }
    }
    
    /// Parse the arguments as JSON
    pub fn parse_arguments<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Represents a function that can be called by the agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    /// Name of the function
    pub name: String,
    
    /// Description of what the function does
    pub description: String,
    
    /// JSON Schema object defining the parameters
    pub parameters: serde_json::Value,
    
    /// Whether the function is enabled
    pub enabled: bool,
    
    /// Whether the function requires authentication
    pub requires_auth: bool,
    
    /// Optional rate limit in calls per minute
    pub rate_limit: Option<u32>,
    
    /// Optional metadata
    pub metadata: serde_json::Value,
}

impl Function {
    /// Create a new function
    pub fn new<S: Into<String>>(name: S, description: S, parameters: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            enabled: true,
            requires_auth: false,
            rate_limit: None,
            metadata: serde_json::json!({}),
        }
    }
    
    /// Set whether the function is enabled
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
    
    /// Set whether the function requires authentication
    pub fn with_requires_auth(mut self, requires_auth: bool) -> Self {
        self.requires_auth = requires_auth;
        self
    }
    
    /// Set the rate limit for the function
    pub fn with_rate_limit(mut self, calls_per_minute: u32) -> Self {
        self.rate_limit = Some(calls_per_minute);
        self
    }
    
    /// Add metadata to the function
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Trait for tools that can be used by the agent
#[async_trait]
pub trait Tool: Send + Sync + std::fmt::Debug {
    /// Get the name of the tool
    fn name(&self) -> &str;
    
    /// Get the description of the tool
    fn description(&self) -> &str;
    
    /// Get the function definition for the tool
    fn function(&self) -> Function;
    
    /// Execute the tool with the given arguments
    async fn execute(&self, arguments: serde_json::Value) -> Result<serde_json::Value, String>;
    
    /// Get the schema of the tool's parameters
    fn parameters_schema(&self) -> serde_json::Value {
        self.function().parameters
    }
    
    /// Whether the tool is enabled
    fn is_enabled(&self) -> bool {
        true
    }
    
    /// Whether the tool requires authentication
    fn requires_auth(&self) -> bool {
        false
    }
    
    /// Get the rate limit for the tool in calls per minute
    fn rate_limit(&self) -> Option<u32> {
        None
    }
}

/// Trait for reasoning engines that can be used by the agent
#[async_trait]
pub trait ReasoningEngine: Send + Sync + std::fmt::Debug {
    /// Process a list of messages and return the model's response
    async fn process(
        &self, 
        messages: &[Message],
        tools: &[Box<dyn Tool>],
        temperature: Option<f32>,
        max_tokens: Option<usize>,
    ) -> AgentResult<Message>;
    
    /// Generate embeddings for the given texts
    async fn generate_embeddings(
        &self,
        texts: &[String],
    ) -> AgentResult<Vec<Embedding>>;
    
    /// Train the model on the given examples
    async fn train(
        &self,
        examples: &[(&str, &str)],
        learning_rate: Option<f32>,
        batch_size: Option<usize>,
        epochs: Option<usize>,
    ) -> AgentResult<()>;
    
    /// Add a function to the model's context
    async fn add_function(&mut self, function: Function) -> AgentResult<()>;
    
    /// Get all functions available to the model
    async fn get_functions(&self) -> Vec<Function>;
    
    /// Get the model's context window size in tokens
    fn context_window_size(&self) -> usize;
    
    /// Get the model's maximum output length in tokens
    fn max_output_length(&self) -> usize;
    
    /// Get the model's name or identifier
    fn model_name(&self) -> &str;
}

#[async_trait]
pub trait ReasoningEngine: Send + Sync + std::fmt::Debug {
    async fn process(&self, messages: &[Message]) -> Result<Message, AgentError>;
    
    async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AgentError>;
    
    async fn train(&self, data: &[(&str, &str)]) -> Result<(), AgentError>;
    
pub struct TransformerConfig {
    /// Size of the vocabulary
    pub vocab_size: usize,
    
    /// Maximum sequence length
    pub max_seq_len: usize,
    
    /// Size of the hidden layers
    pub hidden_size: usize,
    
    /// Number of attention heads
    pub num_attention_heads: usize,
    
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    
    /// Size of the intermediate layer in the feed-forward network
    pub intermediate_size: usize,
    
    /// Dropout probability for hidden layers
    pub hidden_dropout_prob: f32,
    
    /// Dropout probability for attention layers
    pub attention_probs_dropout_prob: f32,
    
    /// Range for parameter initialization
    pub initializer_range: f64,
    
    /// Layer normalization epsilon
    pub layer_norm_eps: f64,
    
    /// Whether to use gradient checkpointing
    pub gradient_checkpointing: bool,
    
    /// Whether to use mixed precision training
    pub use_mixed_precision: bool,
    
    /// Model name or identifier
    pub model_name: String,
    
    /// Model version
    pub model_version: String,
    
    /// Optional path to pre-trained weights
    pub model_path: Option<String>,
    
    /// Optional path to tokenizer
    pub tokenizer_path: Option<String>,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            max_seq_len: 512,
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            gradient_checkpointing: false,
            use_mixed_precision: false,
            model_name: "transformer".to_string(),
            model_version: "1.0.0".to_string(),
            model_path: None,
            tokenizer_path: None,
        }
    }
}

/// A transformer-based reasoning engine
#[derive(Debug)]
pub struct TransformerEngine {
    /// Model configuration
    config: TransformerConfig,
    
    /// Model parameters (weights and biases)
    parameters: HashMap<String, ArrayD<f32>>,
    
    /// Available functions that can be called
    functions: RwLock<HashMap<String, Function>>,
    
    /// Tokenizer for text processing
    tokenizer: Option<Box<dyn Tokenizer>>,
    
    /// Device to run computations on (CPU/GPU)
    device: Device,
    
    /// Whether the model is in training mode
    is_training: bool,
    
    /// Optional memory system for the model
    memory: Option<Arc<MemorySystem>>,
    
    /// Cache for attention keys and values
    cache: RwLock<HashMap<String, Vec<ArrayD<f32>>>>,
}

impl TransformerEngine {
    /// Create a new transformer engine with the given configuration
    pub fn new(config: TransformerConfig) -> Self {
        // Initialize model parameters based on config
        let mut parameters = HashMap::new();
        
        // Initialize embeddings
        parameters.insert(
            "embeddings.word_embeddings.weight".to_string(),
            Array2::random(
                (config.vocab_size, config.hidden_size),
                Normal::new(0.0, config.initializer_range).unwrap(),
            ).into_dyn(),
        );
        
        // Initialize position embeddings
        parameters.insert(
            "embeddings.position_embeddings.weight".to_string(),
            Array2::random(
                (config.max_seq_len, config.hidden_size),
                Normal::new(0.0, config.initializer_range).unwrap(),
            ).into_dyn(),
        );
        
        // Initialize layer normalization
        parameters.insert(
            "embeddings.LayerNorm.weight".to_string(),
            Array1::ones(config.hidden_size).into_dyn(),
        );
        
        parameters.insert(
            "embeddings.LayerNorm.bias".to_string(),
            Array1::zeros(config.hidden_size).into_dyn(),
        );
        
        // Initialize transformer layers
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{}", layer_idx);
            
            // Self attention
            parameters.insert(
                format!("{}.attention.self.query.weight", prefix),
                Array2::random(
                    (config.hidden_size, config.hidden_size),
                    Normal::new(0.0, config.initializer_range).unwrap(),
                ).into_dyn(),
            );
            
            // Initialize other attention parameters...
            
            // Layer normalization
            parameters.insert(
                format!("{}.attention.output.LayerNorm.weight", prefix),
                Array1::ones(config.hidden_size).into_dyn(),
            );
            
            parameters.insert(
                format!("{}.attention.output.LayerNorm.bias", prefix),
                Array1::zeros(config.hidden_size).into_dyn(),
            );
            
            // Feed-forward network
            parameters.insert(
                format!("{}.intermediate.dense.weight", prefix),
                Array2::random(
                    (config.intermediate_size, config.hidden_size),
                    Normal::new(0.0, config.initializer_range).unwrap(),
                ).into_dyn(),
            );
            
            // Initialize other feed-forward parameters...
        }
        
        // Initialize pooler
        parameters.insert(
            "pooler.dense.weight".to_string(),
            Array2::random(
                (config.hidden_size, config.hidden_size),
                Normal::new(0.0, config.initializer_range).unwrap(),
            ).into_dyn(),
        );
        
        parameters.insert(
            "pooler.dense.bias".to_string(),
            Array1::zeros(config.hidden_size).into_dyn(),
        );
        
        // Create device (CPU/GPU)
        let device = Device::default();
        
        // Load tokenizer if path is provided
        let tokenizer = config.tokenizer_path
            .as_ref()
            .map(|path| {
                // In a real implementation, this would load a tokenizer from disk
                Box::new(DummyTokenizer) as Box<dyn Tokenizer>
            });
        
        Self {
            config,
            parameters,
            functions: RwLock::new(HashMap::new()),
            tokenizer,
            device,
            is_training: false,
            memory: None,
            cache: RwLock::new(HashMap::new()),
        }
    }
    
    fn attention(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        // Scaled dot-product attention
        let dk = q.shape()[1] as f32;
        let scores = q.dot(&k.t()) / dk.sqrt();
        let attention_weights = softmax(&scores, 1);
        attention_weights.dot(v)
    }
    
    fn forward(&self, input_ids: &[usize]) -> Array2<f32> {
        // Simplified forward pass
        let seq_len = input_ids.len();
        let mut hidden_states = Array2::zeros((seq_len, self.config.model_size));
        
        // Embedding lookup
        for (i, &id) in input_ids.iter().enumerate() {
            // In a real implementation, this would be an embedding lookup
            hidden_states[[i, id % self.config.model_size]] = 1.0;
        }
        
        // Transformer layers
        for i in 0..self.config.num_layers {
            // Self-attention
            let q = &hidden_states;
            let k = &hidden_states;
            let v = &hidden_states;
            
            let attention_output = self.attention(q, k, v);
            
            // Add & Norm
            hidden_states = &hidden_states + &attention_output;
            hidden_states = layer_norm(&hidden_states);
            
            // Feed-forward
            // ...
        }
        
        hidden_states
    }
}

#[async_trait]
impl ReasoningEngine for TransformerEngine {
    async fn process(&self, messages: &[Message]) -> Result<Message, AgentError> {
        // Convert messages to prompt
        let prompt = messages
            .iter()
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");
            
        // Tokenize (simplified)
        let token_ids: Vec<usize> = prompt.chars().map(|c| c as usize).collect();
        
        // Run forward pass
        let logits = self.forward(&token_ids);
        
        // Generate response (simplified)
        let response = "This is a simulated response from the reasoning engine."
            .to_string();
            
        Ok(Message {
            role: MessageRole::Assistant,
            content: response,
            name: None,
            function_call: None,
        })
    }
    
    async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AgentError> {
        // Simulate embedding generation
        let mut embeddings = Vec::with_capacity(texts.len());
        
        for text in texts {
            // In a real implementation, this would use a proper embedding model
            let mut embedding = vec![0.0; self.config.model_size];
            let bytes: Vec<u8> = text.bytes().collect();
            
            for (i, &b) in bytes.iter().enumerate() {
                if i < self.config.model_size {
                    embedding[i] = (b as f32) / 255.0;
                }
            }
            
            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }
            
            embeddings.push(embedding);
        }
        
        Ok(embeddings)
    }
    
    async fn train(&self, data: &[(&str, &str)]) -> Result<(), AgentError> {
        // Simulate training
        println!("Training on {} examples...", data.len());
        Ok(())
    }
    
    fn add_function(&mut self, function: Function) -> Result<(), AgentError> {
        if self.functions.iter().any(|f| f.name == function.name) {
            return Err(AgentError::TrainingError(
                format!("Function {} already exists", function.name)
            ));
        }
        self.functions.push(function);
        Ok(())
    }
    
    fn get_functions(&self) -> Vec<Function> {
        self.functions.clone()
    }
}

// Utility functions
fn softmax(x: &Array2<f32>, axis: usize) -> Array2<f32> {
    let max = x.fold_axis(ndarray::Axis(axis), f32::NEG_INFINITY, |&a, &b| a.max(b));
    let exp = (x - &max.insert_axis(ndarray::Axis(axis))).mapv(f32::exp);
    &exp / &exp.sum_axis(ndarray::Axis(axis)).insert_axis(ndarray::Axis(axis))
}

fn layer_norm(x: &Array2<f32>) -> Array2<f32> {
    let mean = x.mean_axis(ndarray::Axis(1)).unwrap();
    let var = x.var_axis(ndarray::Axis(1), 1.0);
    
    let epsilon = 1e-5;
    let normalized = (x - &mean.insert_axis(ndarray::Axis(1))) 
        / (&var.mapv(|v| (v + epsilon).sqrt()).insert_axis(ndarray::Axis(1)));
    
    // In a real implementation, apply learned scale and shift
    normalized
}

// Agent implementation
#[derive(Debug)]
pub struct Agent {
    engine: Arc<RwLock<dyn ReasoningEngine>>,
    conversation_history: Vec<Message>,
    tools: HashMap<String, Box<dyn Tool + Send + Sync>>,
}

#[async_trait]
pub trait Tool: std::fmt::Debug {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;
    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String>;
}

impl Agent {
    pub fn new(engine: Arc<RwLock<dyn ReasoningEngine>>) -> Self {
        Self {
            engine,
            conversation_history: Vec::new(),
            tools: HashMap::new(),
        }
    }
    
    pub fn add_tool<T: Tool + Send + Sync + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().to_string(), Box::new(tool));
    }
    
    pub async fn process_message(&mut self, message: Message) -> Result<Message, AgentError> {
        self.conversation_history.push(message.clone());
        
        // Get engine read lock
        let engine = self.engine.read().map_err(|e| 
            AgentError::InferenceError(format!("Failed to acquire read lock: {}", e))
        )?;
        
        // Process with engine
        let response = engine.process(&[message]).await?;
        
        // Handle function calls if any
        if let Some(function_call) = &response.function_call {
            if let Some(tool) = self.tools.get(&function_call.name) {
                let args: serde_json::Value = serde_json::from_str(&function_call.arguments)
                    .map_err(AgentError::SerializationError)?;
                
                let result = tool.execute(args).await
                    .map_err(|e| AgentError::InferenceError(e))?;
                
                // Add function result to conversation
                let result_message = Message {
                    role: MessageRole::Function,
                    content: result,
                    name: Some(function_call.name.clone()),
                    function_call: None,
                };
                
                self.conversation_history.push(response);
                self.conversation_history.push(result_message.clone());
                
                // Get the final response
                return self.process_message(result_message).await;
            }
        }
        
        self.conversation_history.push(response.clone());
        Ok(response)
    }
    
    pub async fn train(&self, data: &[(&str, &str)]) -> Result<(), AgentError> {
        let mut engine = self.engine.write().map_err(|e| 
            AgentError::TrainingError(format!("Failed to acquire write lock: {}", e))
        )?;
        
        engine.train(data).await
    }
}

// Example tool implementation
#[derive(Debug)]
pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str { "calculator" }
    
    fn description(&self) -> &str { "Performs mathematical calculations" }
    
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        })
    }
    
    async fn execute(&self, arguments: serde_json::Value) -> Result<String, String> {
        // In a real implementation, use a proper expression evaluator
        let expr = arguments["expression"]
            .as_str()
            .ok_or("Invalid expression format")?;
            
        // Very basic and unsafe evaluation - for demonstration only
        // In production, use a proper math expression evaluator
        let result = if let Ok(val) = evalexpr::eval(expr) {
            val.to_string()
        } else {
            return Err("Failed to evaluate expression".to_string());
        };
        
        Ok(result)
    }
}

// Example usage
pub async fn simulate_reasoning(input: &str) -> String {
    let config = TransformerConfig::default();
    let engine = Arc::new(RwLock::new(TransformerEngine::new(config)));
    
    let mut agent = Agent::new(engine);
    agent.add_tool(CalculatorTool);
    
    let message = Message {
        role: MessageRole::User,
        content: input.to_string(),
        name: None,
        function_call: None,
    };
    
    match agent.process_message(message).await {
        Ok(response) => response.content,
        Err(e) => format!("Error: {}", e),
    }
}


fn calculate_payload() {
    // TODO: implement logic
}



fn build_data() {
    // TODO: implement logic
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn apply_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
    x.iter().enumerate().map(|(i, &val)| {
        weight[i] * (val - mean) / (var + eps).sqrt() + bias[i]
    }).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}


pub fn compute_attention_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    let scaled_scores: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_score = scaled_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp = exp_scores.iter().sum::<f32>();
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}

