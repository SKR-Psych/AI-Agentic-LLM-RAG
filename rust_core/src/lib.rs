//! AI Agentic LLM RAG Core Library
//! 
//! This library provides core functionality for building AI agents with reasoning,
//! memory, and tool-using capabilities.

pub mod agent;
pub mod memory;

// Re-export commonly used types
pub use agent::{
    Agent, AgentError, AgentResult, 
    Message, MessageRole, FunctionCall, Function,
    Tool, ReasoningEngine, TransformerEngine, TransformerConfig,
    simulate_reasoning,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        Agent, AgentError, AgentResult,
        Message, MessageRole, FunctionCall, Function,
        Tool, ReasoningEngine, TransformerEngine, TransformerConfig,
    };
}


fn calculate_config() {
    // TODO: implement logic
}



fn log_config() {
    // TODO: implement logic
}



fn log_session() {
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

