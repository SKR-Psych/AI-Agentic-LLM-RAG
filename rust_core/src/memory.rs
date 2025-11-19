use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::{Array1, Array2, ArrayD};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

use crate::agent::Embedding;

/// Error type for memory operations
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Type alias for memory result
type MemoryResult<T> = Result<T, MemoryError>;

/// Represents the type of memory
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryType {
    Fact,
    Event,
    Skill,
    Preference,
    Observation,
    Plan,
    Goal,
}

/// Represents a single memory item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub embedding: Option<Embedding>,
    pub metadata: serde_json::Value,
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: u64,
    pub importance: f32,
}

impl MemoryItem {
    /// Create a new memory item
    pub fn new<S: Into<String>>(
        content: S,
        memory_type: MemoryType,
        embedding: Option<Embedding>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            memory_type,
            embedding,
            metadata: serde_json::json!({}),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            importance: 0.5, // Default importance
        }
    }
    
    /// Update the last accessed time and increment access count
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.access_count += 1;
    }
}

/// Configuration for the memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub max_short_term_items: usize,
    pub max_long_term_items: usize,
    pub importance_threshold: f32,
    pub decay_factor: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_short_term_items: 100,
            max_long_term_items: 10_000,
            importance_threshold: 0.7,
            decay_factor: 0.9,
        }
    }
}

/// The main memory system
pub struct MemorySystem {
    short_term: RwLock<HashMap<String, MemoryItem>>,
    long_term: RwLock<HashMap<String, MemoryItem>>,
    config: MemoryConfig,
}

impl MemorySystem {
    /// Create a new memory system with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryConfig::default())
    }
    
    /// Create a new memory system with custom configuration
    pub fn with_config(config: MemoryConfig) -> Self {
        Self {
            short_term: RwLock::new(HashMap::new()),
            long_term: RwLock::new(HashMap::new()),
            config,
        }
    }
    
    /// Store a new memory item
    pub async fn store(
        &self,
        content: String,
        memory_type: MemoryType,
        embedding: Option<Embedding>,
    ) -> MemoryResult<String> {
        let mut item = MemoryItem::new(content, memory_type, embedding);
        
        // If importance is high enough, store in long-term memory
        if item.importance >= self.config.importance_threshold {
            let mut long_term = self.long_term.write().await;
            
            // If we've reached capacity, remove the least important memory
            if long_term.len() >= self.config.max_long_term_items {
                self.evict_least_important(&mut long_term).await;
            }
            
            let id = item.id.clone();
            long_term.insert(id.clone(), item);
            Ok(id)
        } else {
            let mut short_term = self.short_term.write().await;
            
            // If we've reached capacity, remove the least important memory
            if short_term.len() >= self.config.max_short_term_items {
                self.evict_least_important(&mut short_term).await;
            }
            
            let id = item.id.clone();
            short_term.insert(id.clone(), item);
            Ok(id)
        }
    }
    
    /// Retrieve a memory by ID
    pub async fn recall(&self, id: &str) -> MemoryResult<Option<MemoryItem>> {
        // First check short-term memory
        {
            let short_term = self.short_term.read().await;
            if let Some(mut item) = short_term.get(id).cloned() {
                item.touch();
                return Ok(Some(item));
            }
        }
        
        // Then check long-term memory
        {
            let mut long_term = self.long_term.write().await;
            if let Some(item) = long_term.get_mut(id) {
                item.touch();
                return Ok(Some(item.clone()));
            }
        }
        
        Ok(None)
    }
    
    /// Search for memories similar to the query embedding
    pub async fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_similarity: f32,
    ) -> MemoryResult<Vec<(MemoryItem, f32)>> {
        let mut results = Vec::new();
        
        // Search short-term memory
        {
            let short_term = self.short_term.read().await;
            for item in short_term.values() {
                if let Some(embedding) = &item.embedding {
                    let similarity = cosine_similarity(query_embedding, embedding);
                    if similarity >= min_similarity {
                        results.push((item.clone(), similarity));
                    }
                }
            }
        }
        
        // Search long-term memory
        {
            let long_term = self.long_term.read().await;
            for item in long_term.values() {
                if let Some(embedding) = &item.embedding {
                    let similarity = cosine_similarity(query_embedding, embedding);
                    if similarity >= min_similarity {
                        results.push((item.clone(), similarity));
                    }
                }
            }
        }
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top results
        results.truncate(limit);
        
        Ok(results)
    }
    
    /// Consolidate important short-term memories to long-term storage
    pub async fn consolidate(&self) -> MemoryResult<usize> {
        let mut count = 0;
        let mut to_remove = Vec::new();
        
        // Identify important memories to move to long-term
        {
            let short_term = self.short_term.read().await;
            for (id, item) in short_term.iter() {
                if item.importance >= self.config.importance_threshold {
                    to_remove.push(id.clone());
                }
            }
        }
        
        // Move memories to long-term storage
        for id in to_remove {
            if let Some(item) = self.short_term.write().await.remove(&id) {
                let mut long_term = self.long_term.write().await;
                
                // If we've reached capacity, remove the least important memory
                if long_term.len() >= self.config.max_long_term_items {
                    self.evict_least_important(&mut long_term).await;
                }
                
                long_term.insert(id, item);
                count += 1;
            }
        }
        
        Ok(count)
    }
    
    /// Forget a memory by ID
    pub async fn forget(&self, id: &str) -> MemoryResult<bool> {
        if self.short_term.write().await.remove(id).is_some() {
            return Ok(true);
        }
        
        if self.long_term.write().await.remove(id).is_some() {
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Save memories to disk
    pub async fn save_to_disk<P: AsRef<std::path::Path>>(&self, path: P) -> MemoryResult<()> {
        let snapshot = MemorySnapshot {
            short_term: self.short_term.read().await.values().cloned().collect(),
            long_term: self.long_term.read().await.values().cloned().collect(),
            config: self.config.clone(),
        };
        
        let json = serde_json::to_string_pretty(&snapshot)?;
        tokio::fs::write(path, json).await?;
        
        Ok(())
    }
    
    /// Load memories from disk
    pub async fn load_from_disk<P: AsRef<std::path::Path>>(path: P) -> MemoryResult<Self> {
        let json = tokio::fs::read_to_string(path).await?;
        let snapshot: MemorySnapshot = serde_json::from_str(&json)?;
        
        let mut short_term = HashMap::new();
        let mut long_term = HashMap::new();
        
        for item in snapshot.short_term {
            short_term.insert(item.id.clone(), item);
        }
        
        for item in snapshot.long_term {
            long_term.insert(item.id.clone(), item);
        }
        
        Ok(Self {
            short_term: RwLock::new(short_term),
            long_term: RwLock::new(long_term),
            config: snapshot.config,
        })
    }
    
    /// Evict the least important memory from a storage
    async fn evict_least_important<T: std::ops::DerefMut<Target = HashMap<String, MemoryItem>>>(
        &self,
        storage: &mut T,
    ) -> Option<MemoryItem> {
        let (id, _) = storage
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.importance
                    .partial_cmp(&b.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
            
        let id = id.clone();
        storage.remove(&id)
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Snapshot of the memory system for serialization
#[derive(Debug, Serialize, Deserialize)]
struct MemorySnapshot {
    short_term: Vec<MemoryItem>,
    long_term: Vec<MemoryItem>,
    config: MemoryConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_memory_storage() {
        let memory = MemorySystem::new();
        
        // Test storing and recalling
        let id = memory.store("Test memory".to_string(), MemoryType::Fact, None)
            .await
            .unwrap();
            
        let item = memory.recall(&id).await.unwrap().unwrap();
        assert_eq!(item.content, "Test memory");
        
        // Test forgetting
        assert!(memory.forget(&id).await.unwrap());
        assert!(memory.recall(&id).await.unwrap().is_none());
    }
    
    #[tokio::test]
    async fn test_memory_search() {
        let memory = MemorySystem::new();
        
        // Store some memories with embeddings
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![0.0, 1.0, 0.0];
        let query_embedding = vec![0.9, 0.1, 0.0];
        
        memory.store("Memory 1".to_string(), MemoryType::Fact, Some(embedding1))
            .await
            .unwrap();
            
        memory.store("Memory 2".to_string(), MemoryType::Fact, Some(embedding2))
            .await
            .unwrap();
        
        // Search for similar memories
        let results = memory.search(&query_embedding, 1, 0.7).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.content, "Memory 1");
    }
}


fn fetch_cache() {
    // TODO: implement logic
}



fn build_state() {
    // TODO: implement logic
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

