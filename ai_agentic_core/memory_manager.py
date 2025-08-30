from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from enum import Enum

class MemoryType(Enum):
    FACT = "fact"
    EVENT = "event"
    SKILL = "skill"
    PREFERENCE = "preference"
    
@dataclass
class MemoryItem:
    content: str
    memory_type: MemoryType
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    last_accessed: datetime = None
    access_count: int = 0
    importance: float = 0.5  # 0 to 1 scale
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "memory_type": self.memory_type.value,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        item = cls(
            content=data['content'],
            memory_type=MemoryType(data['memory_type']),
            embedding=np.array(data['embedding']) if data['embedding'] else None,
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            access_count=data.get('access_count', 0),
            importance=data.get('importance', 0.5)
        )
        return item

class MemoryManager:
    def __init__(self, embedding_dim: int = 384):
        self.long_term: Dict[str, MemoryItem] = {}
        self.short_term: List[MemoryItem] = []
        self.embedding_dim = embedding_dim
        self.max_short_term = 10
        
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for memory content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def store(self, 
             content: str, 
             memory_type: MemoryType = MemoryType.FACT,
             embedding: Optional[np.ndarray] = None,
             metadata: Optional[Dict[str, Any]] = None,
             importance: float = 0.5) -> str:
        """
        Store a memory with optional embedding and metadata
        
        Args:
            content: The content to store
            memory_type: Type of memory (FACT, EVENT, etc.)
            embedding: Optional vector embedding of the content
            metadata: Additional metadata
            importance: Importance score (0-1)
            
        Returns:
            str: Memory ID
        """
        memory_id = self._generate_id(content)
        
        # Update if exists, else create new
        if memory_id in self.long_term:
            self.long_term[memory_id].last_accessed = datetime.utcnow()
            self.long_term[memory_id].access_count += 1
            if metadata:
                self.long_term[memory_id].metadata.update(metadata)
        else:
            self.long_term[memory_id] = MemoryItem(
                content=content,
                memory_type=memory_type,
                embedding=embedding,
                metadata=metadata or {},
                importance=importance
            )
            
        return memory_id
    
    def retrieve(self, 
                query: str, 
                top_k: int = 5,
                min_similarity: float = 0.7,
                memory_type: Optional[MemoryType] = None) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieve memories similar to the query
        
        Args:
            query: Query string
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity score (0-1)
            memory_type: Optional filter by memory type
            
        Returns:
            List of (memory_item, similarity_score) tuples
        """
        # In a real implementation, this would use a proper vector database
        # For now, we'll simulate semantic search with a simple keyword match
        query = query.lower()
        results = []
        
        for mem_id, item in self.long_term.items():
            if memory_type and item.memory_type != memory_type:
                continue
                
            # Simple keyword matching as a placeholder for real semantic search
            score = sum(1 for word in query.split() if word in item.content.lower()) / len(query.split())
            
            # Add some randomness to simulate vector similarity
            score += np.random.uniform(-0.1, 0.1)
            score = max(0, min(1, score))  # Clamp to [0, 1]
            
            if score >= min_similarity:
                results.append((item, score))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def buffer(self, item: str, importance: float = 0.5) -> str:
        """Add an item to short-term memory"""
        memory_item = MemoryItem(
            content=item,
            memory_type=MemoryType.EVENT,
            importance=importance
        )
        self.short_term.append(memory_item)
        
        # Enforce short-term memory limit
        if len(self.short_term) > self.max_short_term:
            # Remove least important or oldest memory
            self.short_term.sort(key=lambda x: (x.importance, x.last_accessed))
            self.short_term.pop(0)
            
        return self._generate_id(item)
    
    def consolidate(self) -> List[str]:
        """
        Consolidate short-term memories into long-term storage
        
        Returns:
            List of memory IDs that were consolidated
        """
        consolidated = []
        for item in self.short_term:
            # Only consolidate important enough memories
            if item.importance > 0.6:  # Threshold for consolidation
                mem_id = self.store(
                    content=item.content,
                    memory_type=item.memory_type,
                    embedding=item.embedding,
                    metadata=item.metadata,
                    importance=item.importance
                )
                consolidated.append(mem_id)
        
        # Clear short-term memory after consolidation
        self.short_term = []
        return consolidated
    
    def forget(self, memory_id: str) -> bool:
        """Remove a memory by ID"""
        if memory_id in self.long_term:
            del self.long_term[memory_id]
            return True
        return False
    
    def save_to_disk(self, filepath: str) -> bool:
        """Save memories to disk"""
        try:
            data = {
                'long_term': {k: v.to_dict() for k, v in self.long_term.items()},
                'short_term': [item.to_dict() for item in self.short_term]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, default=str, indent=2)
            return True
        except Exception as e:
            print(f"Error saving memories: {e}")
            return False
    
    @classmethod
    def load_from_disk(cls, filepath: str) -> 'MemoryManager':
        """Load memories from disk"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            manager = cls()
            manager.long_term = {
                k: MemoryItem.from_dict(v) 
                for k, v in data.get('long_term', {}).items()
            }
            manager.short_term = [
                MemoryItem.from_dict(item) 
                for item in data.get('short_term', [])
            ]
            return manager
        except Exception as e:
            print(f"Error loading memories: {e}")
            return cls()  # Return new manager if loading fails

def compute_memory_retrieval_score(query, memory):
    """Compute retrieval score for memory search."""
    query_embedding = self.encode_query(query)
    memory_embedding = memory.embedding
    similarity = torch.cosine_similarity(query_embedding, memory_embedding, dim=0)
    return similarity * memory.importance


def self_reflection_loop(initial_response, max_iterations=3):
    """Implement self-reflection for response improvement."""
    current_response = initial_response
    
    for iteration in range(max_iterations):
        # Analyze current response
        analysis = self.analyze_response_quality(current_response)
        
        if analysis['score'] > 0.8:  # Good enough
            break
        
        # Generate improvement suggestions
        suggestions = self.generate_improvement_suggestions(analysis)
        
        # Apply improvements
        current_response = self.improve_response(current_response, suggestions)
    
    return current_response


def self_reflection_loop(initial_response, max_iterations=3):
    """Implement self-reflection for response improvement."""
    current_response = initial_response
    
    for iteration in range(max_iterations):
        # Analyze current response
        analysis = self.analyze_response_quality(current_response)
        
        if analysis['score'] > 0.8:  # Good enough
            break
        
        # Generate improvement suggestions
        suggestions = self.generate_improvement_suggestions(analysis)
        
        # Apply improvements
        current_response = self.improve_response(current_response, suggestions)
    
    return current_response


def compute_kl_divergence(p, q):
    """Compute KL divergence between two probability distributions."""
    return torch.sum(p * torch.log(p / q))

