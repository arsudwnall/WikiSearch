import numpy as np
import faiss
import requests
from pydantic import BaseModel
from datetime import datetime

class MemoryItem(BaseModel):
    """A single memory entry"""
    text: str
    timestamp: str = datetime.now().isoformat()
    tool_name: str = None
    user_query: str = None

class MemoryManager:
    """Manages semantic memory storage and retrieval"""
    
    def __init__(self):
        self.embed_url = "http://localhost:11434/api/embeddings"
        self.model = "nomic-embed-text"
        self.index = None
        self.memories: list[MemoryItem] = []
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text"""
        response = requests.post(
            self.embed_url,
            json={"model": self.model, "input": text}
        )
        return np.array(response.json()["embedding"], dtype=np.float32)
    
    def add(self, item: MemoryItem):
        """Add a memory item"""
        embedding = self._get_embedding(item.text)
        
        # Initialize index on first add
        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embedding))
        
        self.index.add(embedding.reshape(1, -1))
        self.memories.append(item)
    
    def add_tool_result(self, tool_name: str, arguments: dict, result: str, user_query: str):
        """Convenience method to add tool execution results"""
        memory_text = f"Used {tool_name} with {arguments}, got: {result}"
        self.add(MemoryItem(
            text=memory_text,
            tool_name=tool_name,
            user_query=user_query
        ))
    
    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        """Retrieve most relevant memories"""
        if not self.index or len(self.memories) == 0:
            return []
        
        query_vec = self._get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)
        
        return [self.memories[idx] for idx in indices[0] if idx < len(self.memories)]