from mcp.server.fastmcp import FastMCP
import faiss
import numpy as np
from pathlib import Path
import requests
from markitdown import MarkItDown
import json
import hashlib


mcp = FastMCP("mcp_server")

# Configuration
EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 40
ROOT = Path(__file__).parent.resolve()
DOC_PATH = ROOT / "documents"
INDEX_PATH = ROOT / "faiss_index"

# Get embedding vector for text
def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for text"""
    response = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    return np.array(response.json()["embedding"], dtype=np.float32)

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
    return chunks

@mcp.tool()
def search_documents(query: str) -> list[str]:
    """Search for relevant content from uploaded documents"""
    index = faiss.read_index(str(INDEX_PATH / "index.bin"))
    metadata = json.loads((INDEX_PATH / "metadata.json").read_text())
    
    query_vec = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vec, k=5)
    
    results = []
    for idx in indices[0]:
        data = metadata[idx]
        results.append(f"{data['chunk']}\n[Source: {data['doc']}]")
    
    return results

def process_documents():
    """Index documents with FAISS"""
    INDEX_PATH.mkdir(exist_ok=True)
    cache_file = INDEX_PATH / "doc_cache.json"
    
    # Load existing index and cache
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    metadata = []
    index = None
    
    if (INDEX_PATH / "index.bin").exists():
        index = faiss.read_index(str(INDEX_PATH / "index.bin"))
        metadata = json.loads((INDEX_PATH / "metadata.json").read_text())
    
    converter = MarkItDown()
    
    for file in DOC_PATH.glob("*.*"):
        file_hash = hashlib.md5(file.read_bytes()).hexdigest()
        
        # Skip if unchanged
        if cache.get(file.name) == file_hash:
            print(f"Skipping: {file.name}")
            continue
        
        print(f"Processing: {file.name}")
        
        # Convert and chunk document
        markdown = converter.convert(str(file)).text_content
        chunks = chunk_text(markdown)
        
        # Generate embeddings
        embeddings = [get_embedding(chunk) for chunk in chunks]
        
        # Initialize or update index
        if index is None:
            index = faiss.IndexFlatL2(len(embeddings[0]))
        
        index.add(np.stack(embeddings))
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            metadata.append({
                "doc": file.name,
                "chunk": chunk,
                "chunk_id": f"{file.stem}_{i}"
            })
        
        cache[file.name] = file_hash
    
    # Save everything
    if index and index.ntotal > 0:
        faiss.write_index(index, str(INDEX_PATH / "index.bin"))
        (INDEX_PATH / "metadata.json").write_text(json.dumps(metadata, indent=2))
        cache_file.write_text(json.dumps(cache, indent=2))
        print("Index saved successfully")


if __name__ == "__main__":
    import sys
    
    if "dev" in sys.argv:
        mcp.run()
    else:
        # Process documents then start server
        process_documents()
        mcp.run(transport="stdio")