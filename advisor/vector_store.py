import numpy as np
import faiss
import openai
import os
import pickle
import hashlib
from datetime import datetime
from langsmith import traceable

class VectorStore:
    def __init__(self, dimension=1536, index_file="data/vector_index.faiss", data_file="data/vector_data.pkl"):
        self.dimension = dimension
        self.index_file = index_file
        self.data_file = data_file
        self.index = None
        self.data = []
        
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index or load existing one"""
        if os.path.exists(self.index_file) and os.path.exists(self.data_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.data_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.data = []
    
    @traceable(name="get_embedding", run_type="embedding")
    def _get_embedding(self, text):
        """Get embedding for text using OpenAI's embedding model"""
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = np.array([response.data[0].embedding], dtype=np.float32)
        
        return embedding
    
    @traceable(name="add_document", run_type="tool")
    def add_document(self, document, source_type, timestamp=None):
        """Add document to vector store"""
        if not timestamp:
            timestamp = datetime.now().isoformat()
        
        doc_hash = hashlib.md5(document.encode()).hexdigest()
        
        for item in self.data:
            if item.get('hash') == doc_hash:
                return {"status": "skipped", "reason": "duplicate", "hash": doc_hash}
        
        embedding = self._get_embedding(document)
        self.index.add(embedding)
        
        self.data.append({
            'text': document,
            'source_type': source_type,
            'timestamp': timestamp,
            'hash': doc_hash,
            'index': len(self.data)
        })
        
        self._save()
        
        return {
            "status": "added",
            "hash": doc_hash,
            "index": len(self.data) - 1,
            "document_length": len(document)
        }
    
    @traceable(name="vector_search", run_type="retriever")
    def search(self, query, k=5):
        """Search for similar documents"""
        if len(self.data) == 0:
            return []
        
        query_embedding = self._get_embedding(query)
        distances, indices = self.index.search(query_embedding, min(k, len(self.data)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.data):  
                result = {
                    'text': self.data[idx]['text'],
                    'source_type': self.data[idx]['source_type'],
                    'timestamp': self.data[idx]['timestamp'],
                    'distance': float(distances[0][i]),
                    'similarity_score': 1.0 / (1.0 + float(distances[0][i]))  
                }
                results.append(result)
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results
    
    def _save(self):
        """Save index and data to disk"""
        faiss.write_index(self.index, self.index_file)
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.data, f)
    
    @traceable(name="get_vector_store_stats", run_type="tool")
    def get_stats(self):
        """Get statistics about the vector store"""
        stats = {
            "total_documents": len(self.data),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension
        }
        
        source_counts = {}
        for item in self.data:
            source_type = item.get('source_type', 'unknown')
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        stats["source_type_counts"] = source_counts
        
        return stats