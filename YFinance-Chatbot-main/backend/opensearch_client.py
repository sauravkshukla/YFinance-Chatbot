"""OpenSearch Vector Database Client"""
import os
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json


class OpenSearchVectorDB:
    def __init__(self):
        """Initialize OpenSearch client and embedding model"""
        self.host = os.getenv("OPENSEARCH_HOST", "localhost")
        self.port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        self.user = os.getenv("OPENSEARCH_USER", "admin")
        self.password = os.getenv("OPENSEARCH_PASSWORD", "admin")
        self.use_ssl = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
        
        # Initialize OpenSearch client
        try:
            self.client = OpenSearch(
                hosts=[{'host': self.host, 'port': self.port}],
                http_auth=(self.user, self.password),
                use_ssl=self.use_ssl,
                verify_certs=False,
                ssl_show_warn=False
            )
            print(f"✅ OpenSearch connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"⚠️ OpenSearch connection failed: {e}")
            self.client = None
        
        # Initialize embedding model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence Transformer model loaded")
        except Exception as e:
            print(f"⚠️ Embedding model load failed: {e}")
            self.model = None
    
    def create_index(self, index_name: str):
        """Create an index with vector search capabilities"""
        if not self.client:
            return False
        
        try:
            if self.client.indices.exists(index=index_name):
                print(f"Index '{index_name}' already exists")
                return True
            
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 384,  # all-MiniLM-L6-v2 dimension
                            "method": {
                                "name": "hnsw",
                                "space_type": "l2",
                                "engine": "nmslib"
                            }
                        },
                        "text": {"type": "text"},
                        "title": {"type": "text"},
                        "summary": {"type": "text"},
                        "url": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "published": {"type": "date"},
                        "timestamp": {"type": "date"},
                        "type": {"type": "keyword"}
                    }
                }
            }
            
            self.client.indices.create(index=index_name, body=index_body)
            print(f"✅ Created index '{index_name}'")
            return True
        except Exception as e:
            print(f"Error creating index: {e}")
            return False
    
    def embed_text(self, text: str):
        """Generate embedding for text"""
        if not self.model:
            return None
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def index_document(self, index_name: str, doc_id: str, document: dict, skip_if_exists=True):
        """Index a document with vector embedding (optimized)"""
        if not self.client:
            return False
        
        try:
            # Check if document already exists (skip re-indexing)
            if skip_if_exists:
                try:
                    if self.client.exists(index=index_name, id=doc_id):
                        return True  # Already indexed
                except:
                    pass
            
            # Generate embedding from text
            text_to_embed = document.get('text', document.get('summary', document.get('title', '')))
            embedding = self.embed_text(text_to_embed)
            
            if embedding:
                document['embedding'] = embedding
                document['timestamp'] = datetime.now().isoformat()
                
                self.client.index(
                    index=index_name,
                    id=doc_id,
                    body=document,
                    refresh=False  # Don't refresh immediately for better performance
                )
                return True
        except Exception as e:
            print(f"Indexing error: {e}")
            return False
    
    def search_similar(self, index_name: str, query_text: str, k: int = 5):
        """Search for similar documents using vector similarity"""
        if not self.client or not self.model:
            return []
        
        try:
            query_embedding = self.embed_text(query_text)
            if not query_embedding:
                return []
            
            search_body = {
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": k
                        }
                    }
                }
            }
            
            response = self.client.search(index=index_name, body=search_body)
            return [hit['_source'] for hit in response['hits']['hits']]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def hybrid_search(self, index_name: str, query_text: str, k: int = 5):
        """Hybrid search combining vector and keyword search"""
        if not self.client or not self.model:
            return []
        
        try:
            query_embedding = self.embed_text(query_text)
            if not query_embedding:
                return []
            
            search_body = {
                "size": k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": k
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["title^3", "summary^2", "text"]
                                }
                            }
                        ]
                    }
                }
            }
            
            response = self.client.search(index=index_name, body=search_body)
            return [hit['_source'] for hit in response['hits']['hits']]
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []
    
    def get_document(self, index_name: str, doc_id: str):
        """Get a specific document by ID"""
        if not self.client:
            return None
        
        try:
            response = self.client.get(index=index_name, id=doc_id)
            return response['_source']
        except Exception as e:
            print(f"Get document error: {e}")
            return None
    
    def delete_index(self, index_name: str):
        """Delete an index"""
        if not self.client:
            return False
        
        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                print(f"✅ Deleted index '{index_name}'")
                return True
        except Exception as e:
            print(f"Delete index error: {e}")
            return False


# Global instance
vector_db = None

def get_vector_db():
    """Get or create OpenSearch vector DB instance"""
    global vector_db
    if vector_db is None:
        vector_db = OpenSearchVectorDB()
    return vector_db
