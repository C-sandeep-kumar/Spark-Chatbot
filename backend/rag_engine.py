import logging
import os
import time
from typing import List, Tuple, Optional
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from spark_docs_loader import SparkDocsLoader, get_sample_spark_docs
from config import settings

logger = logging.getLogger(__name__)

class RAGEngine:
    """Retrieval Augmented Generation Engine for Spark Chatbot"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.top_k = settings.TOP_K_DOCUMENTS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        
        # Initialize embeddings
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Initialize vector database
        self._init_vectordb()
        
        # Load documents
        self._load_documents()
    
    def _init_vectordb(self):
        """Initialize vector database (FAISS)"""
        try:
            # Create data directory if it doesn't exist
            Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
            
            self.db_path = os.path.join(settings.VECTOR_DB_PATH, "documents.pkl")
            self.embeddings_path = os.path.join(settings.VECTOR_DB_PATH, "embeddings.pkl")
            
            # Load existing database if available
            if os.path.exists(self.db_path) and os.path.exists(self.embeddings_path):
                with open(self.db_path, 'rb') as f:
                    self.documents = pickle.load(f)
                with open(self.embeddings_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Loaded existing vector database with {len(self.documents)} documents")
            else:
                self.documents = []
                self.embeddings = []
            
            logger.info(f"Vector database initialized at {settings.VECTOR_DB_PATH}")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            self.documents = []
            self.embeddings = []
    
    def _load_documents(self):
        """Load documents into vector database"""
        try:
            # Check if documents already exist
            if len(self.documents) > 0:
                logger.info(f"Vector database already contains {len(self.documents)} documents")
                return
            
            logger.info("Loading Spark documentation...")
            
            # Load documentation
            loader = SparkDocsLoader()
            docs = loader.fetch_documentation()
            
            # Fall back to sample docs if fetching fails
            if not docs:
                logger.warning("Using sample Spark documentation")
                docs = [("Sample Spark Docs", get_sample_spark_docs())]
            
            # Process documents
            doc_id = 0
            for title, content in docs:
                chunks = SparkDocsLoader.chunk_text(
                    content,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
                for chunk in chunks:
                    if chunk.strip():
                        doc_id += 1
                        self._add_to_vectordb(
                            doc_id=f"doc_{doc_id}",
                            content=chunk,
                            metadata={"source": title, "type": "spark_docs"}
                        )
            
            logger.info(f"Loaded {doc_id} document chunks into vector database")
            self._save_vectordb()
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            # Continue with empty database rather than failing
    
    def _add_to_vectordb(self, doc_id: str, content: str, metadata: dict):
        """Add a document to the vector database"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Add to database
            self.documents.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata
            })
            self.embeddings.append(embedding)
        except Exception as e:
            logger.error(f"Error adding document {doc_id} to vector database: {str(e)}")
    
    def _save_vectordb(self):
        """Save vector database to disk"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.documents, f)
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info("Vector database saved")
        except Exception as e:
            logger.error(f"Error saving vector database: {str(e)}")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float, str]]:
        """
        Retrieve relevant documents for a query
        Returns list of (content, similarity_score, source) tuples
        """
        try:
            if top_k is None:
                top_k = self.top_k
            
            if not self.documents or not self.embeddings:
                logger.warning("No documents in vector database")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarities
            import numpy as np
            query_embedding = np.array(query_embedding)
            embeddings_array = np.array(self.embeddings)
            
            # Cosine similarity
            similarities = np.dot(embeddings_array, query_embedding) / (
                np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            retrieved = []
            for idx in top_indices:
                similarity = float(similarities[idx])
                
                # Filter by similarity threshold
                if similarity >= self.similarity_threshold:
                    doc = self.documents[idx]
                    source = doc["metadata"].get("source", "Spark Documentation")
                    retrieved.append((doc["content"], similarity, source))
            
            return retrieved
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def build_context(self, query: str, top_k: Optional[int] = None) -> Tuple[str, List[dict]]:
        """
        Build context from retrieved documents
        Returns (context_text, sources_list)
        """
        retrieved = self.retrieve(query, top_k)
        
        context_parts = []
        sources = []
        
        for content, similarity, source in retrieved:
            context_parts.append(content)
            sources.append({
                "title": source,
                "relevance_score": round(float(similarity), 3),
                "url": "https://spark.apache.org/docs/latest/"
            })
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant documentation found."
        
        return context, sources
    
    def generate_answer(self, query: str, llm_provider) -> Tuple[str, List[dict], float]:
        """
        Generate an answer using the LLM with RAG context
        Returns (answer, sources, confidence)
        """
        start_time = time.time()
        
        try:
            # Build context
            context, sources = self.build_context(query)
            
            # Generate answer using LLM
            answer = None
            try:
                # Try async generation
                import asyncio
                answer = asyncio.run(llm_provider.generate(query, context))
            except:
                # Fall back to sync
                answer = llm_provider.generate(query, context)
            
            # Calculate confidence based on retrieved documents
            confidence = 0.7 if sources else 0.3
            if sources:
                avg_relevance = sum(s['relevance_score'] for s in sources) / len(sources)
                confidence = min(0.95, 0.5 + (avg_relevance * 0.5))
            
            processing_time = time.time() - start_time
            
            return answer, sources, round(confidence, 3)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

def get_rag_engine() -> RAGEngine:
    """Get or create RAG engine singleton"""
    if not hasattr(get_rag_engine, "_instance"):
        get_rag_engine._instance = RAGEngine()
    return get_rag_engine._instance
