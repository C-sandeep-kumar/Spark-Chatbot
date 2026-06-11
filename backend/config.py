import os
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Spark Chatbot"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # LLM Model Settings
    MODEL_NAME: str = "gpt-3.5-turbo"  # For OpenAI
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000
    
    # Spark Documentation
    SPARK_DOCS_URL: str = os.getenv("SPARK_DOCS_URL", "https://spark.apache.org/docs/latest/")
    
    # Vector Database
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vectordb")
    VECTOR_DB_TYPE: str = "faiss"  # Options: chroma, faiss
    
    # RAG Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", 5))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # CORS
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:5173",
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
