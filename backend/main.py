import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import settings
from schemas import ChatRequest, ChatResponse, HealthResponse, Source, ErrorResponse
from rag_engine import get_rag_engine
from llm_provider import get_llm_provider

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
rag_engine = None
llm_provider = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global rag_engine, llm_provider
    
    # Startup
    logger.info("Starting Spark Chatbot...")
    try:
        rag_engine = get_rag_engine()
        llm_provider = get_llm_provider()
        logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Spark Chatbot...")

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="A specialized chatbot for Apache Spark questions using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
try:
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {str(e)}")

@app.get("/", response_class=FileResponse)
async def root():
    """Serve the main HTML page"""
    try:
        return FileResponse("../frontend/index.html")
    except Exception as e:
        logger.warning(f"Could not serve frontend: {str(e)}")
        return {"message": "Spark Chatbot API - Use /docs for API documentation"}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        llm_provider=settings.LLM_PROVIDER,
        vectordb_status="operational"
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat endpoint - Ask a question about Apache Spark
    
    Request body:
    - query: The question about Spark
    - chat_history: (Optional) Previous messages for context
    - user_id: (Optional) User identifier for tracking
    
    Returns:
    - answer: The chatbot's response
    - sources: Documentation sources used
    - confidence: Confidence score of the answer
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        if len(request.query) > 2000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is too long (max 2000 characters)"
            )
        
        start_time = time.time()
        
        # Log the query
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Generate answer using RAG
        answer, sources, confidence = rag_engine.generate_answer(
            request.query,
            llm_provider
        )
        
        # Convert sources to response format
        sources_list = [
            Source(
                title=source['title'],
                url=source['url'],
                relevance_score=source['relevance_score']
            )
            for source in sources
        ]
        
        processing_time = time.time() - start_time
        
        response = ChatResponse(
            answer=answer,
            sources=sources_list,
            confidence=confidence,
            processing_time=round(processing_time, 3),
            timestamp=datetime.now()
        )
        
        logger.info(f"Query processed in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )

@app.get("/docs", tags=["Documentation"])
async def documentation():
    """API documentation endpoint"""
    return {
        "title": "Spark Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Check chatbot health status",
            "POST /chat": "Ask a question about Apache Spark",
            "GET /docs": "View this documentation"
        },
        "example_query": {
            "query": "How do I create a DataFrame in PySpark?",
            "chat_history": []
        }
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return ErrorResponse(
        detail="An unexpected error occurred. Please try again.",
        timestamp=datetime.now()
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )