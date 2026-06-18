"""
FastAPI Health status checks router.
Exposes database and vector storage connectivity diagnostic endpoints.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from database.connection import get_db_session
from indexing.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["system"])

@router.get("")
def check_health(db: Session = Depends(get_db_session)):
    """
    Checks liveness of local vector stores and PostgreSQL databases.
    """
    status = {
        "status": "healthy",
        "postgres": "connected",
        "chromadb": "connected"
    }
    
    # 1. Check PostgreSQL
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        status["postgres"] = "unreachable"
        status["status"] = "unhealthy"

    # 2. Check ChromaDB
    try:
        store = ChromaVectorStore()
        # Verify collection count queries
        store.collection.count()
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        status["chromadb"] = "unreachable"
        status["status"] = "unhealthy"

    if status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=status)
        
    return status

# Import text parser for SQL health checks
from sqlalchemy import text
