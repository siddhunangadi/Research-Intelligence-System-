"""
FastAPI Application Entry Point.
Initializes middleware filters and registers route controllers.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, upload, search, evaluate

# Setup logging levels
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Intelligence System REST API",
    description="Production-Grade Retrieval-Augmented Generation (RAG) platform endpoints.",
    version="1.0.0"
)

# Configure CORS policies to support web frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register route controllers
app.include_router(health.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(evaluate.router, prefix="/api")

@app.get("/")
def get_root():
    return {
        "service": "Research Intelligence System API",
        "version": "1.0.0",
        "documentation": "/docs"
    }
