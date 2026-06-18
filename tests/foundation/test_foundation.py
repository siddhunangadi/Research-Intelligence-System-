"""
Phase 1 Unit Tests.
Verifies system configs, ChromaDB persistence indices, and SQL database schemas.
"""

import os
import pytest
from configs.config_manager import get_config, ConfigManager
from indexing.chroma_store import ChromaVectorStore
from indexing.postgres_store import PaperModel, ChunkModel
from sqlalchemy import inspect

def test_system_config_loading():
    """
    Verifies that system configurations are parsed and validated correctly.
    """
    config = get_config()
    assert config is not None
    assert config.chunking.target_size >= 100
    assert config.chunking.overlap >= 0
    assert config.embedding.dimension == 1024
    assert config.database.postgres_url is not None
    assert config.database.chroma_path is not None

def test_chroma_store_operations(tmp_path):
    """
    Verifies vector store additions, similarity queries, and deletion capabilities.
    """
    # Override chroma path with temp directory for clean test isolation
    test_chroma_path = os.path.join(tmp_path, "test_chroma")
    os.environ["CHROMA_DB_PATH"] = test_chroma_path
    
    # Reload config manager instance to pick up overrides
    manager = ConfigManager()
    
    store = ChromaVectorStore(collection_name="test_collection")
    
    ids = ["chk_1", "chk_2"]
    embeddings = [
        [0.1] * 1024,
        [0.9] * 1024
    ]
    documents = [
        "The transformer was optimized using AdamW optimizer.",
        "We achieved a validation accuracy of 92.4% on GLUE benchmarks."
    ]
    metadatas = [
        {"paper_id": "p1", "section": "methodology", "page_number": 4},
        {"paper_id": "p1", "section": "results", "page_number": 6}
    ]
    
    # Add vectors
    success = store.add_vectors(ids, embeddings, documents, metadatas)
    assert success is True
    
    # Similarity query
    query_vec = [0.1] * 1024
    results = store.query_similarity(query_vec, top_k=2)
    assert len(results) == 2
    # The first item should be 'chk_1' because cosine distance should be 0 (similarity 1.0)
    assert results[0]["chunk_id"] == "chk_1"
    assert "methodology" in results[0]["section"]
    
    # Filter query
    results_filtered = store.query_similarity(query_vec, top_k=1, where_filter={"section": "results"})
    assert len(results_filtered) == 1
    assert results_filtered[0]["chunk_id"] == "chk_2"
    
    # Delete test
    delete_success = store.delete_by_paper("p1")
    assert delete_success is True
    
    # Validate delete
    results_after_delete = store.query_similarity(query_vec, top_k=2)
    assert len(results_after_delete) == 0

def test_sql_model_schemas():
    """
    Asserts SQL models are mapped with the correct schemas.
    """
    # Check paper model properties
    assert hasattr(PaperModel, "paper_id")
    assert hasattr(PaperModel, "title")
    assert hasattr(PaperModel, "authors")
    assert hasattr(PaperModel, "year")
    assert hasattr(PaperModel, "venue")
    
    # Check chunk model properties
    assert hasattr(ChunkModel, "chunk_id")
    assert hasattr(ChunkModel, "paper_id")
    assert hasattr(ChunkModel, "section")
    assert hasattr(ChunkModel, "page_number")
    assert hasattr(ChunkModel, "token_count")
    assert hasattr(ChunkModel, "content")
    assert hasattr(ChunkModel, "embedding_id")
