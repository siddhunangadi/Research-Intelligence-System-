"""
Phase 3 Embedding & Sync Diagnostic Suite.
Tests embedding dimensions, prefixes, and atomic database sync checks using standard unittest.
"""

import os
import sys
import unittest
import tempfile
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class TestRISPhase3(unittest.TestCase):
    def test_01_embedding_generation(self):
        """
        Verifies that embedder outputs vectors with correct dimensionality (1024).
        """
        print("\n[TEST] Running Embedding Vector checks...")
        from embeddings.embedder import ScientificEmbedder
        embedder = ScientificEmbedder()
        
        # Test document embedding dimension
        docs = ["Test document content", "Another scientific context text block."]
        vectors = embedder.embed_documents(docs)
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 1024)
        
        # Test query prefix embedding
        query_vector = embedder.embed_query("What is the learning rate?")
        self.assertEqual(len(query_vector), 1024)
        print("-> Vector dimension checked successfully (1024 dims).")

    def test_02_batch_embedder(self):
        """
        Tests batch slicing mechanisms.
        """
        print("\n[TEST] Running Batch Embedder Slices...")
        from embeddings.batch_embedding import BatchEmbedder
        from embeddings.embedder import ScientificEmbedder
        
        embedder = ScientificEmbedder()
        batch_embedder = BatchEmbedder(embedder=embedder, batch_size=2)
        
        chunks = [
            {"chunk_id": "c1", "content": "Text block 1"},
            {"chunk_id": "c2", "content": "Text block 2"},
            {"chunk_id": "c3", "content": "Text block 3"}
        ]
        
        vectorized = batch_embedder.embed_chunks_batch(chunks)
        self.assertEqual(len(vectorized), 3)
        self.assertIn("embedding", vectorized[0])
        self.assertEqual(len(vectorized[0]["embedding"]), 1024)
        print("-> Sliced lists processed cleanly.")

    def test_03_index_builder_transaction_sync(self):
        """
        Verifies rollback logic on storage builder: if vector writes fail, PostgreSQL records must clear.
        """
        print("\n[TEST] Running Transactional Index Sync...")
        from indexing.index_builder import IndexBuilder
        
        # Mock stores
        mock_pg = MagicMock()
        mock_chroma = MagicMock()
        
        # Setup Postgres return states
        mock_pg.insert_paper.return_value = True
        mock_pg.insert_chunks_batch.return_value = 1
        
        # Force Chroma insert failure to trigger rollback
        mock_chroma.add_vectors.return_value = False
        
        builder = IndexBuilder(
            postgres_store=mock_pg,
            chroma_store=mock_chroma
        )
        
        # Mock ingestion pipeline outputs
        builder.pipeline.ingest_document = MagicMock(return_value=(
            {"paper_id": "paper_fail", "title": "Test Paper", "authors": None, "year": None, "venue": None},
            [{"chunk_id": "chk_1", "paper_id": "paper_fail", "section": "intro", "page_number": 1, "token_count": 10, "embedding_id": "emb_1", "content": "Intro", "start_char": 0, "end_char": 5}]
        ))
        
        # Run builder
        success = builder.build_index_for_pdf("dummy.pdf")
        
        # Index build must report failure
        self.assertFalse(success)
        # PostgreSQL must run delete_paper to roll back the inserted paper record
        mock_pg.delete_paper.assert_called_with("paper_fail")
        print("-> Relational records rolled back correctly on vector DB failure.")

if __name__ == "__main__":
    unittest.main()
