"""
Phase 1 Diagnostic and Verification Suite.
Runs foundation validation using the built-in unittest library.
"""

import os
import sys
import tempfile
import unittest

# Adjust python path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class TestRISPhase1(unittest.TestCase):
    def test_01_config_loading(self):
        """
        Tests Pydantic config validation and environment reading.
        """
        print("\n[TEST] Running Configuration Validation...")
        from configs.config_manager import get_config
        config = get_config()
        self.assertIsNotNone(config)
        self.assertGreaterEqual(config.chunking.target_size, 100)
        self.assertGreaterEqual(config.chunking.overlap, 0)
        self.assertEqual(config.embedding.dimension, 1024)
        self.assertIsNotNone(config.database.postgres_url)
        self.assertIsNotNone(config.database.chroma_path)
        print("-> Configuration parsed successfully.")

    def test_02_chroma_operations(self):
        """
        Tests ChromaDB vector index additions and similarity searches using temporary path.
        """
        print("\n[TEST] Running ChromaDB Operations...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["CHROMA_DB_PATH"] = tmp_dir
            
            # Reload configuration config manager
            from configs.config_manager import ConfigManager
            ConfigManager._instance = None # Force re-init
            
            from indexing.chroma_store import ChromaVectorStore
            store = ChromaVectorStore(collection_name="test_temp_collection")
            
            ids = ["test_c1"]
            embeddings = [[0.15] * 1024]
            documents = ["The transformer was optimized using AdamW with learning rate 1e-4."]
            metadatas = [{"paper_id": "p1", "section": "methodology", "page_number": 4}]
            
            # Insert vector
            add_ok = store.add_vectors(ids, embeddings, documents, metadatas)
            self.assertTrue(add_ok)
            
            # Vector query
            results = store.query_similarity([0.15] * 1024, top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["chunk_id"], "test_c1")
            self.assertEqual(results[0]["section"], "methodology")
            
            # Deletion check
            del_ok = store.delete_by_paper("p1")
            self.assertTrue(del_ok)
            
            # Validate deleted state
            results_after = store.query_similarity([0.15] * 1024, top_k=1)
            self.assertEqual(len(results_after), 0)
            print("-> ChromaDB vector addition, querying, and deletion completed successfully.")

    def test_03_sql_schemas(self):
        """
        Tests SQLAlchemy declarative mapping and columns declarations.
        """
        print("\n[TEST] Checking PostgreSQL SQL Schemas...")
        from indexing.postgres_store import PaperModel, ChunkModel
        
        self.assertTrue(hasattr(PaperModel, "paper_id"))
        self.assertTrue(hasattr(PaperModel, "title"))
        self.assertTrue(hasattr(ChunkModel, "chunk_id"))
        self.assertTrue(hasattr(ChunkModel, "content"))
        print("-> Database schema mapping validated successfully.")

if __name__ == "__main__":
    unittest.main()
