"""
Phase 5 Retrieval Pipeline Orchestration Diagnostic Suite.
Tests query routing intent classification, SQL dynamic clause generation, and pipeline routing queries.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class TestRISPhase5(unittest.TestCase):
    def test_01_query_intent_classifier(self):
        """
        Tests query classification keyword triggers routing.
        """
        print("\n[TEST] Running Query Intent Classifier...")
        from retrieval.intent_classifier import IntentClassifier
        classifier = IntentClassifier()
        
        # Methodology triggers
        self.assertIn("methodology", classifier.classify_intent("What is the learning rate of SGD?"))
        self.assertIn("methodology", classifier.classify_intent("Which optimizer was used in the training pipeline?"))
        
        # Results triggers
        self.assertIn("results", classifier.classify_intent("What validation accuracy did the model achieve on GLUE?"))
        
        # No triggers (should return empty list)
        self.assertEqual(len(classifier.classify_intent("Hello, how are you?")), 0)
        print("-> Query intents routed correctly.")

    def test_02_metadata_filter_builders(self):
        """
        Verifies filter mapping translates search conditions to valid DB criteria.
        """
        print("\n[TEST] Running Metadata Filter Builders...")
        from retrieval.metadata_filters import MetadataFilterBuilder
        builder = MetadataFilterBuilder()
        
        # 1. Single section filter
        f1 = builder.build_chroma_filter(paper_id="paper_abc", sections=["intro"])
        self.assertEqual(f1["$and"][0]["paper_id"], "paper_abc")
        self.assertEqual(f1["$and"][1]["section"], "intro")
        
        # 2. Multi-section filter
        f2 = builder.build_chroma_filter(paper_id="p123", sections=["methodology", "results"])
        self.assertEqual(f2["$and"][0]["paper_id"], "p123")
        self.assertEqual(f2["$and"][1]["section"]["$in"], ["methodology", "results"])
        
        # 3. PostgreSQL dynamic clauses builder
        sql_clause, params = builder.build_postgres_clause(paper_id="p9", sections=["intro", "results"])
        self.assertIn("WHERE paper_id = :paper_id AND section IN", sql_clause)
        self.assertEqual(params["paper_id"], "p9")
        self.assertEqual(params["sec_0"], "intro")
        self.assertEqual(params["sec_1"], "results")
        print("-> Database filtering clauses compiled successfully.")

    def test_03_pipeline_orchestration(self):
        """
        Asserts pipeline orchestrates retrieval flows, RRF, and reordering.
        """
        print("\n[TEST] Running Full Retrieval Pipeline Orchestrator...")
        from retrieval.retrieval_pipeline import RetrievalPipeline
        
        # Mock database integrations
        mock_pg = MagicMock()
        mock_chroma = MagicMock()
        
        # Setup mock postgres chunk repository output
        mock_pg_chunks = [
            {"chunk_id": "c1", "paper_id": "p1", "section": "methodology", "page_number": 4, "token_count": 20, "embedding_id": "emb_1", "content": "The AdamW optimizer learning rate is 1e-4."}
        ]
        
        # Setup mock ChromaDB query result
        mock_chroma_chunks = [
            {"chunk_id": "c1", "content": "The AdamW optimizer learning rate is 1e-4.", "similarity_score": 0.85, "paper_id": "p1", "section": "methodology", "page_number": 4, "token_count": 20, "embedding_id": "emb_1"}
        ]
        mock_chroma.query_similarity.return_value = mock_chroma_chunks
        
        pipeline = RetrievalPipeline(
            postgres_store=mock_pg,
            chroma_store=mock_chroma
        )
        
        # Mock postgres method inside retrieve_grounded_context
        pipeline._fetch_postgres_chunks = MagicMock(return_value=mock_pg_chunks)
        
        context_str, chunks = pipeline.retrieve_grounded_context("What optimizer was used?", paper_id="p1")
        
        # Assertions
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "c1")
        self.assertIn("SOURCE: [p1:methodology:4]", context_str)
        self.assertIn("AdamW optimizer", context_str)
        print("-> Hybrid pipeline retrieval flow completed successfully.")

if __name__ == "__main__":
    unittest.main()
