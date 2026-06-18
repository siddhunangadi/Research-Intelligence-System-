"""
Phase 4 Retrieval Pipeline Diagnostic Suite.
Tests BM25 math, RRF scores fusion, rerank thresholds, and packing reorders using standard unittest.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class TestRISPhase4(unittest.TestCase):
    def test_01_bm25_calculations(self):
        """
        Tests BM25 term matching and document score ordering.
        """
        print("\n[TEST] Running BM25 Ranking checks...")
        from retrieval.bm25_retriever import BM25Retriever
        
        chunks = [
            {"chunk_id": "c1", "paper_id": "p1", "content": "The AdamW optimizer is used for training models."},
            {"chunk_id": "c2", "paper_id": "p1", "content": "We achieve validation accuracy on GLUE benchmarks."},
            {"chunk_id": "c3", "paper_id": "p2", "content": "SGD optimizer is another gradient descent method."}
        ]
        
        retriever = BM25Retriever()
        retriever.fit(chunks)
        
        # Test baseline retrieval query
        results = retriever.retrieve("AdamW optimizer")
        self.assertEqual(len(results), 2)  # Should return c1 and c3 containing word 'optimizer'
        self.assertEqual(results[0]["chunk_id"], "c1")  # c1 has 'AdamW' and 'optimizer' so higher score
        
        # Test metadata filtering bounds
        results_filtered = retriever.retrieve("optimizer", paper_id="p2")
        self.assertEqual(len(results_filtered), 1)
        self.assertEqual(results_filtered[0]["chunk_id"], "c3")
        print("-> BM25 TF-IDF matches calculated successfully.")

    def test_02_rrf_fusion_merges(self):
        """
        Tests merging of rank rankings from sparse and dense candidate files.
        """
        print("\n[TEST] Running RRF Rankings Fusion...")
        from retrieval.rrf_fusion import compute_rrf
        
        sparse_hits = [
            {"chunk_id": "c1", "content": "AdamW weights"},
            {"chunk_id": "c2", "content": "SGD weights"}
        ]
        dense_hits = [
            {"chunk_id": "c2", "content": "SGD weights"},
            {"chunk_id": "c3", "content": "Adagrad weights"}
        ]
        
        # Unify candidates
        fused = compute_rrf(sparse_hits, dense_hits, k=60, top_n=3)
        self.assertEqual(len(fused), 3)
        
        # chunk c2 appears in both lists (rank 2 in sparse, rank 1 in dense), so it should rank highest!
        self.assertEqual(fused[0]["chunk_id"], "c2")
        self.assertTrue(fused[0]["rrf_score"] > fused[1]["rrf_score"])
        print("-> Reciprocal Rank Fusion consolidated lists correctly.")

    def test_03_context_reordering_limits(self):
        """
        Asserts reordering places rank 1 at the boundaries to solve "Lost in the Middle".
        """
        print("\n[TEST] Running Lost-in-the-Middle Context Reordering...")
        from retrieval.context_packer import lost_in_the_middle_reorder
        
        chunks = [
            {"chunk_id": "c1", "rrf_score": 0.9},
            {"chunk_id": "c2", "rrf_score": 0.8},
            {"chunk_id": "c3", "rrf_score": 0.7},
            {"chunk_id": "c4", "rrf_score": 0.6},
            {"chunk_id": "c5", "rrf_score": 0.5}
        ]
        
        reordered = lost_in_the_middle_reorder(chunks)
        # Reordered output structure should have top rank c1 at index 0, and c2 at the end index
        self.assertEqual(reordered[0]["chunk_id"], "c1")
        self.assertEqual(reordered[-1]["chunk_id"], "c2")
        # Lower ranks (c5, c4) must nest inward in the middle of the array
        self.assertEqual(reordered[2]["chunk_id"], "c5")
        print("-> Attention profile reordering resolved successfully.")

if __name__ == "__main__":
    unittest.main()
