"""
Phase 8 Evaluation Framework Diagnostic Suite.
Tests Precision@K, Recall@K, MRR, NDCG, and batch database logging using standard unittest.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class TestRISPhase8(unittest.TestCase):
    def test_01_retrieval_accuracy_metrics(self):
        """
        Tests mathematical retrieval accuracy calculations.
        """
        print("\n[TEST] Running Retrieval Accuracy Metrics...")
        from evaluation.metrics.retrieval_metrics import (
            calculate_precision_at_k,
            calculate_recall_at_k,
            calculate_mrr,
            calculate_ndcg
        )
        
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        ground_truth = ["c2", "c5"]
        
        # 1. Precision@5 = 2/5 = 0.4
        self.assertEqual(calculate_precision_at_k(retrieved, ground_truth, k=5), 0.4)
        # 2. Precision@2 = 1/2 = 0.5
        self.assertEqual(calculate_precision_at_k(retrieved, ground_truth, k=2), 0.5)
        
        # 3. Recall@5 = 2/2 = 1.0
        self.assertEqual(calculate_recall_at_k(retrieved, ground_truth, k=5), 1.0)
        # 4. Recall@2 = 1/2 = 0.5
        self.assertEqual(calculate_recall_at_k(retrieved, ground_truth, k=2), 0.5)
        
        # 5. MRR = 1 / index(c2) = 1/2 = 0.5
        self.assertEqual(calculate_mrr(retrieved, ground_truth), 0.5)
        
        # 6. NDCG@5 DCG calculations
        # DCG@5 = 0 / log2(2) + 1 / log2(3) + 0 / log2(4) + 0 / log2(5) + 1 / log2(6)
        # = 0 + 0.6309 + 0 + 0 + 0.3868 = 1.0177
        # IDCG@5 = 1 / log2(2) + 1 / log2(3) = 1 + 0.6309 = 1.6309
        # NDCG@5 = 1.0177 / 1.6309 = 0.624
        ndcg_score = calculate_ndcg(retrieved, ground_truth, k=5)
        self.assertAlmostEqual(ndcg_score, 1.0177/1.6309, places=2)
        print("-> Precision, Recall, MRR, and NDCG values match mathematical targets.")

    def test_02_answer_relevancy_metrics(self):
        """
        Tests semantic query similarity alignment.
        """
        print("\n[TEST] Running Answer Relevancy Metrics...")
        from evaluation.metrics.answer_relevancy import AnswerRelevancyMetric
        
        # Mock embedder output
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024
        mock_embedder.embed_documents.return_value = [[0.1] * 1024]
        
        relevancy_eval = AnswerRelevancyMetric(embedder=mock_embedder)
        score = relevancy_eval.calculate_relevancy("AdamW optimizer", "The AdamW optimizer is used for models.")
        
        # Same vector output should result in 1.0 cosine similarity
        self.assertAlmostEqual(score, 1.0)
        print("-> Relevancy checks match cosine similarity profiles.")

    def test_03_evaluation_runner_batch(self):
        """
        Verifies batch evaluations run without failures and write output summaries.
        """
        print("\n[TEST] Running Batch Evaluation Runner...")
        from evaluation.evaluation_runner import RISEvaluationRunner
        
        runner = RISEvaluationRunner()
        
        # Mock connection and query outputs
        runner.log_evaluation_to_db = MagicMock(return_value=True)
        runner.generator.generate_answer = MagicMock(return_value="The transformer architecture was optimized using the AdamW optimizer with a learning rate of 1e-4 [paper_01:methodology:4].")
        
        # Setup mock DB payload
        mock_db = {
            "sparse_db": [
                {"chunk_id": "c1", "paper_id": "paper_01", "section": "methodology", "page_number": 4, "content": "The architecture was optimized using AdamW with a learning rate of 1e-4.", "embedding": [0.1] * 1024}
            ],
            "dense_db": [
                {"chunk_id": "c1", "paper_id": "paper_01", "section": "methodology", "page_number": 4, "content": "The architecture was optimized using AdamW with a learning rate of 1e-4.", "embedding": [0.1] * 1024}
            ]
        }
        
        # Run evaluations
        report = runner.run_evaluations(mock_db=mock_db)
        
        # Assertions
        self.assertIn("metrics", report)
        self.assertTrue(os.path.exists("results.json"))
        print("-> Batch evaluation metrics compiled successfully.")
        
        # Clean local file
        if os.path.exists("results.json"):
            os.remove("results.json")

if __name__ == "__main__":
    unittest.main()
