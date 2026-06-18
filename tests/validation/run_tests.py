"""
Phase 7 Validation Pipeline Diagnostic Suite.
Tests claim extraction, NLI checks, and refusal handlers using standard unittest.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class TestRISPhase7(unittest.TestCase):
    def test_01_claim_extractor(self):
        """
        Tests parsing response text into claims and mapping inline citations.
        """
        print("\n[TEST] Running Claim Extractor...")
        from validation.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor()
        
        text = "The AdamW optimizer learning rate is 1e-4 [paper_01:methodology:4]. We report high validation accuracy [paper_01:results:6]."
        claims = extractor.extract_claims(text)
        
        self.assertEqual(len(claims), 2)
        # Check claim 1
        self.assertEqual(claims[0]["clean_claim"], "The AdamW optimizer learning rate is 1e-4 .")
        self.assertEqual(claims[0]["citations"], [("paper_01", "methodology", "4")])
        # Check claim 2
        self.assertEqual(claims[1]["clean_claim"], "We report high validation accuracy .")
        self.assertEqual(claims[1]["citations"], [("paper_01", "results", "6")])
        print("-> Inline citations and clean claims parsed successfully.")

    def test_02_nli_entailment_mock(self):
        """
        Checks mock NLI calculations based on token overlap ratios.
        """
        print("\n[TEST] Running Mock NLI checks...")
        from validation.nli_validator import NLIValidator
        validator = NLIValidator()
        
        premise = "The transformer architecture was optimized using the AdamW optimizer."
        
        # High overlap should pass
        res_pass = validator.check_entailment(premise, "optimized using AdamW optimizer")
        self.assertTrue(res_pass["passed"])
        self.assertEqual(res_pass["label"], "entailment")
        
        # Low overlap should fail
        res_fail = validator.check_entailment(premise, "SGD batch size was 256")
        self.assertFalse(res_fail["passed"])
        print("-> Mock NLI overlap rules triggered correctly.")

    def test_03_refusal_handler_triggers(self):
        """
        Verifies refusal triggers redirect ungrounded completions.
        """
        print("\n[TEST] Running Refusal Handler Gates...")
        from validation.refusal_handler import RefusalHandler
        handler = RefusalHandler()
        
        # Scenario A: Explicit refusal tokens in response
        self.assertTrue(handler.is_refusal_trigger("I cannot answer from the provided documents."))
        self.assertTrue(handler.is_refusal_trigger("INSUFFICIENT_EVIDENCE"))
        
        # Scenario B: Low faithfulness score triggers refusal redirect
        val_results = {"faithfulness_score": 0.33}
        final_ans = handler.process_response(
            raw_answer="The model achieves 99% accuracy.",
            validation_results=val_results,
            retrieved_chunks=[{"chunk_id": "c1"}]
        )
        self.assertEqual(final_ans, handler.fallback_message)
        
        # Scenario C: Clean response returns raw answer
        val_clean = {"faithfulness_score": 1.0}
        final_clean = handler.process_response(
            raw_answer="The model is optimized using AdamW [p1:m:4].",
            validation_results=val_clean,
            retrieved_chunks=[{"chunk_id": "c1"}]
        )
        self.assertEqual(final_clean, "The model is optimized using AdamW [p1:m:4].")
        print("-> Refusal gates checked successfully.")

if __name__ == "__main__":
    unittest.main()
