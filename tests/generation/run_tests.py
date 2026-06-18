"""
Phase 6 Answer Generation Diagnostic Suite.
Tests grounded system instructions and formatting cleaners using standard unittest.
"""

import os
import sys
import unittest

os.environ["TEST_MODE"] = "true"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TestRISPhase6(unittest.TestCase):
    def test_01_prompt_loader(self):
        """
        Tests that prompt compilation includes strict grounding keywords.
        """
        print("\n[TEST] Inspecting System Prompt Loader...")
        from generation.prompt_loader import PromptLoader
        loader = PromptLoader()
        
        system = loader.get_system_prompt()
        self.assertIn("INSUFFICIENT_EVIDENCE", system)
        self.assertIn("citation token", system)
        
        user_compiled = loader.compile_user_prompt("What learning rate?", "SOURCE: [p1:m:4] details")
        self.assertIn("What learning rate?", user_compiled)
        self.assertIn("SOURCE: [p1:m:4]", user_compiled)
        print("-> Prompt definitions compiled correctly.")

    def test_02_citation_formatter_period_swap(self):
        """
        Validates citation period cleaner moves periods outside brackets.
        """
        print("\n[TEST] Running Period Citation reordering...")
        from generation.citation_formatter import CitationFormatter
        formatter = CitationFormatter()
        
        # Test offset correction
        raw_text = "The AdamW optimizer is used for training models. [arxiv_1706_03762:methodology:4]"
        cleaned = formatter.clean_citation_periods(raw_text)
        self.assertEqual(
            cleaned,
            "The AdamW optimizer is used for training models [arxiv_1706_03762:methodology:4]."
        )
        
        # Verify it doesn't break already correct citations
        correct_text = "The model was trained with SGD [paper_abc:intro:2]."
        self.assertEqual(formatter.clean_citation_periods(correct_text), correct_text)
        print("-> Misplaced periods reordered successfully.")

    def test_03_completion_generator(self):
        """
        Asserts generator triggers simulated routes and resolves formats.
        """
        print("\n[TEST] Running Answer Generator pipeline...")
        from generation.answer_generator import AnswerGenerator
        generator = AnswerGenerator()
        
        # Simulated optimizer lookup
        ans1 = generator.generate_answer("What optimizer was used?", "Context blocks")
        self.assertIn("AdamW optimizer", ans1)
        self.assertTrue(ans1.endswith("]."))  # Period must be at the end
        
        # Simulated out-of-domain query
        ans2 = generator.generate_answer("What was the weather?", "Context blocks")
        self.assertEqual(ans2, "INSUFFICIENT_EVIDENCE")
        print("-> LLM Factory responses simulated correctly.")

if __name__ == "__main__":
    unittest.main()
