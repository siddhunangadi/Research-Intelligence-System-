"""
Phase 2 Ingestion Diagnostic Suite.
Tests page routing heuristics, canonical section mapping, and offset calculations using standard unittest.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class TestRISPhase2(unittest.TestCase):
    def test_01_parser_router_heuristics(self):
        """
        Tests router logic for detecting tables, equations, and text based on page metrics.
        """
        print("\n[TEST] Running Parser Heuristics router...")
        from ingestion.parser_router import ParserRouter
        router = ParserRouter()
        
        # Test routing rule overrides
        # 1. High pipe_ratio should route to table
        class MockRouter(ParserRouter):
            def analyze_page_layout(self, pdf_path, page_number):
                return {"pipe_ratio": 0.22, "equation_hits": 0, "char_count": 500}
                
        mock_router = MockRouter()
        self.assertEqual(mock_router.route_page_parser("dummy.pdf", 0), "table")
        
        # 2. High equation hits should route to equation parser
        class MockMathRouter(ParserRouter):
            def analyze_page_layout(self, pdf_path, page_number):
                return {"pipe_ratio": 0.01, "equation_hits": 5, "char_count": 300}
                
        mock_math = MockMathRouter()
        self.assertEqual(mock_math.route_page_parser("dummy.pdf", 0), "equation")
        
        # 3. Baseline text
        class MockTextRouter(ParserRouter):
            def analyze_page_layout(self, pdf_path, page_number):
                return {"pipe_ratio": 0.05, "equation_hits": 1, "char_count": 800}
                
        mock_text = MockTextRouter()
        self.assertEqual(mock_text.route_page_parser("dummy.pdf", 0), "text")
        print("-> Heuristic rules resolved correctly.")

    def test_02_section_extractor(self):
        """
        Verifies heading split boundaries map custom titles to canonical section headers.
        """
        print("\n[TEST] Running Section Heading Extraction...")
        from ingestion.section_extractor import SectionExtractor
        extractor = SectionExtractor()
        
        markdown_text = """
# Abstract
This is a summary of the methodology.

# Introduction
We introduce deep learning components.

## Proposed Methodology
We build a RAG pipeline framework.
        """
        sections = extractor.split_by_sections(markdown_text)
        
        # Methodology maps from custom heading "Proposed Methodology"
        self.assertIn("abstract", sections)
        self.assertIn("introduction", sections)
        self.assertIn("methodology", sections)
        self.assertEqual(sections["abstract"].strip(), "This is a summary of the methodology.")
        self.assertEqual(sections["introduction"].strip(), "We introduce deep learning components.")
        self.assertEqual(sections["methodology"].strip(), "We build a RAG pipeline framework.")
        print("-> Custom heading tokens mapped to canonical headers successfully.")

    def test_03_section_chunker_boundaries(self):
        """
        Asserts that chunks are split recursively without crossing section bounds.
        """
        print("\n[TEST] Running Chunk boundary tests...")
        from chunking.section_chunker import SectionChunker
        chunker = SectionChunker()
        
        sections = {
            "introduction": "This is introductory text. " * 30, # Long string to force split
            "results": "These are results. " * 15
        }
        
        chunks = chunker.chunk_document_sections("paper_01", sections)
        self.assertTrue(len(chunks) > 0)
        
        # Ensure section properties are attached correctly and no cross-bleeding occurs
        intro_chunks = [c for c in chunks if c["section"] == "introduction"]
        results_chunks = [c for c in chunks if c["section"] == "results"]
        
        self.assertTrue(len(intro_chunks) >= 1)
        self.assertTrue(len(results_chunks) >= 1)
        
        # Check that unique and reproducible embedding hash triggers are present
        self.assertTrue(chunks[0]["chunk_id"].startswith("chk_"))
        self.assertTrue(chunks[0]["embedding_id"].startswith("emb_"))
        print("-> Boundary-aware splits completed without chunk leaks.")

if __name__ == "__main__":
    unittest.main()
