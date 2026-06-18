"""
Page Parser Router.
Inspects physical layout properties and character frequencies of PDF pages 
to dynamically assign standard text, tabular, or mathematical parsers.
"""

import logging
import re
from ingestion.pdf_parser import PDFParser
from ingestion.table_parser import TableParser
from ingestion.equation_parser import EquationParser

logger = logging.getLogger(__name__)

# Fallback checking
HAS_FITZ = False
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    pass

class ParserRouter:
    def __init__(self):
        self.text_parser = PDFParser()
        self.table_parser = TableParser()
        self.equation_parser = EquationParser()

    def analyze_page_layout(self, pdf_path: str, page_number: int) -> dict:
        """
        Extracts structural layout metrics from a PDF page to determine routing.
        """
        if not HAS_FITZ:
            return {"pipe_ratio": 0.0, "equation_hits": 0, "char_count": 100}
            
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_number)
            text = page.get_text("text")
            drawings = page.get_drawings()
            
            # 1. Pipe Ratio: Checks for vertical cell boundaries or grid line paths
            vertical_lines = sum(
                1 for d in drawings 
                if any(p[0] == "l" and abs(p[1].x - p[2].x) < 1.0 for p in d.get("items", []))
            )
            char_count = len(text)
            # Combine '|' counts and vertical line boundaries
            pipe_ratio = (text.count("|") / max(char_count, 1)) + (vertical_lines / max(char_count, 1))
            
            # 2. Equation Hits: Searches for LaTeX expressions or math symbols
            math_patterns = [
                r"\$\$.*?\$\$",
                r"\$.*?\$",
                r"\\begin\{equation\}",
                r"\\int|\\sum|\\alpha|\\beta|\\gamma|\\theta|\\infty|\\partial",
                r"[\+\-\*=/\^]\s*[xXyYzZ]\b"
            ]
            equation_hits = sum(len(re.findall(pat, text)) for pat in math_patterns)
            
            return {
                "pipe_ratio": pipe_ratio,
                "equation_hits": equation_hits,
                "char_count": char_count
            }
        except Exception as e:
            logger.error(f"Error analyzing page {page_number}: {e}")
            return {"pipe_ratio": 0.0, "equation_hits": 0, "char_count": 0}

    def route_page_parser(self, pdf_path: str, page_number: int) -> str:
        """
        Resolves parser assignment based on page metrics.
        
        Rules:
            pipe_ratio > 0.15   -> "docling" (Table extraction)
            equation_hits > 3   -> "nougat"  (Equation translation)
            Otherwise           -> "pymupdf" (Text extraction)
        """
        metrics = self.analyze_page_layout(pdf_path, page_number)
        
        pipe_ratio = metrics["pipe_ratio"]
        equation_hits = metrics["equation_hits"]
        
        if pipe_ratio > 0.15:
            logger.info(f"Page {page_number} routed to TableParser (pipe_ratio: {pipe_ratio:.4f})")
            return "table"
        elif equation_hits > 3:
            logger.info(f"Page {page_number} routed to EquationParser (equation_hits: {equation_hits})")
            return "equation"
        else:
            logger.info(f"Page {page_number} routed to StandardPDFParser")
            return "text"

    def parse_page(self, pdf_path: str, page_number: int) -> str:
        """
        Routes and parses a page.
        """
        parser_type = self.route_page_parser(pdf_path, page_number)
        
        if parser_type == "table":
            return self.table_parser.parse_table_page(pdf_path, page_number)
        elif parser_type == "equation":
            return self.equation_parser.parse_equations_page(pdf_path, page_number)
        else:
            return self.text_parser.parse_page_to_markdown(pdf_path, page_number)
