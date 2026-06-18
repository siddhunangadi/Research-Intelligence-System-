"""
PDF Text Parser.
Extracts raw text pages and formats them into structured Markdown blocks.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Fallback checker for pymupdf4llm
HAS_MUPDF4LLM = False
try:
    import pymupdf4llm
    import fitz
    HAS_MUPDF4LLM = True
except ImportError:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("fitz (PyMuPDF) is not installed. PDF parsing will run in simulation mode.")

class PDFParser:
    def __init__(self):
        self.enabled = "fitz" in globals() or HAS_MUPDF4LLM

    def parse_page_to_markdown(self, pdf_path: str, page_number: int) -> str:
        """
        Parses a single page from a PDF and returns the text as markdown.
        
        Args:
            pdf_path (str): Filepath to the PDF document.
            page_number (int): 0-indexed page number.
            
        Returns:
            str: Markdown formatted text.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF document not found: {pdf_path}")
            
        if not self.enabled:
            logger.warning("Parsing in simulation mode. Returning empty page text.")
            return f"# Page {page_number + 1}\nSimulation text for page extraction."

        try:
            if HAS_MUPDF4LLM:
                # pymupdf4llm extracts layout-aware markdown directly
                # To parse a single page, we slice the document
                doc = fitz.open(pdf_path)
                if page_number < 0 or page_number >= len(doc):
                    raise IndexError(f"Page index {page_number} out of range for doc length {len(doc)}.")
                
                # pymupdf4llm.to_markdown can take page list parameter
                md_text = pymupdf4llm.to_markdown(doc, pages=[page_number])
                return md_text.strip()
            else:
                # Fallback to standard PyMuPDF
                doc = fitz.open(pdf_path)
                if page_number < 0 or page_number >= len(doc):
                    raise IndexError(f"Page index {page_number} out of range for doc length {len(doc)}.")
                
                page = doc.load_page(page_number)
                text = page.get_text("text")
                # Structure simple headings or paragraphs
                lines = text.split("\n")
                structured_lines = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    # Check if line looks like a header (short, capitalized, no ending period)
                    if len(stripped) < 60 and stripped.isupper() and not stripped.endswith("."):
                        structured_lines.append(f"\n## {stripped}\n")
                    else:
                        structured_lines.append(stripped)
                return "\n".join(structured_lines).strip()
        except Exception as e:
            logger.error(f"Failed to parse page {page_number} of PDF '{pdf_path}': {e}")
            raise
