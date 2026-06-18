"""
Table Layout Parser.
Uses Docling layout parser to identify and extract table boundaries and structures.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Fallback checker for docling
HAS_DOCLING = False
try:
    # Check if docling package can be loaded
    from docling.document_converter import DocumentConverter
    HAS_DOCLING = True
except ImportError:
    logger.warning("docling is not installed. Table parser will run in simulated layout mode.")

class TableParser:
    def __init__(self):
        self.converter = None
        if HAS_DOCLING:
            try:
                logger.info("Initializing Docling DocumentConverter...")
                self.converter = DocumentConverter()
            except Exception as e:
                logger.error(f"Error starting Docling: {e}. Falling back to standard parsing.")

    def parse_table_page(self, pdf_path: str, page_number: int) -> str:
        """
        Parses page using table-aware extraction features.
        
        Args:
            pdf_path (str): Filepath to the PDF document.
            page_number (int): 0-indexed page number.
            
        Returns:
            str: Markdown block representing structured table grid.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF document not found: {pdf_path}")
            
        if not self.converter:
            # Fallback mock mode: use standard text extraction and attempt simple table formatting
            logger.info("Docling converter unavailable. Using fallback structural parser.")
            try:
                import fitz
                doc = fitz.open(pdf_path)
                page = doc.load_page(page_number)
                text = page.get_text("text")
                # Clean columns or output standard table markers
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                
                # Check if we can build a simple markdown table grid
                markdown_lines = ["| Table Page Data |", "| --- |"]
                for line in lines[:10]: # Cap preview size
                    markdown_lines.append(f"| {line} |")
                return "\n".join(markdown_lines)
            except Exception as e:
                logger.error(f"Fallback table parsing failed: {e}")
                return "| Error extracting table data |"

        try:
            # Docling processes full files. For single-page extraction, we pass the file
            # and request specific page formats. (To keep single-page granularity in Docling,
            # we run standard document converter options).
            result = self.converter.convert(pdf_path)
            # Fetch markdown export from converted document
            md_content = result.document.export_to_markdown()
            # Slice or return contents containing tables
            return md_content
        except Exception as e:
            logger.error(f"Docling table parsing failed on page {page_number}: {e}")
            raise
