"""
Ingestion Pipeline Coordinator.
Orchestrates PDF page parsing, records page offset coordinates, 
groups text by canonical headers, splits chunks, and maps indices back to physical pages.
"""

import logging
import os
import hashlib
from ingestion.parser_router import ParserRouter
from ingestion.section_extractor import SectionExtractor
from chunking.section_chunker import SectionChunker

logger = logging.getLogger(__name__)

# Fallback checker
HAS_FITZ = False
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    pass

class IngestionPipeline:
    def __init__(self):
        self.router = ParserRouter()
        self.extractor = SectionExtractor()
        self.chunker = SectionChunker()

    def generate_paper_id(self, pdf_path: str) -> str:
        """
        Creates a short, reproducible hash identifier for a document based on its basename.
        """
        basename = os.path.basename(pdf_path)
        # Clean naming tokens
        clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", basename)
        hash_val = hashlib.md5(basename.encode("utf-8")).hexdigest()[:8]
        return f"paper_{clean_name[:15]}_{hash_val}"

    def ingest_document(self, pdf_path: str, title: str = None) -> tuple[dict, list[dict]]:
        """
        Ingests a research paper PDF, segments it into chunks, and maps structural page fields.
        
        Args:
            pdf_path (str): Filepath to the scientific paper PDF.
            title (str, optional): Title override. Defaults to file basename.
            
        Returns:
            tuple[dict, list[dict]]: Paper metadata registry dictionary 
                                     and a list of chunk data dictionaries.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF document not found: {pdf_path}")
            
        paper_id = self.generate_paper_id(pdf_path)
        final_title = title or os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Load PDF page count
        page_count = 1
        if HAS_FITZ:
            try:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
            except Exception as e:
                logger.error(f"Error loading document page count: {e}")
                
        logger.info(f"Starting ingestion for paper: '{paper_id}' (Title: {final_title}, Pages: {page_count})")
        
        # 1. Parse page by page, tracking character offsets to resolve physical page mapping
        page_offsets = []
        full_document_text = ""
        
        for p_idx in range(page_count):
            page_text = self.router.parse_page(pdf_path, p_idx)
            
            start_offset = len(full_document_text)
            full_document_text += page_text + "\n\n"
            end_offset = len(full_document_text)
            
            # Map coordinates: page index -> character boundaries
            page_offsets.append({
                "page_number": p_idx + 1,
                "start": start_offset,
                "end": end_offset
            })
            
        logger.debug(f"Parsing complete. Character boundary mapping: {page_offsets}")
        
        # 2. Extract canonical scientific sections
        sections_data = self.extractor.split_by_sections(full_document_text)
        
        # 3. Generate section-aware chunks
        chunks = self.chunker.chunk_document_sections(paper_id, sections_data)
        
        # 4. Map chunk character coordinates back to physical page numbers
        # Find which page offset boundary a chunk's start offset belongs to
        for chunk in chunks:
            # We locate character index inside the full parsed text
            # We assume section offsets correspond to indices within full_document_text
            # Let's adjust chunk start index relative to full text
            # Our chunker splits text segment by segment. Let's align chunk offsets:
            chunk_start = chunk.get("start_char", 0)
            section_name = chunk.get("section", "")
            
            # Locate section boundary in full_document_text to map absolute index coordinates
            # A simple lookup: search section substring
            section_start_in_doc = full_document_text.find(sections_data.get(section_name, ""))
            if section_start_in_doc == -1:
                section_start_in_doc = 0
                
            absolute_chunk_start = section_start_in_doc + chunk_start
            
            # Map absolute offset to matching page numbers
            mapped_page = 1
            for offset_data in page_offsets:
                if offset_data["start"] <= absolute_chunk_start <= offset_data["end"]:
                    mapped_page = offset_data["page_number"]
                    break
                    
            chunk["page_number"] = mapped_page
            # Clean temporary offset metrics from final chunk schema
            if "start_char" in chunk:
                del chunk["start_char"]
            if "end_char" in chunk:
                del chunk["end_char"]

        # Build paper metadata
        paper_metadata = {
            "paper_id": paper_id,
            "title": final_title,
            "authors": None,  # Can be augmented by LLM parsing later
            "year": None,
            "venue": None
        }
        
        logger.info(f"Ingestion pipeline completed for '{paper_id}'. Generated {len(chunks)} mapped chunks.")
        return paper_metadata, chunks

# Import regex wrapper helper for clean character filters
import re
