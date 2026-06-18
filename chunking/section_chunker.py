"""
Section-Aware Chunking engine.
Splits text within section boundaries to prevent topic bleed 
and attaches structured metadata parameters to each block.
"""

import hashlib
import logging
from configs.config_manager import get_config
from retrieval.context_packer import count_tokens

logger = logging.getLogger(__name__)

class SectionChunker:
    def __init__(self):
        self.config = get_config()
        self.target_size = self.config.chunking.target_size
        self.overlap = self.config.chunking.overlap

    def split_text_recursive(self, text: str, max_chars: int, overlap_chars: int) -> list[tuple[str, int, int]]:
        """
        Splits text into chunks recursively based on characters.
        Returns a list of tuples containing (chunk_content, start_idx, end_idx).
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Simple chunk window slicing
            end = min(start + max_chars, text_len)
            
            # If we are not at the end of the text, try to find a natural line break or space to split on
            if end < text_len:
                for sep in ["\n\n", "\n", " ", ""]:
                    if sep == "":
                        break
                    last_sep_idx = text.rfind(sep, start, end)
                    # Don't split too early: ensure the chunk is at least 60% of the target window size
                    if last_sep_idx != -1 and last_sep_idx > start + (max_chars * 0.6):
                        end = last_sep_idx + len(sep)
                        break
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append((chunk_content, start, end))
                
            # Move window start index forward, accounting for overlap
            start = max(end - overlap_chars, start + 1)
            
        return chunks

    def chunk_document_sections(self, paper_id: str, sections_data: dict[str, str]) -> list[dict]:
        """
        Splits mapped document sections independently and attaches metadata payloads.
        
        Returns:
            list[dict]: Array of formatted chunks ready for storage.
        """
        all_chunks = []
        
        # Estimate character budgets mapping close to token parameters (approx 4 chars per token)
        max_chars = self.target_size * 4
        overlap_chars = self.overlap * 4
        
        for section, text in sections_data.items():
            if not text.strip():
                continue
                
            logger.debug(f"Chunking section '{section}' (character count: {len(text)})")
            raw_splits = self.split_text_recursive(text, max_chars, overlap_chars)
            
            for index, (content, start_idx, end_idx) in enumerate(raw_splits, start=1):
                token_count = count_tokens(content)
                
                # Generate unique, reproducible UUID hash based on parent paper and chunk parameters
                hash_source = f"{paper_id}:{section}:{index}:{content[:100]}"
                uuid_hash = hashlib.md5(hash_source.encode("utf-8")).hexdigest()
                
                chunk_id = f"chk_{uuid_hash}"
                embedding_id = f"emb_{uuid_hash}"
                
                chunk_item = {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "section": section,
                    # Fallback page assignment (ingestion manager resolves physical PDF page coordinates)
                    "page_number": 1,
                    "token_count": token_count,
                    "embedding_id": embedding_id,
                    "content": content,
                    "start_char": start_idx,
                    "end_char": end_idx
                }
                all_chunks.append(chunk_item)
                
        logger.info(f"Generated {len(all_chunks)} chunks for paper '{paper_id}'.")
        return all_chunks
