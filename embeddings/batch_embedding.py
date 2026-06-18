"""
Batch Vectorization helper.
Slices large lists of document chunks into smaller batches to prevent GPU/CPU memory exhausts.
"""

import logging
from typing import Generator
from embeddings.embedder import ScientificEmbedder

logger = logging.getLogger(__name__)

class BatchEmbedder:
    def __init__(self, embedder: ScientificEmbedder = None, batch_size: int = 16):
        self.embedder = embedder or ScientificEmbedder()
        self.batch_size = batch_size

    def get_batches(self, items: list, batch_size: int) -> Generator[list, None, None]:
        """
        Yields successive batches of items.
        """
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def embed_chunks_batch(self, chunks: list[dict]) -> list[dict]:
        """
        Processes list of chunk dicts, generates vectors, and injects 'embedding' values.
        
        Args:
            chunks (list[dict]): List of chunk payload dictionaries.
            
        Returns:
            list[dict]: Chunks containing newly generated embeddings arrays.
        """
        if not chunks:
            return []
            
        logger.info(f"Slicing {len(chunks)} chunks into batches of size {self.batch_size} for vectorization.")
        
        text_contents = [c["content"] for c in chunks]
        all_embeddings = []
        
        for batch_texts in self.get_batches(text_contents, self.batch_size):
            logger.debug(f"Processing embedding inference batch of size {len(batch_texts)}")
            batch_vectors = self.embedder.embed_documents(batch_texts)
            all_embeddings.extend(batch_vectors)
            
        # Inject embeddings back to the original dictionary structures
        for idx, chunk in enumerate(chunks):
            chunk["embedding"] = all_embeddings[idx]
            
        logger.info(f"Batch vectorization complete. Generated {len(all_embeddings)} embeddings vectors.")
        return chunks
