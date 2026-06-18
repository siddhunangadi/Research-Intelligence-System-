"""
Okapi BM25 Sparse Retriever.
Implements TF-IDF lexical retrieval scoring over document text chunks.
"""

import math
import logging
import re
from collections import Counter
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Counter] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, chunks: List[Dict]):
        """Fits BM25 index over a collection of chunk dictionaries."""
        self.corpus = chunks
        self.doc_count = len(chunks)
        if self.doc_count == 0:
            self.avg_doc_len = 0.0
            return

        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = Counter()

        for chunk in chunks:
            tokens = self._tokenize(chunk.get("content", ""))
            self.doc_lengths.append(len(tokens))
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_freqs.update(tf.keys())

        self.avg_doc_len = sum(self.doc_lengths) / self.doc_count
        logger.info(f"Fitted BM25 index over {self.doc_count} chunks. Avg len: {self.avg_doc_len:.2f}")

    def _calculate_idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return max(math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0), 1e-9)

    def retrieve(self, query: str, top_k: int = 20, paper_id: Optional[str] = None) -> List[Dict]:
        """Evaluates query relevance scores and returns top ranking chunks."""
        if self.doc_count == 0:
            return []
            
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self.corpus[:top_k]

        scored_results = []
        for idx, chunk in enumerate(self.corpus):
            if paper_id and chunk.get("paper_id") != paper_id:
                continue
                
            doc_len = self.doc_lengths[idx]
            tf_dict = self.term_freqs[idx]
            
            score = 0.0
            for token in query_tokens:
                if token in tf_dict:
                    tf = tf_dict[token]
                    idf = self._calculate_idf(token)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf * (tf * (self.k1 + 1)) / denom
                
            if score > 0.0:
                scored_results.append({**chunk, "bm25_score": score})

        return sorted(scored_results, key=lambda x: x["bm25_score"], reverse=True)[:top_k]
