"""
Metadata Filter Builder.
Constructs syntax-compliant filter configurations for ChromaDB and PostgreSQL query backends.
"""

import logging
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

class MetadataFilterBuilder:
    def build_chroma_filter(
        self, 
        paper_id: Optional[str] = None, 
        sections: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """Builds a syntax-compliant filter dictionary for ChromaDB queries."""
        filters = []
        if paper_id:
            filters.append({"paper_id": paper_id})
        if sections:
            filters.append({"section": sections[0]} if len(sections) == 1 else {"section": {"$in": list(sections)}})

        if not filters:
            return None
        return filters[0] if len(filters) == 1 else {"$and": filters}

    def build_postgres_clause(
        self, 
        paper_id: Optional[str] = None, 
        sections: Optional[List[str]] = None
    ) -> Tuple[str, Dict]:
        """Synthesizes a WHERE clause and parameter dictionary for SQL queries."""
        clauses = []
        params = {}
        
        if paper_id:
            clauses.append("paper_id = :paper_id")
            params["paper_id"] = paper_id
            
        if sections:
            placeholders = []
            for idx, sec in enumerate(sections):
                name = f"sec_{idx}"
                placeholders.append(f":{name}")
                params[name] = sec
            clauses.append(f"section IN ({', '.join(placeholders)})")

        if not clauses:
            return "", {}
            
        return "WHERE " + " AND ".join(clauses), params
