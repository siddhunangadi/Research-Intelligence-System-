"""
RIS Query Telemetry Tracker.
Handles logging query execution parameters, token usage counts, refusals, NLI validation outputs, and elapsed time to metadata DB.
"""

import uuid
import logging
from typing import Optional
from indexing.postgres_store import PostgresMetadataStore

logger = logging.getLogger("query_tracker")

class QueryTracker:
    def __init__(self, store: Optional[PostgresMetadataStore] = None):
        self.store = store or PostgresMetadataStore()

    def log_query(
        self,
        query_text: str,
        response_text: str,
        latency_ms: float,
        token_count: int,
        refusal_triggered: bool,
        nli_passed: bool
    ) -> str:
        """
        Logs a search/generation query request details to the PostgreSQL/SQLite database.
        
        Returns:
            str: Generated unique query ID.
        """
        query_id = f"q_{uuid.uuid4().hex[:12]}"
        
        # Log to file-based logs first
        logger.info(
            f"Query Telemetry: ID={query_id} | Query='{query_text[:50]}...' | "
            f"Latency={latency_ms:.2f}ms | Tokens={token_count} | "
            f"Refused={refusal_triggered} | NLI={nli_passed}"
        )
        
        # Log to relational database store
        success = self.store.insert_query_log(
            query_id=query_id,
            query_text=query_text,
            response_text=response_text,
            latency_ms=latency_ms,
            token_count=token_count,
            refusal_triggered=refusal_triggered,
            nli_passed=nli_passed
        )
        
        if not success:
            logger.error(f"Failed to record query log {query_id} in relational database.")
            
        return query_id

    def fetch_all_logs(self) -> list[dict]:
        """
        Retrieves all query telemetry logs for monitoring dashboard visualizations.
        """
        return self.store.list_query_logs()
