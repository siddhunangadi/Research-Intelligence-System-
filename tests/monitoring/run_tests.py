"""
Phase 10 Observability Diagnostic and Verification Suite.
Runs monitoring tests using the built-in unittest library.
"""

import os
import sys
import time
import unittest
import tempfile
import logging

# Adjust python path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class TestRISPhase10(unittest.TestCase):
    def setUp(self):
        # Redirect DB connection for testing to SQLite memory DB to avoid dirtying postgres
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        
        # Reset DatabaseConnectionManager and PostgresMetadataStore singleton/cache if needed
        from database.connection import DatabaseConnectionManager
        DatabaseConnectionManager._instance = None

    def test_01_logger_setup(self):
        """
        Tests central logging configuration and verify log files are written.
        """
        print("\n[TEST] Running Logger setup tests...")
        from monitoring.logger import setup_logging
        
        # Test setting up a test logger with custom level
        logger = setup_logging("DEBUG")
        self.assertEqual(logger.level, logging.DEBUG)
        
        # Verify handler file exists
        log_file = os.path.join("logs", "app.log")
        self.assertTrue(os.path.exists(log_file))
        
        logger.info("Test log entry from diagnostic suite.")
        
        # Verify log entry is written
        with open(log_file, "r") as f:
            content = f.read()
            self.assertIn("Test log entry from diagnostic suite", content)
        print("-> Logging system configured and writing successfully.")

    def test_02_latency_tracker(self):
        """
        Tests latency tracking context manager and decorator patterns.
        """
        print("\n[TEST] Running Latency Tracker tests...")
        from monitoring.latency_tracker import LatencyTracker, profile_latency
        
        # Test context manager
        with LatencyTracker("test_cm") as tracker:
            time.sleep(0.05)  # 50 ms delay
            
        self.assertIsNotNone(tracker.elapsed_ms)
        self.assertGreaterEqual(tracker.elapsed_ms, 45.0)  # Should be close to 50ms
        
        # Test decorator
        @profile_latency("test_decorator")
        def dummy_function(delay):
            time.sleep(delay)
            return "done"
            
        res = dummy_function(0.03)
        self.assertEqual(res, "done")
        print("-> Latency profiling context manager and decorator validated successfully.")

    def test_03_query_tracker(self):
        """
        Tests recording and listing query telemetry data.
        """
        print("\n[TEST] Running Query Tracker tests...")
        from monitoring.query_tracker import QueryTracker
        from indexing.postgres_store import PostgresMetadataStore
        
        # Set up a clean store (SQLite memory engine)
        store = PostgresMetadataStore()
        tracker = QueryTracker(store=store)
        
        query_text = "What is the optimal batch size for training?"
        response_text = "The batch size is 32."
        latency = 120.5
        tokens = 250
        refused = False
        nli_ok = True
        
        # Record query log
        q_id = tracker.log_query(
            query_text=query_text,
            response_text=response_text,
            latency_ms=latency,
            token_count=tokens,
            refusal_triggered=refused,
            nli_passed=nli_ok
        )
        
        self.assertTrue(q_id.startswith("q_"))
        
        # Retrieve logs and assert matches
        logs = tracker.fetch_all_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["query_id"], q_id)
        self.assertEqual(logs[0]["query_text"], query_text)
        self.assertEqual(logs[0]["response_text"], response_text)
        self.assertEqual(logs[0]["latency_ms"], latency)
        self.assertEqual(logs[0]["token_count"], tokens)
        self.assertEqual(logs[0]["refusal_triggered"], refused)
        self.assertEqual(logs[0]["nli_passed"], nli_ok)
        print("-> Query telemetry logging and listing validated successfully.")

    def test_04_langsmith_tracker(self):
        """
        Tests LangSmith client setup failsafe fallback logic.
        """
        print("\n[TEST] Running LangSmith Tracker fallback tests...")
        from monitoring.langsmith_tracker import LangSmithTracker, traceable
        
        # In a test run without valid LANGCHAIN_API_KEY, tracker should disable tracing gracefully
        tracker = LangSmithTracker()
        
        # The traceable decorator should still be runnable (acting as no-op or wrapper)
        @traceable(name="test_trace")
        def dummy_pipeline_step():
            return 42
            
        val = dummy_pipeline_step()
        self.assertEqual(val, 42)
        print("-> LangSmith tracking failsafe fallback validated successfully.")

if __name__ == "__main__":
    unittest.main()
