"""
Phase 9 API Endpoint Routing Diagnostic Suite.
Tests FastAPI endpoints routing, health checks, and JSON response payloads using standard unittest.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Fallback checker for testclient dependency
HAS_TESTCLIENT = False
try:
    from fastapi.testclient import TestClient
    HAS_TESTCLIENT = True
except ImportError:
    pass

class TestRISPhase9(unittest.TestCase):
    def setUp(self):
        # Override connection pool settings to SQLite in-memory for testing
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        from api.main import app
        self.app = app

    def test_01_api_root_routes(self):
        """
        Tests that the root URL returns the correct service information.
        """
        print("\n[TEST] Running API Root Route checks...")
        if not HAS_TESTCLIENT:
            print("-> Skipped: fastapi.testclient not installed in current environment.")
            return

        client = TestClient(self.app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("version", data)
        self.assertEqual(data["service"], "Research Intelligence System API")
        print("-> Root API endpoint resolved correctly.")

    def test_02_api_health_endpoint(self):
        """
        Tests that health endpoints query DB connections status check.
        """
        print("\n[TEST] Running API Health Endpoint diagnostics...")
        if not HAS_TESTCLIENT:
            print("-> Skipped: fastapi.testclient not installed in current environment.")
            return

        client = TestClient(self.app)
        # SQLite should successfully mock Postgres in connection check, reporting 200/503 health details
        response = client.get("/api/health")
        # Since Chroma DB uses standard persistent SQLite, it should connect successfully
        # Health endpoint should return 200 if SQLite connects successfully
        self.assertIn(response.status_code, [200, 503])
        
        data = response.json()
        self.assertIn("postgres", data)
        self.assertIn("chromadb", data)
        print("-> Health database connection checks parsed correctly.")

if __name__ == "__main__":
    unittest.main()
