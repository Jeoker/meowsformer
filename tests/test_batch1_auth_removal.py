"""
Tests for Batch 1 — Auth Removal & Server Startup Unblocking
=============================================================
Verifies that all auth-related imports, config fields, and routes have been
removed, and that the server can start cleanly without ModuleNotFoundError.

Covers:
  1. Import integrity (main, endpoints, app.api package, FastAPI instance)
  2. Route registration (expected routes present, /auth/* absent)
  3. Health endpoint (GET /health → 200)
  4. Config (no JWT fields in Settings)
  5. Source-level auth residual checks
  6. Vite proxy config (no /auth proxy)
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import sys
import unittest
from unittest.mock import MagicMock

# ── Mock chromadb before importing any app modules ─────────────────────
# Using setdefault (not direct assignment) to avoid clobbering a real
# chromadb if the test runner happens to have it already loaded.
mock_chromadb = MagicMock()
sys.modules.setdefault("chromadb", mock_chromadb)
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())

from fastapi.testclient import TestClient


# ════════════════════════════════════════════════════════════════════════
#  1. Import Integrity Tests
# ════════════════════════════════════════════════════════════════════════


class TestImportIntegrity(unittest.TestCase):
    """Modules modified in Batch 1 must import without errors."""

    def test_main_module_imports(self) -> None:
        """main.py should import successfully (no ModuleNotFoundError)."""
        import main  # noqa: F811
        self.assertIsNotNone(main)

    def test_endpoints_module_imports(self) -> None:
        """app.api.endpoints should import without auth dependency."""
        import app.api.endpoints as ep
        self.assertIsNotNone(ep)
        self.assertTrue(hasattr(ep, "router"))

    def test_api_package_imports(self) -> None:
        """app.api package (empty __init__.py) should import cleanly."""
        import app.api
        self.assertIsNotNone(app.api)

    def test_api_init_does_not_reexport_endpoints(self) -> None:
        """app.api.__init__ should NOT auto-import endpoints or define api_router."""
        import app.api
        self.assertFalse(
            hasattr(app.api, "api_router"),
            "app.api should not export api_router (old dead code removed)",
        )

    def test_fastapi_app_instance_created(self) -> None:
        """main.app should be a valid FastAPI instance."""
        from main import app as fastapi_app
        from fastapi import FastAPI

        self.assertIsInstance(fastapi_app, FastAPI)

    def test_config_module_imports(self) -> None:
        """app.core.config should import without errors."""
        from app.core.config import Settings, settings
        self.assertIsNotNone(Settings)
        self.assertIsNotNone(settings)


# ════════════════════════════════════════════════════════════════════════
#  2. Server Startup / Route Registration Tests
# ════════════════════════════════════════════════════════════════════════


class TestRouteRegistration(unittest.TestCase):
    """Verify the FastAPI app has the correct routes after Batch 1."""

    @classmethod
    def setUpClass(cls) -> None:
        from main import app
        # FastAPI 0.138+ may keep included routers as lazy _IncludedRouter
        # entries without a ``path`` attribute. Walk their original routers
        # so HTTP and WebSocket routes are covered across FastAPI versions.
        cls.route_paths = []
        for route in app.routes:
            if hasattr(route, "path"):
                cls.route_paths.append(route.path)
                continue
            original_router = getattr(route, "original_router", None)
            include_context = getattr(route, "include_context", None)
            if original_router is None or include_context is None:
                continue
            prefix = include_context.prefix
            cls.route_paths.extend(
                f"{prefix}{child.path}" for child in original_router.routes
            )

    def test_health_route_registered(self) -> None:
        self.assertIn("/health", self.route_paths)

    def test_api_translate_route_registered(self) -> None:
        self.assertIn("/api/translate", self.route_paths)

    def test_api_v1_translate_route_registered(self) -> None:
        self.assertIn("/api/v1/translate", self.route_paths)

    def test_ws_translate_route_registered(self) -> None:
        self.assertIn("/ws/translate", self.route_paths)

    def test_no_auth_routes(self) -> None:
        """No route path should start with /auth."""
        auth_routes = [p for p in self.route_paths if "/auth" in p]
        self.assertEqual(
            auth_routes,
            [],
            f"Found unexpected auth routes: {auth_routes}",
        )


# ════════════════════════════════════════════════════════════════════════
#  3. Health Endpoint Tests
# ════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint(unittest.TestCase):
    """GET /health should return 200 with correct JSON."""

    @classmethod
    def setUpClass(cls) -> None:
        from main import app
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_health_returns_200(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_health_response_body(self) -> None:
        response = self.client.get("/health")
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["app"], "MeowTranslator")


# ════════════════════════════════════════════════════════════════════════
#  4. Configuration Tests
# ════════════════════════════════════════════════════════════════════════


class TestConfigNoJWT(unittest.TestCase):
    """Settings should not contain JWT-related fields."""

    JWT_FIELDS = ("JWT_SECRET_KEY", "JWT_ALGORITHM", "JWT_ACCESS_TOKEN_EXPIRE_MINUTES")

    def test_settings_class_no_jwt_fields(self) -> None:
        from app.core.config import Settings
        model_fields = Settings.model_fields
        for field_name in self.JWT_FIELDS:
            self.assertNotIn(
                field_name,
                model_fields,
                f"Settings still contains JWT field: {field_name}",
            )

    def test_settings_instance_no_jwt_attrs(self) -> None:
        from app.core.config import settings
        for field_name in self.JWT_FIELDS:
            self.assertFalse(
                hasattr(settings, field_name),
                f"settings instance still has JWT attribute: {field_name}",
            )

    def test_core_settings_fields_present(self) -> None:
        """Essential non-auth fields should still exist."""
        from app.core.config import Settings
        model_fields = Settings.model_fields
        for required in ("OPENAI_API_KEY", "CHROMA_DB_PATH", "DEBUG_MODE"):
            self.assertIn(required, model_fields, f"Missing required field: {required}")


# ════════════════════════════════════════════════════════════════════════
#  5. Source-Level Auth Residual Checks
# ════════════════════════════════════════════════════════════════════════


class TestNoAuthResiduals(unittest.TestCase):
    """Source code should not contain auth-related imports or references."""

    @staticmethod
    def _get_source(module) -> str:
        return inspect.getsource(module)

    @staticmethod
    def _get_import_names(source: str) -> list[str]:
        """Parse all imported names from source using AST (ignores comments)."""
        tree = ast.parse(source)
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    names.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.append(alias.name)
        return names

    def test_main_no_auth_imports(self) -> None:
        """main.py should have no import statements referencing 'auth'."""
        import main
        source = self._get_source(main)
        import_names = self._get_import_names(source)
        auth_imports = [n for n in import_names if "auth" in n.lower()]
        self.assertEqual(
            auth_imports,
            [],
            f"main.py still imports auth-related modules: {auth_imports}",
        )

    def test_endpoints_no_auth_imports(self) -> None:
        """endpoints.py should have no import statements referencing 'auth'."""
        import app.api.endpoints as ep
        source = self._get_source(ep)
        import_names = self._get_import_names(source)
        auth_imports = [n for n in import_names if "auth" in n.lower()]
        self.assertEqual(
            auth_imports,
            [],
            f"endpoints.py still imports auth-related modules: {auth_imports}",
        )

    def test_main_no_create_tables_call(self) -> None:
        """main.py should not call create_tables()."""
        import main
        source = self._get_source(main)
        self.assertNotIn(
            "create_tables",
            source,
            "main.py still references create_tables()",
        )

    def test_main_no_auth_router_inclusion(self) -> None:
        """main.py should not register auth_router."""
        import main
        source = self._get_source(main)
        self.assertNotIn(
            "auth_router",
            source,
            "main.py still references auth_router",
        )

    def test_endpoints_no_get_optional_user(self) -> None:
        """endpoints.py should not reference get_optional_user."""
        import app.api.endpoints as ep
        source = self._get_source(ep)
        self.assertNotIn(
            "get_optional_user",
            source,
            "endpoints.py still references get_optional_user",
        )


# ════════════════════════════════════════════════════════════════════════
#  6. Vite Proxy Config Tests
# ════════════════════════════════════════════════════════════════════════


class TestViteProxyConfig(unittest.TestCase):
    """vite.config.ts should not proxy /auth routes."""

    _PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    VITE_CONFIG_PATH = _PROJECT_ROOT / "src" / "ui" / "vite.config.ts"

    @classmethod
    def setUpClass(cls) -> None:
        cls.vite_content = cls.VITE_CONFIG_PATH.read_text()

    def test_no_auth_proxy(self) -> None:
        self.assertNotIn(
            '"/auth"',
            self.vite_content,
            "vite.config.ts still contains /auth proxy configuration",
        )

    def test_api_proxy_present(self) -> None:
        """API proxy should still be configured."""
        self.assertIn('"/api"', self.vite_content)

    def test_ws_proxy_present(self) -> None:
        """WebSocket proxy should still be configured."""
        self.assertIn('"/ws"', self.vite_content)


if __name__ == "__main__":
    unittest.main()
