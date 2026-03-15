"""
Tests for API Client Factory — app/core/api_client.py
=======================================================
Covers:
  1. get_openai_client() — openai provider: uses OPENAI_API_KEY, no base_url
  2. get_openai_client() — ai_builders provider: uses AI_BUILDER_TOKEN + base_url
  3. get_instructor_client() — wraps get_openai_client() via instructor.from_openai()
  4. LLM_MODEL is read from settings, not hard-coded "gpt-4o"
  5. config.py defaults: API_PROVIDER, LLM_MODEL, AI_BUILDER_BASE_URL
  6. transcription_service._get_client() lazy-load caching
  7. llm_service._get_client() lazy-load caching

No real API calls are made — all external clients are fully mocked.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

# ── Mock chromadb before any app import (project convention) ─────────────────
# Use the same pattern as test_batch1_auth_removal.py: setdefault keeps any
# mock already installed by another module that loads earlier alphabetically.
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())

# ── Test parameters ───────────────────────────────────────────────────────────
from tests.shared_params import (  # noqa: E402
    AI_BUILDER_BASE_URL,
    MODEL_OPENAI_DEFAULT,
    PROVIDER_AI_BUILDERS,
    PROVIDER_OPENAI,
)

_DUMMY_SK_KEY = "sk-key"  # generic dummy used when key value is not under test

# ── Subjects under test ───────────────────────────────────────────────────────
from app.core.api_client import get_instructor_client, get_openai_client  # noqa: E402
from app.core.config import Settings  # noqa: E402
import app.services.transcription_service as _ts  # noqa: E402
import app.services.llm_service as _llm  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  1. get_openai_client() — openai provider
# ══════════════════════════════════════════════════════════════════════════════


class TestGetOpenAIClientOpenAIProvider(unittest.TestCase):
    """get_openai_client() with API_PROVIDER='openai'."""

    def _call_with_mock(self, api_key: str = "sk-test") -> MagicMock:
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_settings.OPENAI_API_KEY = api_key
            get_openai_client()
            return MockOpenAI

    def test_openai_provider_calls_openai_with_api_key(self) -> None:
        """OpenAI client must be constructed with the OPENAI_API_KEY."""
        MockOpenAI = self._call_with_mock("sk-my-openai-key")
        MockOpenAI.assert_called_once_with(api_key="sk-my-openai-key")

    def test_openai_provider_no_base_url_kwarg(self) -> None:
        """base_url must NOT be passed when provider is openai."""
        MockOpenAI = self._call_with_mock()
        kwargs = MockOpenAI.call_args.kwargs
        self.assertNotIn("base_url", kwargs)

    def test_openai_provider_returns_openai_instance(self) -> None:
        """Return value must be the object returned by OpenAI()."""
        fake_instance = MagicMock(name="fake_openai_client")
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI", return_value=fake_instance),
        ):
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_settings.OPENAI_API_KEY = _DUMMY_SK_KEY
            result = get_openai_client()
        self.assertIs(result, fake_instance)

    def test_factory_creates_fresh_instance_each_call(self) -> None:
        """get_openai_client() has no internal cache — each call returns new object."""
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_settings.OPENAI_API_KEY = _DUMMY_SK_KEY
            MockOpenAI.side_effect = lambda **_kw: MagicMock()

            c1 = get_openai_client()
            c2 = get_openai_client()

        self.assertIsNot(c1, c2)
        self.assertEqual(MockOpenAI.call_count, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  2. get_openai_client() — ai_builders provider
# ══════════════════════════════════════════════════════════════════════════════


class TestGetOpenAIClientAiBuilders(unittest.TestCase):
    """get_openai_client() with API_PROVIDER='ai_builders'."""

    def _call_with_builder_settings(
        self,
        token: str = "sk-builder",
        base_url: str = AI_BUILDER_BASE_URL,
    ) -> MagicMock:
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            mock_settings.API_PROVIDER = PROVIDER_AI_BUILDERS
            mock_settings.AI_BUILDER_TOKEN = token
            mock_settings.AI_BUILDER_BASE_URL = base_url
            get_openai_client()
            return MockOpenAI

    def test_ai_builders_provider_uses_builder_token(self) -> None:
        """api_key must be AI_BUILDER_TOKEN, not OPENAI_API_KEY."""
        MockOpenAI = self._call_with_builder_settings(token="sk-my-builder-token")
        kwargs = MockOpenAI.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "sk-my-builder-token")

    def test_ai_builders_provider_includes_base_url(self) -> None:
        """base_url must be set to AI_BUILDER_BASE_URL."""
        url = AI_BUILDER_BASE_URL
        MockOpenAI = self._call_with_builder_settings(base_url=url)
        kwargs = MockOpenAI.call_args.kwargs
        self.assertIn("base_url", kwargs)
        self.assertEqual(kwargs["base_url"], url)

    def test_ai_builders_provider_does_not_use_openai_key(self) -> None:
        """OPENAI_API_KEY must not be passed for ai_builders provider."""
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            mock_settings.API_PROVIDER = PROVIDER_AI_BUILDERS
            mock_settings.AI_BUILDER_TOKEN = "sk-builder-token"
            mock_settings.AI_BUILDER_BASE_URL = AI_BUILDER_BASE_URL
            mock_settings.OPENAI_API_KEY = "sk-should-not-be-used"
            get_openai_client()

        kwargs = MockOpenAI.call_args.kwargs
        self.assertNotEqual(kwargs.get("api_key"), "sk-should-not-be-used")
        self.assertEqual(kwargs["api_key"], "sk-builder-token")

    def test_switching_providers_changes_call_signature(self) -> None:
        """Switching API_PROVIDER between calls must yield different constructor args."""
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            MockOpenAI.side_effect = lambda **kw: MagicMock(**kw)

            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_settings.OPENAI_API_KEY = "sk-openai"
            get_openai_client()

            mock_settings.API_PROVIDER = PROVIDER_AI_BUILDERS
            mock_settings.AI_BUILDER_TOKEN = "sk-builder"
            mock_settings.AI_BUILDER_BASE_URL = AI_BUILDER_BASE_URL
            get_openai_client()

        self.assertEqual(MockOpenAI.call_count, 2)
        first_kwargs = MockOpenAI.call_args_list[0].kwargs
        second_kwargs = MockOpenAI.call_args_list[1].kwargs
        self.assertNotEqual(first_kwargs.get("api_key"), second_kwargs.get("api_key"))


# ══════════════════════════════════════════════════════════════════════════════
#  3. get_instructor_client() — instructor wrapping
# ══════════════════════════════════════════════════════════════════════════════


class TestGetInstructorClient(unittest.TestCase):
    """get_instructor_client() must wrap get_openai_client() via instructor."""

    def test_delegates_to_get_openai_client(self) -> None:
        """instructor factory must call get_openai_client() exactly once."""
        mock_openai = MagicMock()
        with (
            patch("app.core.api_client.get_openai_client", return_value=mock_openai) as mock_factory,
            patch("app.core.api_client.instructor.from_openai", return_value=MagicMock()),
        ):
            get_instructor_client()

        mock_factory.assert_called_once()

    def test_passes_openai_client_to_instructor(self) -> None:
        """instructor.from_openai() must receive the client from get_openai_client()."""
        mock_openai = MagicMock()
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.get_openai_client", return_value=mock_openai),
            patch("app.core.api_client.instructor.from_openai") as mock_from_openai,
        ):
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_from_openai.return_value = MagicMock()
            get_instructor_client()

        mock_from_openai.assert_called_once_with(mock_openai)

    def test_returns_instructor_result(self) -> None:
        """Return value must be what instructor.from_openai() returned."""
        mock_instructor_client = MagicMock(name="instructor_client")
        with (
            patch("app.core.api_client.get_openai_client", return_value=MagicMock()),
            patch("app.core.api_client.instructor.from_openai", return_value=mock_instructor_client),
        ):
            result = get_instructor_client()

        self.assertIs(result, mock_instructor_client)

    def test_returns_real_instructor_instance(self) -> None:
        """Integration: calling with a real OpenAI client returns an Instructor."""
        import instructor
        from openai import OpenAI

        real_client = OpenAI(api_key="sk-fake-for-test")
        with patch("app.core.api_client.get_openai_client", return_value=real_client):
            result = get_instructor_client()

        self.assertIsInstance(result, instructor.Instructor)

    def test_factory_creates_fresh_instance_each_call(self) -> None:
        """get_instructor_client() has no internal cache — fresh result each call."""
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.get_openai_client", return_value=MagicMock()),
            patch("app.core.api_client.instructor.from_openai") as mock_from_openai,
        ):
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_from_openai.side_effect = lambda *a, **k: MagicMock()
            c1 = get_instructor_client()
            c2 = get_instructor_client()

        self.assertIsNot(c1, c2)
        self.assertEqual(mock_from_openai.call_count, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  4. LLM_MODEL propagation — settings.LLM_MODEL used, not "gpt-4o"
# ══════════════════════════════════════════════════════════════════════════════


class TestLLMModelPropagation(unittest.TestCase):
    """Services must pass settings.LLM_MODEL to LLM calls, not the hard-coded 'gpt-4o'."""

    def setUp(self) -> None:
        _llm._client = None

    def tearDown(self) -> None:
        _llm._client = None

    def test_llm_service_uses_settings_llm_model(self) -> None:
        from app.schemas.translation import CatTranslationResponse

        mock_response = MagicMock(spec=CatTranslationResponse)
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        with (
            patch("app.services.llm_service.get_instructor_client", return_value=mock_instructor),
            patch("app.services.llm_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = "custom-model-llm"
            _llm.analyze_intention("hello", {}, "")

        kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "custom-model-llm")

    def test_llm_service_model_not_hardcoded_gpt4o(self) -> None:
        """Changing LLM_MODEL must be reflected in the API call."""
        from app.schemas.translation import CatTranslationResponse

        mock_response = MagicMock(spec=CatTranslationResponse)
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        with (
            patch("app.services.llm_service.get_instructor_client", return_value=mock_instructor),
            patch("app.services.llm_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = "deepseek-chat"
            _llm.analyze_intention("test", {}, "")

        kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        self.assertNotEqual(kwargs["model"], MODEL_OPENAI_DEFAULT)
        self.assertEqual(kwargs["model"], "deepseek-chat")

    def test_openai_provider_llm_model_default_is_gpt4o(self) -> None:
        """Default LLM_MODEL 'gpt-4o' is used when no override is set."""
        from app.schemas.translation import CatTranslationResponse

        mock_response = MagicMock(spec=CatTranslationResponse)
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        # Use the real settings object (not mocked) to verify the default
        with (
            patch("app.services.llm_service.get_instructor_client", return_value=mock_instructor),
        ):
            from app.core.config import settings as real_settings
            _llm.analyze_intention("test", {}, "")

        kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], real_settings.LLM_MODEL)

    def test_ai_builders_provider_llm_model_propagated(self) -> None:
        """LLM_MODEL must work for ai_builders models as well."""
        from app.schemas.translation import CatTranslationResponse

        mock_response = MagicMock(spec=CatTranslationResponse)
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        with (
            patch("app.services.llm_service.get_instructor_client", return_value=mock_instructor),
            patch("app.services.llm_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = "gemini-2.5-pro"
            _llm.analyze_intention("test", {}, "")

        kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-pro")


# ══════════════════════════════════════════════════════════════════════════════
#  5. config.py default field values
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigDefaults(unittest.TestCase):
    """Settings schema must declare correct defaults for the new fields."""

    @staticmethod
    def _default(field: str):
        return Settings.model_fields[field].default

    def test_api_provider_default_is_openai(self) -> None:
        self.assertEqual(self._default("API_PROVIDER"), PROVIDER_OPENAI)

    def test_llm_model_default_is_gpt4o(self) -> None:
        """With API_PROVIDER=openai, LLM_MODEL resolves to gpt-4o."""
        s = Settings(API_PROVIDER=PROVIDER_OPENAI)
        self.assertEqual(s.LLM_MODEL, MODEL_OPENAI_DEFAULT)

    def test_ai_builder_base_url_default_not_empty(self) -> None:
        url: str = self._default("AI_BUILDER_BASE_URL")
        self.assertIsInstance(url, str)
        self.assertNotEqual(url, "")
        self.assertIn("ai-builders", url)

    def test_ai_builder_base_url_default_value(self) -> None:
        self.assertEqual(
            self._default("AI_BUILDER_BASE_URL"),
            AI_BUILDER_BASE_URL,
        )

    def test_ai_builder_token_default_is_empty(self) -> None:
        self.assertEqual(self._default("AI_BUILDER_TOKEN"), "")

    def test_api_provider_field_exists(self) -> None:
        self.assertIn("API_PROVIDER", Settings.model_fields)

    def test_llm_model_field_exists(self) -> None:
        self.assertIn("LLM_MODEL", Settings.model_fields)

    def test_api_provider_rejects_invalid_literal(self) -> None:
        """Pydantic Literal validation must reject unknown provider values."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            Settings(API_PROVIDER="invalid_provider")  # type: ignore[arg-type]

    def test_settings_instance_has_api_provider(self) -> None:
        """Instantiated settings object must expose API_PROVIDER attribute."""
        from app.core.config import settings

        self.assertTrue(hasattr(settings, "API_PROVIDER"))

    def test_settings_instance_has_llm_model(self) -> None:
        """Instantiated settings object must expose LLM_MODEL attribute."""
        from app.core.config import settings

        self.assertTrue(hasattr(settings, "LLM_MODEL"))


# ══════════════════════════════════════════════════════════════════════════════
#  6. transcription_service._get_client() — lazy-load caching
# ══════════════════════════════════════════════════════════════════════════════


class TestTranscriptionServiceLazyLoad(unittest.TestCase):
    """_get_client() must create the client once and cache it for subsequent calls."""

    def setUp(self) -> None:
        _ts._client = None  # Reset cache before each test

    def tearDown(self) -> None:
        _ts._client = None

    def test_first_call_calls_factory(self) -> None:
        """Factory (get_openai_client) must be called on first access."""
        mock_client = MagicMock(name="ts_client")
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ) as mock_factory:
            _ts._get_client()

        mock_factory.assert_called_once()

    def test_first_call_returns_factory_result(self) -> None:
        """First call must return the client from get_openai_client()."""
        mock_client = MagicMock(name="ts_client")
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ):
            result = _ts._get_client()

        self.assertIs(result, mock_client)

    def test_first_call_populates_module_cache(self) -> None:
        """After first call, _ts._client must hold the created client."""
        mock_client = MagicMock()
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ):
            _ts._get_client()

        self.assertIs(_ts._client, mock_client)

    def test_second_call_returns_same_instance(self) -> None:
        """Second call must return the cached instance without a new factory call."""
        mock_client = MagicMock()
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ):
            first = _ts._get_client()
            second = _ts._get_client()

        self.assertIs(first, second)

    def test_factory_called_only_once_across_multiple_calls(self) -> None:
        """get_openai_client() must be invoked exactly once no matter how many _get_client() calls."""
        mock_client = MagicMock()
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ) as mock_factory:
            for _ in range(5):
                _ts._get_client()

        mock_factory.assert_called_once()

    def test_warm_cache_skips_factory(self) -> None:
        """If _client is already set, factory must NOT be called."""
        existing = MagicMock(name="pre_warmed_client")
        _ts._client = existing

        with patch(
            "app.services.transcription_service.get_openai_client"
        ) as mock_factory:
            result = _ts._get_client()

        mock_factory.assert_not_called()
        self.assertIs(result, existing)

    def test_reset_cache_forces_new_factory_call(self) -> None:
        """After resetting _client to None, factory must be called again."""
        mock_client_1 = MagicMock(name="client1")
        mock_client_2 = MagicMock(name="client2")
        side_effects = [mock_client_1, mock_client_2]

        with patch(
            "app.services.transcription_service.get_openai_client",
            side_effect=side_effects,
        ) as mock_factory:
            first = _ts._get_client()
            _ts._client = None  # simulate reset
            second = _ts._get_client()

        self.assertEqual(mock_factory.call_count, 2)
        self.assertIs(first, mock_client_1)
        self.assertIs(second, mock_client_2)
        self.assertIsNot(first, second)


# ══════════════════════════════════════════════════════════════════════════════
#  7. llm_service._get_client() — lazy-load caching
# ══════════════════════════════════════════════════════════════════════════════


class TestLLMServiceLazyLoad(unittest.TestCase):
    """_get_client() in llm_service must lazily create and cache the instructor client."""

    def setUp(self) -> None:
        _llm._client = None

    def tearDown(self) -> None:
        _llm._client = None

    def test_first_call_calls_factory(self) -> None:
        mock_instructor = MagicMock()
        with patch(
            "app.services.llm_service.get_instructor_client",
            return_value=mock_instructor,
        ) as mock_factory:
            _llm._get_client()

        mock_factory.assert_called_once()

    def test_first_call_returns_factory_result(self) -> None:
        mock_instructor = MagicMock()
        with patch(
            "app.services.llm_service.get_instructor_client",
            return_value=mock_instructor,
        ):
            result = _llm._get_client()

        self.assertIs(result, mock_instructor)

    def test_first_call_populates_module_cache(self) -> None:
        mock_instructor = MagicMock()
        with patch(
            "app.services.llm_service.get_instructor_client",
            return_value=mock_instructor,
        ):
            _llm._get_client()

        self.assertIs(_llm._client, mock_instructor)

    def test_second_call_returns_cached_instance(self) -> None:
        mock_instructor = MagicMock()
        with patch(
            "app.services.llm_service.get_instructor_client",
            return_value=mock_instructor,
        ):
            first = _llm._get_client()
            second = _llm._get_client()

        self.assertIs(first, second)

    def test_factory_called_only_once(self) -> None:
        mock_instructor = MagicMock()
        with patch(
            "app.services.llm_service.get_instructor_client",
            return_value=mock_instructor,
        ) as mock_factory:
            for _ in range(4):
                _llm._get_client()

        mock_factory.assert_called_once()

    def test_warm_cache_skips_factory(self) -> None:
        existing = MagicMock(name="pre_warmed_llm_client")
        _llm._client = existing

        with patch(
            "app.services.llm_service.get_instructor_client"
        ) as mock_factory:
            result = _llm._get_client()

        mock_factory.assert_not_called()
        self.assertIs(result, existing)

    def test_analyze_intention_uses_cached_client(self) -> None:
        """Two consecutive analyze_intention() calls must share the same client."""
        from app.schemas.translation import CatTranslationResponse

        mock_response = MagicMock(spec=CatTranslationResponse)
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        with (
            patch(
                "app.services.llm_service.get_instructor_client",
                return_value=mock_instructor,
            ) as mock_factory,
            patch("app.services.llm_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = MODEL_OPENAI_DEFAULT
            _llm.analyze_intention("call one", {}, "")
            _llm.analyze_intention("call two", {}, "")

        mock_factory.assert_called_once()

    def test_reset_cache_forces_new_factory_call(self) -> None:
        """After resetting _client to None, factory must be called again."""
        mock_c1 = MagicMock(name="c1")
        mock_c2 = MagicMock(name="c2")
        with patch(
            "app.services.llm_service.get_instructor_client",
            side_effect=[mock_c1, mock_c2],
        ) as mock_factory:
            first = _llm._get_client()
            _llm._client = None
            second = _llm._get_client()

        self.assertEqual(mock_factory.call_count, 2)
        self.assertIs(first, mock_c1)
        self.assertIs(second, mock_c2)
        self.assertIsNot(first, second)


if __name__ == "__main__":
    unittest.main()
