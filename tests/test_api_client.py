"""
Tests for API Client Factory — app/core/api_client.py
=======================================================
Covers:
  1. get_openai_client() — uses OPENAI_API_KEY, no base_url
  2. get_instructor_client() — wraps get_openai_client() via instructor.from_openai()
  3. LLM_MODEL is read from settings, not hard-coded "gpt-4o"
  4. config.py defaults: LLM_MODEL
  5. transcription_service._get_client() lazy-load caching
  6. llm_service._get_client() lazy-load caching

No real API calls are made — all external clients are fully mocked.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

# ── Mock chromadb before any app import (project convention) ─────────────────
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())

# ── Test parameters ───────────────────────────────────────────────────────────
from tests.shared_params import (  # noqa: E402
    MODEL_OPENAI_DEFAULT,
    PROVIDER_OPENAI,
)

_DUMMY_SK_KEY = "sk-key"

# ── Subjects under test ───────────────────────────────────────────────────────
from app.core.api_client import get_instructor_client, get_openai_client  # noqa: E402
from app.core.config import Settings  # noqa: E402
import app.services.transcription_service as _ts  # noqa: E402
import app.services.llm_service as _llm  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  1. get_openai_client()
# ══════════════════════════════════════════════════════════════════════════════


class TestGetOpenAIClient(unittest.TestCase):
    """get_openai_client() must use OPENAI_API_KEY."""

    def _call_with_mock(self, api_key: str = "sk-test") -> MagicMock:
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            mock_settings.OPENAI_API_KEY = api_key
            get_openai_client()
            return MockOpenAI

    def test_calls_openai_with_api_key(self) -> None:
        MockOpenAI = self._call_with_mock("sk-my-openai-key")
        MockOpenAI.assert_called_once_with(api_key="sk-my-openai-key")

    def test_no_base_url_kwarg(self) -> None:
        MockOpenAI = self._call_with_mock()
        kwargs = MockOpenAI.call_args.kwargs
        self.assertNotIn("base_url", kwargs)

    def test_returns_openai_instance(self) -> None:
        fake_instance = MagicMock(name="fake_openai_client")
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI", return_value=fake_instance),
        ):
            mock_settings.OPENAI_API_KEY = _DUMMY_SK_KEY
            result = get_openai_client()
        self.assertIs(result, fake_instance)

    def test_factory_creates_fresh_instance_each_call(self) -> None:
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            mock_settings.OPENAI_API_KEY = _DUMMY_SK_KEY
            MockOpenAI.side_effect = lambda **_kw: MagicMock()

            c1 = get_openai_client()
            c2 = get_openai_client()

        self.assertIsNot(c1, c2)
        self.assertEqual(MockOpenAI.call_count, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  2. get_instructor_client() — instructor wrapping
# ══════════════════════════════════════════════════════════════════════════════


class TestGetInstructorClient(unittest.TestCase):
    """get_instructor_client() must wrap get_openai_client() via instructor."""

    def test_delegates_to_get_openai_client(self) -> None:
        mock_openai = MagicMock()
        with (
            patch("app.core.api_client.get_openai_client", return_value=mock_openai) as mock_factory,
            patch("app.core.api_client.instructor.from_openai", return_value=MagicMock()),
        ):
            get_instructor_client()

        mock_factory.assert_called_once()

    def test_passes_openai_client_to_instructor(self) -> None:
        mock_openai = MagicMock()
        with (
            patch("app.core.api_client.get_openai_client", return_value=mock_openai),
            patch("app.core.api_client.instructor.from_openai") as mock_from_openai,
        ):
            mock_from_openai.return_value = MagicMock()
            get_instructor_client()

        mock_from_openai.assert_called_once_with(mock_openai)

    def test_returns_instructor_result(self) -> None:
        mock_instructor_client = MagicMock(name="instructor_client")
        with (
            patch("app.core.api_client.get_openai_client", return_value=MagicMock()),
            patch("app.core.api_client.instructor.from_openai", return_value=mock_instructor_client),
        ):
            result = get_instructor_client()

        self.assertIs(result, mock_instructor_client)

    def test_returns_real_instructor_instance(self) -> None:
        import instructor
        from openai import OpenAI

        real_client = OpenAI(api_key="sk-fake-for-test")
        with patch("app.core.api_client.get_openai_client", return_value=real_client):
            result = get_instructor_client()

        self.assertIsInstance(result, instructor.Instructor)

    def test_factory_creates_fresh_instance_each_call(self) -> None:
        with (
            patch("app.core.api_client.get_openai_client", return_value=MagicMock()),
            patch("app.core.api_client.instructor.from_openai") as mock_from_openai,
        ):
            mock_from_openai.side_effect = lambda *a, **k: MagicMock()
            c1 = get_instructor_client()
            c2 = get_instructor_client()

        self.assertIsNot(c1, c2)
        self.assertEqual(mock_from_openai.call_count, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  3. LLM_MODEL propagation — settings.LLM_MODEL used, not "gpt-4o"
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

    def test_openai_default_llm_model_is_gpt4o(self) -> None:
        from app.schemas.translation import CatTranslationResponse

        mock_response = MagicMock(spec=CatTranslationResponse)
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        with (
            patch("app.services.llm_service.get_instructor_client", return_value=mock_instructor),
        ):
            from app.core.config import settings as real_settings
            _llm.analyze_intention("test", {}, "")

        kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], real_settings.LLM_MODEL)

    def test_custom_llm_model_propagated(self) -> None:
        """LLM_MODEL override must propagate to the API call."""
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
#  4. config.py default field values
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigDefaults(unittest.TestCase):
    """Settings schema must declare correct defaults."""

    @staticmethod
    def _default(field: str):
        return Settings.model_fields[field].default

    def test_llm_model_default_is_gpt4o(self) -> None:
        self.assertEqual(self._default("LLM_MODEL"), MODEL_OPENAI_DEFAULT)

    def test_llm_model_field_exists(self) -> None:
        self.assertIn("LLM_MODEL", Settings.model_fields)

    def test_settings_instance_has_llm_model(self) -> None:
        from app.core.config import settings
        self.assertTrue(hasattr(settings, "LLM_MODEL"))


# ══════════════════════════════════════════════════════════════════════════════
#  5. transcription_service._get_client() — lazy-load caching
# ══════════════════════════════════════════════════════════════════════════════


class TestTranscriptionServiceLazyLoad(unittest.TestCase):
    """_get_client() must create the client once and cache it for subsequent calls."""

    def setUp(self) -> None:
        _ts._client = None

    def tearDown(self) -> None:
        _ts._client = None

    def test_first_call_calls_factory(self) -> None:
        mock_client = MagicMock(name="ts_client")
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ) as mock_factory:
            _ts._get_client()

        mock_factory.assert_called_once()

    def test_first_call_returns_factory_result(self) -> None:
        mock_client = MagicMock(name="ts_client")
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ):
            result = _ts._get_client()

        self.assertIs(result, mock_client)

    def test_first_call_populates_module_cache(self) -> None:
        mock_client = MagicMock()
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ):
            _ts._get_client()

        self.assertIs(_ts._client, mock_client)

    def test_second_call_returns_same_instance(self) -> None:
        mock_client = MagicMock()
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ):
            first = _ts._get_client()
            second = _ts._get_client()

        self.assertIs(first, second)

    def test_factory_called_only_once_across_multiple_calls(self) -> None:
        mock_client = MagicMock()
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ) as mock_factory:
            for _ in range(5):
                _ts._get_client()

        mock_factory.assert_called_once()

    def test_warm_cache_skips_factory(self) -> None:
        existing = MagicMock(name="pre_warmed_client")
        _ts._client = existing

        with patch(
            "app.services.transcription_service.get_openai_client"
        ) as mock_factory:
            result = _ts._get_client()

        mock_factory.assert_not_called()
        self.assertIs(result, existing)

    def test_reset_cache_forces_new_factory_call(self) -> None:
        mock_client_1 = MagicMock(name="client1")
        mock_client_2 = MagicMock(name="client2")
        side_effects = [mock_client_1, mock_client_2]

        with patch(
            "app.services.transcription_service.get_openai_client",
            side_effect=side_effects,
        ) as mock_factory:
            first = _ts._get_client()
            _ts._client = None
            second = _ts._get_client()

        self.assertEqual(mock_factory.call_count, 2)
        self.assertIs(first, mock_client_1)
        self.assertIs(second, mock_client_2)
        self.assertIsNot(first, second)


# ══════════════════════════════════════════════════════════════════════════════
#  6. llm_service._get_client() — lazy-load caching
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


# ── Async migration imports ──────────────────────────────────────────────────
import asyncio  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402

from app.core.api_client import get_async_openai_client, get_async_instructor_client  # noqa: E402
import app.services.streaming_transcription_service as _sts  # noqa: E402
import app.services.sound_selection_service as _sss  # noqa: E402

from tests.shared_params import DUMMY_PCM_BYTES  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  7. get_async_openai_client()
# ══════════════════════════════════════════════════════════════════════════════


class TestGetAsyncOpenAIClient(unittest.TestCase):
    """get_async_openai_client() must use OPENAI_API_KEY and return AsyncOpenAI."""

    def _call_with_mock(self, api_key: str = "sk-test") -> MagicMock:
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.AsyncOpenAI") as MockAsyncOpenAI,
        ):
            mock_settings.OPENAI_API_KEY = api_key
            get_async_openai_client()
            return MockAsyncOpenAI

    def test_calls_async_openai_with_api_key(self) -> None:
        MockAsyncOpenAI = self._call_with_mock("sk-async-key")
        MockAsyncOpenAI.assert_called_once_with(api_key="sk-async-key")

    def test_no_base_url_kwarg(self) -> None:
        MockAsyncOpenAI = self._call_with_mock()
        kwargs = MockAsyncOpenAI.call_args.kwargs
        self.assertNotIn("base_url", kwargs)

    def test_returns_async_openai_instance(self) -> None:
        fake_instance = MagicMock(name="fake_async_openai")
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.AsyncOpenAI", return_value=fake_instance),
        ):
            mock_settings.OPENAI_API_KEY = _DUMMY_SK_KEY
            result = get_async_openai_client()
        self.assertIs(result, fake_instance)

    def test_factory_creates_fresh_instance_each_call(self) -> None:
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.AsyncOpenAI") as MockAsyncOpenAI,
        ):
            mock_settings.OPENAI_API_KEY = _DUMMY_SK_KEY
            MockAsyncOpenAI.side_effect = lambda **_kw: MagicMock()

            c1 = get_async_openai_client()
            c2 = get_async_openai_client()

        self.assertIsNot(c1, c2)
        self.assertEqual(MockAsyncOpenAI.call_count, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  8. get_async_instructor_client() — async instructor wrapping
# ══════════════════════════════════════════════════════════════════════════════


class TestGetAsyncInstructorClient(unittest.TestCase):
    """get_async_instructor_client() must wrap get_async_openai_client() via instructor."""

    def test_delegates_to_get_async_openai_client(self) -> None:
        mock_async_openai = MagicMock()
        with (
            patch("app.core.api_client.get_async_openai_client", return_value=mock_async_openai) as mock_factory,
            patch("app.core.api_client.instructor.from_openai", return_value=MagicMock()),
        ):
            get_async_instructor_client()

        mock_factory.assert_called_once()

    def test_passes_async_openai_client_to_instructor(self) -> None:
        mock_async_openai = MagicMock()
        with (
            patch("app.core.api_client.get_async_openai_client", return_value=mock_async_openai),
            patch("app.core.api_client.instructor.from_openai") as mock_from_openai,
        ):
            mock_from_openai.return_value = MagicMock()
            get_async_instructor_client()

        mock_from_openai.assert_called_once_with(mock_async_openai)

    def test_returns_instructor_result(self) -> None:
        mock_instructor_client = MagicMock(name="async_instructor_client")
        with (
            patch("app.core.api_client.get_async_openai_client", return_value=MagicMock()),
            patch("app.core.api_client.instructor.from_openai", return_value=mock_instructor_client),
        ):
            result = get_async_instructor_client()

        self.assertIs(result, mock_instructor_client)

    def test_factory_creates_fresh_instance_each_call(self) -> None:
        with (
            patch("app.core.api_client.get_async_openai_client", return_value=MagicMock()),
            patch("app.core.api_client.instructor.from_openai") as mock_from_openai,
        ):
            mock_from_openai.side_effect = lambda *a, **k: MagicMock()
            c1 = get_async_instructor_client()
            c2 = get_async_instructor_client()

        self.assertIsNot(c1, c2)
        self.assertEqual(mock_from_openai.call_count, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  9. streaming_transcription_service._get_client() — async lazy-load
# ══════════════════════════════════════════════════════════════════════════════


class TestStreamingTranscriptionServiceLazyLoad(unittest.TestCase):
    """_get_client() must lazily create and cache the AsyncOpenAI client."""

    def setUp(self) -> None:
        _sts._client = None

    def tearDown(self) -> None:
        _sts._client = None

    def test_first_call_calls_factory(self) -> None:
        mock_client = MagicMock(name="sts_client")
        with patch(
            "app.services.streaming_transcription_service.get_async_openai_client",
            return_value=mock_client,
        ) as mock_factory:
            _sts._get_client()

        mock_factory.assert_called_once()

    def test_first_call_returns_factory_result(self) -> None:
        mock_client = MagicMock(name="sts_client")
        with patch(
            "app.services.streaming_transcription_service.get_async_openai_client",
            return_value=mock_client,
        ):
            result = _sts._get_client()

        self.assertIs(result, mock_client)

    def test_first_call_populates_module_cache(self) -> None:
        mock_client = MagicMock()
        with patch(
            "app.services.streaming_transcription_service.get_async_openai_client",
            return_value=mock_client,
        ):
            _sts._get_client()

        self.assertIs(_sts._client, mock_client)

    def test_second_call_returns_same_instance(self) -> None:
        mock_client = MagicMock()
        with patch(
            "app.services.streaming_transcription_service.get_async_openai_client",
            return_value=mock_client,
        ):
            first = _sts._get_client()
            second = _sts._get_client()

        self.assertIs(first, second)

    def test_factory_called_only_once_across_multiple_calls(self) -> None:
        mock_client = MagicMock()
        with patch(
            "app.services.streaming_transcription_service.get_async_openai_client",
            return_value=mock_client,
        ) as mock_factory:
            for _ in range(5):
                _sts._get_client()

        mock_factory.assert_called_once()

    def test_warm_cache_skips_factory(self) -> None:
        existing = MagicMock(name="pre_warmed_async_client")
        _sts._client = existing

        with patch(
            "app.services.streaming_transcription_service.get_async_openai_client"
        ) as mock_factory:
            result = _sts._get_client()

        mock_factory.assert_not_called()
        self.assertIs(result, existing)

    def test_reset_cache_forces_new_factory_call(self) -> None:
        mock_client_1 = MagicMock(name="client1")
        mock_client_2 = MagicMock(name="client2")

        with patch(
            "app.services.streaming_transcription_service.get_async_openai_client",
            side_effect=[mock_client_1, mock_client_2],
        ) as mock_factory:
            first = _sts._get_client()
            _sts._client = None
            second = _sts._get_client()

        self.assertEqual(mock_factory.call_count, 2)
        self.assertIs(first, mock_client_1)
        self.assertIs(second, mock_client_2)
        self.assertIsNot(first, second)


# ══════════════════════════════════════════════════════════════════════════════
# 10. sound_selection_service._get_client() — async lazy-load
# ══════════════════════════════════════════════════════════════════════════════


class TestSoundSelectionServiceLazyLoad(unittest.TestCase):
    """_get_client() must lazily create and cache the AsyncInstructor client."""

    def setUp(self) -> None:
        _sss._client = None

    def tearDown(self) -> None:
        _sss._client = None

    def test_first_call_calls_factory(self) -> None:
        mock_instructor = MagicMock()
        with patch(
            "app.services.sound_selection_service.get_async_instructor_client",
            return_value=mock_instructor,
        ) as mock_factory:
            _sss._get_client()

        mock_factory.assert_called_once()

    def test_first_call_returns_factory_result(self) -> None:
        mock_instructor = MagicMock()
        with patch(
            "app.services.sound_selection_service.get_async_instructor_client",
            return_value=mock_instructor,
        ):
            result = _sss._get_client()

        self.assertIs(result, mock_instructor)

    def test_first_call_populates_module_cache(self) -> None:
        mock_instructor = MagicMock()
        with patch(
            "app.services.sound_selection_service.get_async_instructor_client",
            return_value=mock_instructor,
        ):
            _sss._get_client()

        self.assertIs(_sss._client, mock_instructor)

    def test_second_call_returns_cached_instance(self) -> None:
        mock_instructor = MagicMock()
        with patch(
            "app.services.sound_selection_service.get_async_instructor_client",
            return_value=mock_instructor,
        ):
            first = _sss._get_client()
            second = _sss._get_client()

        self.assertIs(first, second)

    def test_factory_called_only_once(self) -> None:
        mock_instructor = MagicMock()
        with patch(
            "app.services.sound_selection_service.get_async_instructor_client",
            return_value=mock_instructor,
        ) as mock_factory:
            for _ in range(4):
                _sss._get_client()

        mock_factory.assert_called_once()

    def test_warm_cache_skips_factory(self) -> None:
        existing = MagicMock(name="pre_warmed_async_instructor")
        _sss._client = existing

        with patch(
            "app.services.sound_selection_service.get_async_instructor_client"
        ) as mock_factory:
            result = _sss._get_client()

        mock_factory.assert_not_called()
        self.assertIs(result, existing)

    def test_reset_cache_forces_new_factory_call(self) -> None:
        mock_c1 = MagicMock(name="c1")
        mock_c2 = MagicMock(name="c2")
        with patch(
            "app.services.sound_selection_service.get_async_instructor_client",
            side_effect=[mock_c1, mock_c2],
        ) as mock_factory:
            first = _sss._get_client()
            _sss._client = None
            second = _sss._get_client()

        self.assertEqual(mock_factory.call_count, 2)
        self.assertIs(first, mock_c1)
        self.assertIs(second, mock_c2)
        self.assertIsNot(first, second)


# ══════════════════════════════════════════════════════════════════════════════
# 11. _call_whisper() — async Whisper API call
# ══════════════════════════════════════════════════════════════════════════════


class TestCallWhisperAsync(unittest.TestCase):
    """_call_whisper() must await the async transcriptions.create call."""

    def setUp(self) -> None:
        _sts._client = None

    def tearDown(self) -> None:
        _sts._client = None

    def test_call_whisper_awaits_transcription_create(self) -> None:
        session = _sts.StreamingTranscriptionSession(sample_rate=16000)
        session.add_chunk(DUMMY_PCM_BYTES)

        mock_client = MagicMock()
        mock_create = AsyncMock(return_value="hello world")
        mock_client.audio.transcriptions.create = mock_create

        with patch(
            "app.services.streaming_transcription_service._get_client",
            return_value=mock_client,
        ):
            result = asyncio.run(session._call_whisper())

        mock_create.assert_awaited_once()
        self.assertEqual(result, "hello world")

    def test_call_whisper_passes_model_whisper1(self) -> None:
        session = _sts.StreamingTranscriptionSession(sample_rate=16000)
        session.add_chunk(DUMMY_PCM_BYTES)

        mock_client = MagicMock()
        mock_create = AsyncMock(return_value="text")
        mock_client.audio.transcriptions.create = mock_create

        with patch(
            "app.services.streaming_transcription_service._get_client",
            return_value=mock_client,
        ):
            asyncio.run(session._call_whisper())

        call_kwargs = mock_create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "whisper-1")
        self.assertEqual(call_kwargs["response_format"], "text")
        self.assertIn("file", call_kwargs)

    def test_call_whisper_strips_whitespace(self) -> None:
        session = _sts.StreamingTranscriptionSession(sample_rate=16000)
        session.add_chunk(DUMMY_PCM_BYTES)

        mock_client = MagicMock()
        mock_create = AsyncMock(return_value="  padded text  ")
        mock_client.audio.transcriptions.create = mock_create

        with patch(
            "app.services.streaming_transcription_service._get_client",
            return_value=mock_client,
        ):
            result = asyncio.run(session._call_whisper())

        self.assertEqual(result, "padded text")


# ══════════════════════════════════════════════════════════════════════════════
# 12. generate_target_tags() — async instructor LLM call
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerateTargetTagsAsync(unittest.TestCase):
    """generate_target_tags() must await the async instructor create call."""

    def setUp(self) -> None:
        _sss._client = None

    def tearDown(self) -> None:
        _sss._client = None

    def test_awaits_instructor_create(self) -> None:
        from app.schemas.ws_messages import TargetTagSet

        expected_tags = TargetTagSet(
            emotion=["happy"],
            intent=["greeting"],
            acoustic=["high_pitch", "short"],
            social_context=["near_owner"],
            reasoning="测试推理",
        )

        mock_client = MagicMock()
        mock_create = AsyncMock(return_value=expected_tags)
        mock_client.chat.completions.create = mock_create

        with (
            patch("app.services.sound_selection_service._get_client", return_value=mock_client),
            patch("app.services.sound_selection_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = MODEL_OPENAI_DEFAULT
            result = asyncio.run(_sss.generate_target_tags("你好小猫"))

        mock_create.assert_awaited_once()
        self.assertIsInstance(result, TargetTagSet)

    def test_returns_llm_generated_tags(self) -> None:
        from app.schemas.ws_messages import TargetTagSet

        expected_tags = TargetTagSet(
            emotion=["calm"],
            intent=["expressing_comfort"],
            acoustic=["mid_pitch"],
            social_context=["near_owner"],
            reasoning="推理说明",
        )

        mock_client = MagicMock()
        mock_create = AsyncMock(return_value=expected_tags)
        mock_client.chat.completions.create = mock_create

        with (
            patch("app.services.sound_selection_service._get_client", return_value=mock_client),
            patch("app.services.sound_selection_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = MODEL_OPENAI_DEFAULT
            result = asyncio.run(_sss.generate_target_tags("乖猫咪"))

        self.assertIs(result, expected_tags)
        self.assertEqual(result.emotion, ["calm"])

    def test_returns_default_tags_on_llm_failure(self) -> None:
        from app.schemas.ws_messages import TargetTagSet

        mock_client = MagicMock()
        mock_create = AsyncMock(side_effect=Exception("API error"))
        mock_client.chat.completions.create = mock_create

        with (
            patch("app.services.sound_selection_service._get_client", return_value=mock_client),
            patch("app.services.sound_selection_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = MODEL_OPENAI_DEFAULT
            result = asyncio.run(_sss.generate_target_tags("test"))

        self.assertIsInstance(result, TargetTagSet)
        self.assertEqual(result.emotion, ["calm"])
        self.assertEqual(result.intent, ["expressing_comfort"])
        self.assertEqual(result.acoustic, ["mid_pitch", "medium_length"])
        self.assertEqual(result.social_context, ["near_owner"])
        self.assertIn("language analysis was unavailable", result.reasoning)

    def test_passes_response_model_target_tag_set(self) -> None:
        from app.schemas.ws_messages import TargetTagSet

        mock_client = MagicMock()
        mock_create = AsyncMock(return_value=TargetTagSet(reasoning="ok"))
        mock_client.chat.completions.create = mock_create

        with (
            patch("app.services.sound_selection_service._get_client", return_value=mock_client),
            patch("app.services.sound_selection_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = MODEL_OPENAI_DEFAULT
            asyncio.run(_sss.generate_target_tags("test"))

        call_kwargs = mock_create.call_args.kwargs
        self.assertIs(call_kwargs["response_model"], TargetTagSet)
        self.assertEqual(call_kwargs["model"], MODEL_OPENAI_DEFAULT)


if __name__ == "__main__":
    unittest.main()
