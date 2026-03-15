"""
Tests for API Provider Switch Feature
======================================
Covers:
  1. get_openai_client() factory — openai vs ai_builders provider
  2. get_instructor_client() factory — instructor wrapping
  3. config.py default field values
  4. transcription_service._get_client() lazy initialisation
  5. streaming_transcription_service._get_client() lazy initialisation
  6. llm_service.analyze_intention() model name propagation
  7. sound_selection_service.generate_target_tags() model name propagation
  8. vector_store conditional embedding function initialisation

All external calls (OpenAI, ChromaDB, instructor) are fully mocked —
no real API calls are made.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Mock chromadb BEFORE any app imports (project convention) ────────────────
# We construct linked mocks so that
#   from chromadb.utils import embedding_functions
# inside vector_store.py retrieves the *same* object we can later inspect.

_mock_chromadb = MagicMock()
_mock_chromadb_utils = MagicMock()
_mock_embedding_functions = MagicMock()

_mock_chromadb.utils = _mock_chromadb_utils
_mock_chromadb_utils.embedding_functions = _mock_embedding_functions

sys.modules["chromadb"] = _mock_chromadb
sys.modules["chromadb.utils"] = _mock_chromadb_utils
sys.modules["chromadb.utils.embedding_functions"] = _mock_embedding_functions

# ── Test parameters ───────────────────────────────────────────────────────────
from tests.shared_params import (  # noqa: E402
    AI_BUILDER_BASE_URL,
    CHROMA_PATH_BUILDER,
    CHROMA_PATH_DEFAULT,
    CHROMA_PATH_OPENAI,
    EMBEDDING_MODEL,
    MODEL_AI_BUILDERS_DEFAULT,
    MODEL_OPENAI_DEFAULT,
    PROVIDER_AI_BUILDERS,
    PROVIDER_OPENAI,
)

# File-internal SK key stubs (appear 2+ times within this file)
_SK_OPENAI_CLIENT  = "sk-test-openai-key"     # TestGetOpenAIClientFactory ×2
_SK_BUILDER_TOKEN  = "sk-builder-token"        # TestGetOpenAIClientFactory + TestVectorStore ×3
_SK_BUILDER_SIMPLE = "sk-builder"              # TestGetOpenAIClientFactory + TestVectorStore ×2
_SK_OPENAI_EMBED   = "sk-openai-test-key"      # TestVectorStoreConditionalInit ×2
_SK_BUILDER_EMBED  = "sk-builder-test-token"   # TestVectorStoreConditionalInit ×2

# ── Now import subjects under test ───────────────────────────────────────────
from app.core.api_client import get_instructor_client, get_openai_client  # noqa: E402
from app.core.config import Settings  # noqa: E402
import app.services.transcription_service as _ts  # noqa: E402
import app.services.streaming_transcription_service as _sts  # noqa: E402
import app.services.llm_service as _llm  # noqa: E402
import app.services.sound_selection_service as _sss  # noqa: E402
import app.db.vector_store as _vs  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  1. get_openai_client() factory
# ══════════════════════════════════════════════════════════════════════════════

class TestGetOpenAIClientFactory(unittest.TestCase):
    """get_openai_client() must create different client configs per provider."""

    def test_openai_provider_uses_openai_api_key(self) -> None:
        """API_PROVIDER=openai → client initialised with OPENAI_API_KEY only."""
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_settings.OPENAI_API_KEY = _SK_OPENAI_CLIENT

            get_openai_client()

        MockOpenAI.assert_called_once_with(api_key=_SK_OPENAI_CLIENT)
        # base_url must NOT appear in the call kwargs
        _, kwargs = MockOpenAI.call_args
        self.assertNotIn("base_url", kwargs)

    def test_ai_builders_provider_uses_builder_token_and_base_url(self) -> None:
        """API_PROVIDER=ai_builders → client uses AI_BUILDER_TOKEN + base_url."""
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            mock_settings.API_PROVIDER = PROVIDER_AI_BUILDERS
            mock_settings.AI_BUILDER_TOKEN = _SK_BUILDER_TOKEN
            mock_settings.AI_BUILDER_BASE_URL = AI_BUILDER_BASE_URL

            get_openai_client()

        MockOpenAI.assert_called_once_with(
            api_key=_SK_BUILDER_TOKEN,
            base_url=AI_BUILDER_BASE_URL,
        )

    def test_factory_creates_new_instance_each_call(self) -> None:
        """Factory has no internal cache — each call may return a new object."""
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_settings.OPENAI_API_KEY = "sk-key"
            MockOpenAI.side_effect = lambda **_: MagicMock()

            client1 = get_openai_client()
            client2 = get_openai_client()

        self.assertIsNot(client1, client2)

    def test_different_providers_yield_different_configs(self) -> None:
        """Switching API_PROVIDER between calls produces a different init call."""
        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.OpenAI") as MockOpenAI,
        ):
            MockOpenAI.side_effect = lambda **kw: MagicMock(**kw)

            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_settings.OPENAI_API_KEY = "sk-openai"
            client_openai = get_openai_client()

            mock_settings.API_PROVIDER = PROVIDER_AI_BUILDERS
            mock_settings.AI_BUILDER_TOKEN = _SK_BUILDER_SIMPLE
            mock_settings.AI_BUILDER_BASE_URL = AI_BUILDER_BASE_URL
            client_builder = get_openai_client()

        # The two returned objects are different mocks → different configs
        self.assertIsNot(client_openai, client_builder)
        self.assertEqual(MockOpenAI.call_count, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  2. get_instructor_client() factory
# ══════════════════════════════════════════════════════════════════════════════

class TestGetInstructorClientFactory(unittest.TestCase):
    """get_instructor_client() must delegate to get_openai_client() and wrap it."""

    def test_calls_get_openai_client_exactly_once(self) -> None:
        """instructor factory must call get_openai_client() to obtain the base client."""
        mock_openai_client = MagicMock()
        mock_instructor_client = MagicMock()

        with (
            patch("app.core.api_client.get_openai_client", return_value=mock_openai_client) as mock_factory,
            patch("app.core.api_client.instructor.from_openai", return_value=mock_instructor_client),
        ):
            result = get_instructor_client()

        mock_factory.assert_called_once()
        self.assertIs(result, mock_instructor_client)

    def test_from_openai_receives_openai_client(self) -> None:
        """instructor.from_openai() must receive the client from the factory."""
        mock_openai_client = MagicMock()
        mock_instructor_client = MagicMock()

        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.get_openai_client", return_value=mock_openai_client),
            patch("app.core.api_client.instructor.from_openai", return_value=mock_instructor_client) as mock_from_openai,
        ):
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            get_instructor_client()

        mock_from_openai.assert_called_once_with(mock_openai_client)

    def test_returns_instructor_instance(self) -> None:
        """Return value must be an instructor.Instructor when not mocked.

        instructor.from_openai() validates that the client is a real
        openai.OpenAI instance (it returns None for MagicMock), so we supply
        a real OpenAI client constructed with a dummy key.
        """
        import instructor
        from openai import OpenAI

        real_openai_client = OpenAI(api_key="sk-fake-key-for-test")

        with (
            patch("app.core.api_client.get_openai_client", return_value=real_openai_client),
            patch("app.core.api_client.settings") as mock_settings,
        ):
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            result = get_instructor_client()

        self.assertIsInstance(result, instructor.Instructor)

    def test_ai_builders_uses_json_mode(self) -> None:
        """When API_PROVIDER is ai_builders, instructor.from_openai must use mode=JSON.

        DeepSeek and similar models return plain JSON in content, not tool calls.
        Mode.JSON avoids 'Instructor does not support multiple tool calls' error.
        """
        import instructor

        mock_openai_client = MagicMock()
        mock_instructor_client = MagicMock()

        with (
            patch("app.core.api_client.settings") as mock_settings,
            patch("app.core.api_client.get_openai_client", return_value=mock_openai_client),
            patch("app.core.api_client.instructor.from_openai", return_value=mock_instructor_client) as mock_from_openai,
        ):
            mock_settings.API_PROVIDER = PROVIDER_AI_BUILDERS
            get_instructor_client()

        mock_from_openai.assert_called_once_with(mock_openai_client, mode=instructor.Mode.JSON)


# ══════════════════════════════════════════════════════════════════════════════
#  3. config.py default field values
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigDefaults(unittest.TestCase):
    """New fields must have sensible defaults in the Settings schema."""

    @classmethod
    def _field_default(cls, field_name: str):
        return Settings.model_fields[field_name].default

    def test_api_provider_default_is_openai(self) -> None:
        self.assertEqual(self._field_default("API_PROVIDER"), PROVIDER_OPENAI)

    def test_ai_builder_token_default_is_empty(self) -> None:
        self.assertEqual(self._field_default("AI_BUILDER_TOKEN"), "")

    def test_llm_model_default_is_gpt4o(self) -> None:
        """With API_PROVIDER=openai, LLM_MODEL resolves to gpt-4o."""
        s = Settings(API_PROVIDER=PROVIDER_OPENAI)
        self.assertEqual(s.LLM_MODEL, MODEL_OPENAI_DEFAULT)

    def test_llm_model_default_is_deepseek_for_ai_builders(self) -> None:
        """With API_PROVIDER=ai_builders, LLM_MODEL resolves to deepseek."""
        s = Settings(API_PROVIDER=PROVIDER_AI_BUILDERS)
        self.assertEqual(s.LLM_MODEL, MODEL_AI_BUILDERS_DEFAULT)

    def test_ai_builder_base_url_contains_ai_builders(self) -> None:
        url: str = self._field_default("AI_BUILDER_BASE_URL")
        self.assertIn("ai-builders", url)
        self.assertNotEqual(url, "")

    def test_api_provider_is_literal_typed(self) -> None:
        """API_PROVIDER field must accept only 'openai' or 'ai_builders'."""
        from pydantic import ValidationError
        with self.assertRaises((ValidationError, ValueError)):
            Settings(API_PROVIDER="unknown_provider")  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════════════
#  4. transcription_service lazy client initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestTranscriptionServiceLazyLoad(unittest.TestCase):
    """_get_client() in transcription_service must lazily create and cache the client."""

    def setUp(self) -> None:
        _ts._client = None  # reset module-level cache before each test

    def tearDown(self) -> None:
        _ts._client = None

    def test_first_call_initialises_cache(self) -> None:
        mock_client = MagicMock()
        with patch("app.services.transcription_service.get_openai_client", return_value=mock_client):
            result = _ts._get_client()

        self.assertIs(result, mock_client)
        self.assertIs(_ts._client, mock_client)

    def test_second_call_returns_same_instance(self) -> None:
        mock_client = MagicMock()
        with patch(
            "app.services.transcription_service.get_openai_client",
            return_value=mock_client,
        ) as mock_factory:
            first = _ts._get_client()
            second = _ts._get_client()

        self.assertIs(first, second)
        mock_factory.assert_called_once()

    def test_factory_not_called_when_cache_warm(self) -> None:
        existing = MagicMock()
        _ts._client = existing

        with patch(
            "app.services.transcription_service.get_openai_client"
        ) as mock_factory:
            result = _ts._get_client()

        mock_factory.assert_not_called()
        self.assertIs(result, existing)


# ══════════════════════════════════════════════════════════════════════════════
#  5. streaming_transcription_service lazy client initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamingTranscriptionServiceLazyLoad(unittest.TestCase):
    """_get_client() in streaming_transcription_service must lazily cache the client."""

    def setUp(self) -> None:
        _sts._client = None

    def tearDown(self) -> None:
        _sts._client = None

    def test_first_call_initialises_cache(self) -> None:
        mock_client = MagicMock()
        with patch(
            "app.services.streaming_transcription_service.get_openai_client",
            return_value=mock_client,
        ):
            result = _sts._get_client()

        self.assertIs(result, mock_client)
        self.assertIs(_sts._client, mock_client)

    def test_second_call_returns_same_instance(self) -> None:
        mock_client = MagicMock()
        with patch(
            "app.services.streaming_transcription_service.get_openai_client",
            return_value=mock_client,
        ) as mock_factory:
            first = _sts._get_client()
            second = _sts._get_client()

        self.assertIs(first, second)
        mock_factory.assert_called_once()

    def test_factory_not_called_when_cache_warm(self) -> None:
        existing = MagicMock()
        _sts._client = existing

        with patch(
            "app.services.streaming_transcription_service.get_openai_client"
        ) as mock_factory:
            result = _sts._get_client()

        mock_factory.assert_not_called()
        self.assertIs(result, existing)


# ══════════════════════════════════════════════════════════════════════════════
#  6. llm_service — model name propagation
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMServiceModelName(unittest.TestCase):
    """analyze_intention() must pass settings.LLM_MODEL to the chat completion."""

    def setUp(self) -> None:
        _llm._client = None

    def tearDown(self) -> None:
        _llm._client = None

    def test_analyze_intention_uses_settings_llm_model(self) -> None:
        from app.schemas.translation import CatTranslationResponse

        mock_response = MagicMock(spec=CatTranslationResponse)
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        with (
            patch(
                "app.services.llm_service.get_instructor_client",
                return_value=mock_instructor,
            ),
            patch("app.services.llm_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = "test-model-xyz"

            _llm.analyze_intention("hello cat", {}, "")

        call_kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "test-model-xyz")

    def test_model_name_not_hardcoded(self) -> None:
        """Changing settings.LLM_MODEL must change the model arg, not be ignored."""
        from app.schemas.translation import CatTranslationResponse

        mock_response = MagicMock(spec=CatTranslationResponse)
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        with (
            patch(
                "app.services.llm_service.get_instructor_client",
                return_value=mock_instructor,
            ),
            patch("app.services.llm_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = "custom-model-v2"

            _llm._client = None  # ensure fresh client from factory
            _llm.analyze_intention("test", {}, "")

        call_kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        self.assertNotEqual(call_kwargs["model"], MODEL_OPENAI_DEFAULT)
        self.assertEqual(call_kwargs["model"], "custom-model-v2")

    def test_lazy_client_cached_across_calls(self) -> None:
        """_get_client() inside llm_service must follow the lazy pattern."""
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


# ══════════════════════════════════════════════════════════════════════════════
#  7. sound_selection_service — model name propagation
# ══════════════════════════════════════════════════════════════════════════════

class TestSoundSelectionServiceModelName(unittest.IsolatedAsyncioTestCase):
    """generate_target_tags() must pass settings.LLM_MODEL to the chat completion."""

    def setUp(self) -> None:
        _sss._client = None

    def tearDown(self) -> None:
        _sss._client = None

    async def test_generate_target_tags_uses_settings_llm_model(self) -> None:
        from app.schemas.ws_messages import TargetTagSet

        mock_response = MagicMock(spec=TargetTagSet)
        mock_response.model_dump.return_value = {}
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        with (
            patch(
                "app.services.sound_selection_service.get_instructor_client",
                return_value=mock_instructor,
            ),
            patch("app.services.sound_selection_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = "test-model-abc"

            await _sss.generate_target_tags("I love you cat")

        call_kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "test-model-abc")

    async def test_model_name_reflects_different_values(self) -> None:
        """Model name must change when settings.LLM_MODEL changes."""
        from app.schemas.ws_messages import TargetTagSet

        mock_response = MagicMock(spec=TargetTagSet)
        mock_response.model_dump.return_value = {}
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        with (
            patch(
                "app.services.sound_selection_service.get_instructor_client",
                return_value=mock_instructor,
            ),
            patch("app.services.sound_selection_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = "deepseek-chat"

            _sss._client = None
            await _sss.generate_target_tags("hello")

        call_kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "deepseek-chat")

    async def test_lazy_client_cached_across_calls(self) -> None:
        """_get_client() in sound_selection_service must follow the lazy pattern."""
        from app.schemas.ws_messages import TargetTagSet

        mock_response = MagicMock(spec=TargetTagSet)
        mock_response.model_dump.return_value = {}
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_response

        with (
            patch(
                "app.services.sound_selection_service.get_instructor_client",
                return_value=mock_instructor,
            ) as mock_factory,
            patch("app.services.sound_selection_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = MODEL_OPENAI_DEFAULT

            await _sss.generate_target_tags("call one")
            await _sss.generate_target_tags("call two")

        mock_factory.assert_called_once()

    async def test_llm_failure_returns_default_tags(self) -> None:
        """On LLM error, generate_target_tags must return calm/expressing_comfort defaults."""
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.side_effect = RuntimeError("API error")

        with (
            patch(
                "app.services.sound_selection_service.get_instructor_client",
                return_value=mock_instructor,
            ),
            patch("app.services.sound_selection_service.settings") as mock_settings,
        ):
            mock_settings.LLM_MODEL = MODEL_OPENAI_DEFAULT

            result = await _sss.generate_target_tags("any text")

        self.assertIn("calm", result.emotion)
        self.assertIn("expressing_comfort", result.intent)


# ══════════════════════════════════════════════════════════════════════════════
#  8. vector_store — conditional embedding function initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreConditionalInit(unittest.TestCase):
    """vector_store.py must configure the embedding function based on API_PROVIDER."""

    def _reload_with_provider(self, api_provider: str, **kw) -> None:
        """Reload app.db.vector_store with the given settings configuration."""
        defaults = {
            "API_PROVIDER": api_provider,
            "OPENAI_API_KEY": "sk-openai-default",
            "AI_BUILDER_TOKEN": "sk-builder-default",
            "AI_BUILDER_BASE_URL": "https://space.ai-builders.com/backend/v1",
            "CHROMA_DB_PATH": "/tmp/test_chroma_db",
        }
        defaults.update(kw)

        with patch("app.core.config.settings") as mock_settings:
            for attr, val in defaults.items():
                setattr(mock_settings, attr, val)

            importlib.reload(_vs)

            # Assertions must be inside the context so mock_settings stays active
            return mock_settings  # for post-check convenience (not used currently)

    def setUp(self) -> None:
        # Re-install the linked mocks in case another test file's module-level
        # code (e.g. test_rag_service.py) has replaced sys.modules["chromadb.utils"]
        # with a fresh anonymous MagicMock via direct assignment.
        sys.modules["chromadb"] = _mock_chromadb
        sys.modules["chromadb.utils"] = _mock_chromadb_utils
        sys.modules["chromadb.utils.embedding_functions"] = _mock_embedding_functions
        _mock_chromadb_utils.embedding_functions = _mock_embedding_functions
        _mock_embedding_functions.OpenAIEmbeddingFunction.reset_mock()

    def tearDown(self) -> None:
        # Restore a neutral module state after each test
        _mock_embedding_functions.OpenAIEmbeddingFunction.reset_mock()

    def test_openai_provider_uses_openai_api_key(self) -> None:
        """API_PROVIDER=openai → embedding function initialised with OPENAI_API_KEY."""
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_settings.OPENAI_API_KEY = _SK_OPENAI_EMBED
            mock_settings.AI_BUILDER_TOKEN = ""
            mock_settings.AI_BUILDER_BASE_URL = ""
            mock_settings.CHROMA_DB_PATH = CHROMA_PATH_OPENAI

            importlib.reload(_vs)

            call_args_list = _mock_embedding_functions.OpenAIEmbeddingFunction.call_args_list

        # Find the call that used OPENAI_API_KEY (last reload call)
        last_call_kwargs = call_args_list[-1].kwargs
        self.assertEqual(last_call_kwargs["api_key"], _SK_OPENAI_EMBED)
        self.assertNotIn("api_base", last_call_kwargs)

    def test_openai_provider_uses_correct_model_name(self) -> None:
        """The embedding model must be text-embedding-3-small for openai provider."""
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.API_PROVIDER = PROVIDER_OPENAI
            mock_settings.OPENAI_API_KEY = "sk-openai-key"
            mock_settings.AI_BUILDER_TOKEN = ""
            mock_settings.AI_BUILDER_BASE_URL = ""
            mock_settings.CHROMA_DB_PATH = CHROMA_PATH_DEFAULT

            importlib.reload(_vs)

            call_args_list = _mock_embedding_functions.OpenAIEmbeddingFunction.call_args_list

        last_call_kwargs = call_args_list[-1].kwargs
        self.assertEqual(last_call_kwargs["model_name"], EMBEDDING_MODEL)

    def test_ai_builders_provider_uses_builder_token(self) -> None:
        """API_PROVIDER=ai_builders → embedding function uses AI_BUILDER_TOKEN."""
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.API_PROVIDER = PROVIDER_AI_BUILDERS
            mock_settings.AI_BUILDER_TOKEN = _SK_BUILDER_EMBED
            mock_settings.AI_BUILDER_BASE_URL = AI_BUILDER_BASE_URL
            mock_settings.OPENAI_API_KEY = ""
            mock_settings.CHROMA_DB_PATH = CHROMA_PATH_BUILDER

            importlib.reload(_vs)

            call_args_list = _mock_embedding_functions.OpenAIEmbeddingFunction.call_args_list

        last_call_kwargs = call_args_list[-1].kwargs
        self.assertEqual(last_call_kwargs["api_key"], _SK_BUILDER_EMBED)

    def test_ai_builders_provider_uses_base_url(self) -> None:
        """API_PROVIDER=ai_builders → embedding function includes api_base."""
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.API_PROVIDER = PROVIDER_AI_BUILDERS
            mock_settings.AI_BUILDER_TOKEN = _SK_BUILDER_TOKEN
            mock_settings.AI_BUILDER_BASE_URL = AI_BUILDER_BASE_URL
            mock_settings.OPENAI_API_KEY = ""
            mock_settings.CHROMA_DB_PATH = CHROMA_PATH_BUILDER

            importlib.reload(_vs)

            call_args_list = _mock_embedding_functions.OpenAIEmbeddingFunction.call_args_list

        last_call_kwargs = call_args_list[-1].kwargs
        self.assertIn("api_base", last_call_kwargs)
        self.assertEqual(last_call_kwargs["api_base"], AI_BUILDER_BASE_URL)

    def test_ai_builders_provider_uses_correct_model_name(self) -> None:
        """The embedding model must be text-embedding-3-small for ai_builders provider."""
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.API_PROVIDER = PROVIDER_AI_BUILDERS
            mock_settings.AI_BUILDER_TOKEN = _SK_BUILDER_SIMPLE
            mock_settings.AI_BUILDER_BASE_URL = AI_BUILDER_BASE_URL
            mock_settings.OPENAI_API_KEY = ""
            mock_settings.CHROMA_DB_PATH = CHROMA_PATH_DEFAULT

            importlib.reload(_vs)

            call_args_list = _mock_embedding_functions.OpenAIEmbeddingFunction.call_args_list

        last_call_kwargs = call_args_list[-1].kwargs
        self.assertEqual(last_call_kwargs["model_name"], EMBEDDING_MODEL)


if __name__ == "__main__":
    unittest.main()
