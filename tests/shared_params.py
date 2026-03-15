"""
Shared test parameter constants for the Meowsformer test suite.
Pure values only — no logic, no app imports.
"""

from __future__ import annotations

import json

# ── API providers ─────────────────────────────────────────────────────────────
PROVIDER_OPENAI      = "openai"
PROVIDER_AI_BUILDERS = "ai_builders"

# ── LLM model defaults (one per provider) ────────────────────────────────────
MODEL_OPENAI_DEFAULT      = "gpt-4o"
MODEL_AI_BUILDERS_DEFAULT = "deepseek"

# ── Embedding model (shared across providers) ─────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"

# ── AI Builders endpoint ──────────────────────────────────────────────────────
AI_BUILDER_BASE_URL = "https://space.ai-builders.com/backend/v1"

# ── ChromaDB test paths ───────────────────────────────────────────────────────
CHROMA_PATH_OPENAI  = "/tmp/chroma_openai"
CHROMA_PATH_BUILDER = "/tmp/chroma_builder"
CHROMA_PATH_DEFAULT = "/tmp/chroma"

# ── Cat sample IDs ────────────────────────────────────────────────────────────
SAMPLE_ID_PRIMARY   = "cat_001"
SAMPLE_ID_SECONDARY = "cat_042"

# ── Audio payloads ────────────────────────────────────────────────────────────
AUDIO_B64_STUB    = "AAAA"        # minimal valid base64 stub for result payloads
AUDIO_B64_DECODED = "dGVzdA=="    # base64("test wav data") — used in decode tests
DUMMY_PCM_BYTES   = b"\x00" * 3200

# ── Match scores ──────────────────────────────────────────────────────────────
MATCH_SCORE_HIGH    = 0.85
MATCH_SCORE_PERFECT = 0.9

# ── Server addresses & route paths ───────────────────────────────────────────
SERVER_BASE_URL = "http://localhost:8000"
WS_PATH         = "/ws/translate"
REST_PATH_V1    = "/api/v1/translate"

# ── Breed preferences ─────────────────────────────────────────────────────────
BREED_DEFAULT = "Maine Coon"
BREED_ALT     = "Ragdoll"

# ── Legacy sound IDs ─────────────────────────────────────────────────────────
SOUND_ID_LEGACY = "purr_01"

# ── Async timing ─────────────────────────────────────────────────────────────
STREAMING_SETTLE_SECS = 0.05

# ── Waveform mock return value ────────────────────────────────────────────────
WAVEFORM_MOCK_RETURN = [0.0] * 48

# ── Minimal WS server message that terminates the receiver loop ───────────────
MINIMAL_RESULT_MSG = json.dumps({"type": "result"})
