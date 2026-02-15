# Meowsformer - Backend Project Documentation

## 1. Project Overview
Meowsformer is a FastAPI-based backend service that translates human speech into realistic cat vocalisations. It combines speech-to-text (Whisper), LLM-based intent analysis, bioacoustic RAG, and a DSP audio engine to synthesise breed-appropriate cat meows from a curated corpus.

**Current Status:** Phase 1 (Data Acquisition), Phase 2 (DSP Engine), and Phase 3 (Integration & UI) complete. Core API logic implemented, all unit tests passing (140 total), API endpoints functional with end-to-end synthesis pipeline.

### Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Core API — FastAPI endpoints, Whisper transcription, LLM analysis, RAG | Done |
| **Phase 1** | Data Acquisition — Zenodo corpus download, metadata parsing, registry index | Done |
| **Phase 2** | DSP Engine — VA mapping, audio retrieval, PSOLA prosody transform | Done |
| **Phase 3** | Integration — Wire DSP engine into API pipeline, end-to-end flow, UI preview | Done |
| **Phase 4** | Deployment — Dockerise, CI/CD, production hardening | Pending |

## 2. Tech Stack & Dependencies
*   **Language:** Python 3.10+ (Recommended: 3.10 or 3.11 for best compatibility)
*   **Web Framework:** FastAPI + Uvicorn
*   **Data Validation:** Pydantic V2
*   **AI/LLM:** OpenAI API (GPT-4o, Whisper V3), `instructor` (structured outputs)
*   **Vector Database:** ChromaDB (local persistent storage)
*   **Audio Processing (Phase 0):** FFmpeg (via `subprocess`), `python-multipart`
*   **Audio DSP Engine (Phase 2):** `librosa` (f0 estimation, audio I/O), `pytsmod` (WSOLA time-stretching), `soundfile` (WAV read/write), `scipy`, `numpy`
*   **Data Acquisition (Phase 1):** `zenodo-get` (corpus download from Zenodo)
*   **Environment Management:** `python-dotenv`, `pydantic-settings`
*   **Logging:** `loguru`
*   **Testing:** `unittest`

## 3. Directory Structure
```text
/
├── app/                             # Phase 0 — Core API application
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py             # Main API orchestration (POST /translate)
│   ├── core/
│   │   └── config.py                # Settings management (env vars)
│   ├── db/
│   │   └── vector_store.py          # ChromaDB client initialisation
│   ├── schemas/
│   │   └── translation.py           # Pydantic models (CatTranslationResponse)
│   └── services/
│       ├── audio_processor.py       # FFmpeg wrapper (convert, extract features)
│       ├── llm_service.py           # OpenAI GPT-4o integration (Instructor)
│       ├── rag_service.py           # Knowledge retrieval logic
│       └── transcription_service.py # OpenAI Whisper integration
├── src/                             # Phase 2 & 3 — DSP Engine & Integration
│   ├── __init__.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── dsp_processor.py         # VA mapping, audio retrieval, PSOLA transform
│   │   └── description_generator.py # NatureLM-audio-style confidence descriptions (Phase 3)
│   └── ui/                          # Phase 3 — Frontend preview components
│       ├── vite.config.ts           # Vite dev server config (proxy to FastAPI)
│       └── src/
│           ├── types/
│           │   └── api.ts           # TypeScript API type definitions
│           ├── hooks/
│           │   └── useAudioPreview.ts # Audio preview playback hook
│           └── components/
│               ├── MeowPreviewPlayer.tsx  # Preview player + confirm UI
│               └── MeowPreviewPlayer.css  # Player styles
├── tools/                           # Phase 1 — Data Acquisition
│   ├── __init__.py
│   └── download_datasets.py         # Zenodo download, metadata parsing, registry builder
├── assets/
│   ├── audio_db/
│   │   └── registry.json            # Metadata index (483 CatMeows samples, VA annotations)
│   └── raw_data/                    # Downloaded audio corpora (git-ignored)
│       ├── catmeows/                #   CatMeows dataset (Zenodo 10.5281/zenodo.4007940)
│       └── meowsic/                 #   Meowsic dataset  (Zenodo 10.5281/zenodo.3245999)
├── tests/                           # Unit & integration tests
│   ├── __init__.py
│   ├── test_api_endpoints.py        # API endpoint tests
│   ├── test_audio_services.py       # Audio service tests
│   ├── test_download_datasets.py    # Data acquisition pipeline tests
│   ├── test_dsp_processor.py        # DSP engine tests (45 cases)
│   ├── test_description_generator.py # Description generator tests (31 cases) — Phase 3
│   ├── test_synthesis_service.py    # Synthesis integration tests (15 cases) — Phase 3
│   ├── test_llm_service.py          # LLM service tests
│   └── test_rag_service.py          # RAG service tests
├── .env                             # Environment variables (not committed)
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
└── PROJECT_STATUS.md                # This file
```

## 4. Key Modules & Functionality

### 4.1. Audio Processing (`app/services/audio_processor.py`) — Phase 0
*   **`convert_to_wav(input_path, output_path)`**: Asynchronously converts uploaded audio to 16kHz mono WAV format using FFmpeg.
*   **`extract_basic_features(file_path)`**: Extracts `duration_seconds` and `rms_amplitude` (volume) using `ffprobe`/`ffmpeg`.

### 4.2. Transcription (`app/services/transcription_service.py`) — Phase 0
*   **`transcribe_audio(file_path)`**: Sends processed audio to OpenAI Whisper API to get a text transcription.

### 4.3. RAG (`app/services/rag_service.py`) — Phase 0
*   **`initialize_knowledge_base()`**: Populates ChromaDB with scientific descriptions of cat sounds.
*   **`retrieve_context(query_text)`**: Retrieves the top 3 relevant biological contexts.

### 4.4. LLM Analysis (`app/services/llm_service.py`) — Phase 0
*   **`analyze_intention(...)`**: Combines transcription, audio features, and RAG context into a prompt. Uses `instructor` for structured JSON output.

### 4.5. API Layer (`app/api/endpoints.py`) — Phase 0
*   **`POST /api/v1/translate`**: Main orchestration endpoint (upload → features → transcribe → RAG → LLM → response).

### 4.6. Data Acquisition Pipeline (`tools/download_datasets.py`) — Phase 1
*   **`download_zenodo_dataset(doi, output_dir)`**: Downloads corpora from Zenodo via `zenodo_get`.
*   **`extract_archives(target_dir)`**: Extracts `.zip` and `.tar.gz` archives.
*   **`parse_catmeows_filename(filename)`**: Parses the CatMeows naming convention (`C_NNNNN_BB_SS_OOOOO_RXX`) into structured metadata (context, breed, sex, cat ID).
*   **`build_registry(catmeows_dir, meowsic_dir)`**: Scans datasets and builds `registry.json` with pre-assigned Valence/Arousal values per context (Food → V=0.5/A=0.9, Isolation → V=−0.8/A=0.7, Brushing → per-individual deterministic hash).
*   **`run_pipeline(skip_download=False)`**: Full pipeline entry point.
*   **Usage:** `python -m tools.download_datasets` (full) or `--skip-download` (index only).

### 4.7. DSP Processing Engine (`src/engine/dsp_processor.py`) — Phase 2

#### 4.7.1. Intent → VA Space Mapping
*   **`map_intent_to_va(intent)`**: Maps 10 human communicative intents to target (Valence, Arousal) coordinates based on Russell's Circumplex Model:

    | Intent | Valence | Arousal | Description |
    |--------|---------|---------|-------------|
    | Affiliative | +0.70 | 0.35 | Friendly greeting |
    | Contentment | +0.80 | 0.15 | Purring / relaxed |
    | Play | +0.60 | 0.85 | Playful chirp |
    | Requesting | +0.30 | 0.75 | Food / attention demand |
    | Solicitation | +0.40 | 0.60 | Gentle asking |
    | Agonistic | −0.80 | 0.90 | Hiss / growl threat |
    | Distress | −0.70 | 0.85 | Pain / fear cry |
    | Frustration | −0.50 | 0.70 | Irritation |
    | Alert | 0.00 | 0.65 | Attention signal |
    | Neutral | 0.00 | 0.40 | Baseline vocalisation |

#### 4.7.2. Dynamic Audio Retrieval
*   **`get_best_match(target_v, target_a, ...)`**: Computes Euclidean distance between target VA vector and all registry samples, returns top-K nearest neighbours. Supports optional `breed_filter` and `context_filter`.

#### 4.7.3. PSOLA Prosody Transform
*   **`apply_prosody_transform(audio_path, target_pitch_shift, duration_factor, ...)`**: Implements PSOLA via time-stretch + resample decomposition:
    1. **f0 estimation** via pYIN (librosa).
    2. **Breed-based f0 adjustment** — 8 breed baselines (e.g. Maine Coon 420 Hz, Kitten 750 Hz, Siamese 620 Hz); 50% blend with explicit pitch shift.
    3. **Arousal-based time modulation** — high arousal compresses duration (urgent), low arousal expands (calm).
    4. **WSOLA time-stretching** (pytsmod) — preserves pitch while changing duration.
    5. **Resampling** — shifts pitch to target while restoring correct duration.
    6. **Arousal envelope shaping** — attack/decay curve controlled by arousal (sharp for urgent, gentle for calm).
    7. **Peak normalisation** — prevents clipping (0.95 ceiling).

#### 4.7.4. End-to-End Synthesis
*   **`synthesize_meow(intent, breed, ...)`**: One-call pipeline: intent → VA mapping → nearest-neighbour retrieval → PSOLA transform → output audio.

### 4.8. Confidence Description Generator (`src/engine/description_generator.py`) — Phase 3

#### 4.8.1. NatureLM-audio-Inspired Captioning
*   **`generate_preview_description(intent, match, target_va, ...)`**: Combines matched sample metadata, intent semantics, and DSP transform parameters into a structured Chinese-language preview description. Maps:
    -   Intent → Chinese semantic label (e.g. `"Requesting"` → `"积极求食"`)
    -   Arousal → descriptive adjective (`"极度紧迫的"`, `"舒缓低沉的"`, etc.)
    -   Valence → emotional colour (`"积极"`, `"消极"`, etc.)
    -   Pitch shift → tonal description
    -   Duration factor → tempo description

#### 4.8.2. Confidence Scoring
*   **`_compute_confidence_score(distance)`**: Exponential decay `exp(-distance)` converting VA Euclidean distance to a `[0, 1]` confidence score.
*   **`_confidence_level_cn(score)`**: Maps score to qualitative Chinese levels: `极高 / 高 / 中等 / 较低 / 低`.

#### 4.8.3. Convenience Wrapper
*   **`generate_description_from_synthesis(intent, match, breed)`**: Auto-derives pitch/duration deltas matching `synthesize_meow` internals — ensures description accurately reflects the applied transforms.

### 4.9. Synthesis Integration Service (`app/services/synthesis_service.py`) — Phase 3

*   **`synthesize_and_describe(llm_result, breed, output_sr)`**: End-to-end Phase 3 pipeline:
    1.  Maps LLM `emotion_category` → bioacoustic intent (Hungry→Requesting, Angry→Agonistic, Happy→Affiliative, Alert→Alert).
    2.  Intent → VA → nearest-neighbour retrieval → PSOLA transform.
    3.  Generates NatureLM-audio-style preview description.
    4.  Encodes output audio as base64 WAV (resampled to 16kHz or 44.1kHz).
    5.  Returns `MeowSynthesisResponse` with Phase 0 fields + audio + description + metadata.
    6.  Degrades gracefully if synthesis fails (returns Phase 0 fields with `synthesis_ok=False`).

### 4.10. API Layer — Phase 3 Endpoint (`app/api/endpoints.py`)

*   **`POST /api/v1/translate`**: Full end-to-end pipeline. Accepts `breed` and `output_sr` query parameters. Returns `MeowSynthesisResponse` with base64 WAV audio, preview description, and synthesis metadata.
*   **Backward compatible**: The original `POST /api/translate` endpoint remains unchanged.

### 4.11. Frontend Preview Components (`src/ui/`) — Phase 3

*   **`MeowPreviewPlayer.tsx`**: React component with:
    -   Audio preview playback (play / pause / progress bar).
    -   NatureLM-audio-style confidence description display.
    -   Mandatory "listen before send" — Confirm button disabled until user has listened.
    -   Confirm / Reject action buttons.
*   **`useAudioPreview.ts`**: React hook managing base64 → Blob → ObjectURL audio lifecycle.
*   **`api.ts`**: TypeScript type definitions mirroring backend Pydantic schemas.

## 5. Setup & Running

### Prerequisites
*   Python 3.10+ installed.
*   FFmpeg installed and available in system PATH.
*   OpenAI API Key.

### Installation
1.  **Clone/Open Project**
2.  **Create Virtual Environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or venv\Scripts\activate # Windows
    ```
3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Environment Variables**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=sk-your-key-here
    CHROMA_DB_PATH=./db/chroma_db
    DEBUG_MODE=True
    ```
5.  **Download Audio Corpora (Phase 1)**
    ```bash
    python -m tools.download_datasets
    # Or index-only (if audio already downloaded):
    python -m tools.download_datasets --skip-download
    ```

### Running the Server
```bash
python main.py
# Or with uvicorn directly:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The API documentation will be available at `http://localhost:8000/docs`.

## 6. Testing
Tests are located in the `tests/` directory. They cover core logic and use mocks/synthetic audio to avoid external API costs.

Run all tests:
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m unittest discover tests
```

### Test Summary

| Test File | Module | Cases | Description |
|-----------|--------|-------|-------------|
| `test_api_endpoints.py` | API | — | `POST /translate` with mocked services |
| `test_audio_services.py` | Audio | — | `extract_basic_features`, `convert_to_wav` (FFmpeg) |
| `test_llm_service.py` | LLM | — | `analyze_intention` with mocked OpenAI |
| `test_rag_service.py` | RAG | — | `initialize_knowledge_base`, `retrieve_context` |
| `test_download_datasets.py` | Data | — | Filename parsing, registry building, download |
| `test_dsp_processor.py` | DSP | 45 | VA mapping, audio retrieval, f0 estimation, PSOLA transform, arousal envelope, breed baselines, time-stretch |
| `test_description_generator.py` | Descriptions | 31 | Intent labels, confidence scoring, descriptor lookup, preview generation, convenience wrapper |
| `test_synthesis_service.py` | Synthesis | 15 | Emotion→intent mapping, base64 encoding, schema conversion, full pipeline (mocked), graceful degradation |

**Note on Compatibility:**
If you encounter `pydantic.v1.errors.ConfigError` related to `chromadb`, ensure you are using Python 3.12. Python 3.14 has compatibility issues with `chromadb`'s Pydantic V1 dependency.

## 7. Next Steps for Development
*   **Phase 4 — Database Integration:** Store translation history in SQLite/PostgreSQL via SQLAlchemy or Tortoise-ORM.
*   **Phase 4 — Frontend Build:** Wire `src/ui/` React components into a full app (npm install, routing, API integration layer).
*   **Phase 4 — Deployment:** Dockerise the application (ensure FFmpeg + librosa dependencies are in the Dockerfile).
*   **Future — Advanced Features:** User accounts, feedback loops (RLHF), multi-language intent support, real-time streaming synthesis.
