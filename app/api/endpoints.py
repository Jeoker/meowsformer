import os
import tempfile
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from loguru import logger
from typing import Dict, Any, Optional

from app.schemas.translation import CatTranslationResponse, MeowSynthesisResponse
from app.services.audio_processor import extract_basic_features, convert_to_wav
from app.services.transcription_service import transcribe_audio
from app.services.rag_service import retrieve_context
from app.services.llm_service import analyze_intention
from app.services.synthesis_service import synthesize_and_describe

router = APIRouter()


@router.post("/translate", response_model=CatTranslationResponse)
async def translate_cat_sound(file: UploadFile = File(...)):
    """
    Translate a cat's meow into human language based on acoustic features
    and biological context.

    Process Flow (Phase 0 — analysis only):
    1.  Save uploaded audio file temporarily.
    2.  Extract acoustic features (Duration, RMS Volume) using ffmpeg/ffprobe.
    3.  Transcribe audio to text using OpenAI Whisper API.
    4.  Retrieve relevant scientific context from ChromaDB (RAG).
    5.  Analyze intention using LLM (GPT-4o) with combined inputs.
    6.  Return structured JSON response.
    """

    temp_file_path = None

    try:
        # 1. Save uploaded file
        suffix = os.path.splitext(file.filename)[1] if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"Received file: {file.filename}, saved to {temp_file_path}")

        # 2. Extract audio features
        loop = asyncio.get_event_loop()
        audio_features = await loop.run_in_executor(
            None, extract_basic_features, temp_file_path
        )
        logger.info(f"Extracted features: {audio_features}")

        # 3. Transcribe audio
        transcript_text = await transcribe_audio(temp_file_path)

        # Determine query for RAG based on transcription
        if transcript_text and len(transcript_text.strip()) > 2:
            rag_query = transcript_text
            logger.info(f"Transcription: '{transcript_text}'")
        else:
            rag_query = "generic cat meow"
            logger.info("Transcription empty/short, using generic query for RAG.")

        # 4. Retrieve scientific context (RAG)
        rag_context = retrieve_context(rag_query, n_results=3)
        logger.info(f"Retrieved RAG context (first 50 chars): {rag_context[:50]}...")

        # 5. Analyze intention (LLM)
        result = analyze_intention(
            text=transcript_text if transcript_text else "Unknown sound",
            audio_features=audio_features,
            rag_context=rag_context,
        )
        logger.info(f"LLM Analysis result: {result}")

        return result

    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error processing translation request: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {str(e)}"
        )

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to delete temp file {temp_file_path}: {cleanup_error}"
                )


@router.post("/v1/translate", response_model=MeowSynthesisResponse)
async def translate_and_synthesize(
    file: UploadFile = File(...),
    breed: str = Query(
        default="Default",
        description="Target cat breed for synthesis (e.g. 'Maine Coon', 'Siamese', 'Kitten').",
    ),
    output_sr: int = Query(
        default=16000,
        description="Output audio sample rate in Hz. Must be 16000 or 44100.",
    ),
):
    """
    Full pipeline: analyse human speech → synthesise cat vocalisation (Phase 3).

    End-to-end flow:
    1.  Upload & save audio file.
    2.  Extract acoustic features (FFmpeg).
    3.  Transcribe via Whisper.
    4.  RAG context retrieval.
    5.  LLM intention analysis (GPT-4o).
    6.  **DSP Synthesis** — map emotion → intent → VA space → nearest-neighbour
        audio retrieval → PSOLA prosody transform.
    7.  **Preview Description** — NatureLM-audio-style confidence caption.
    8.  Return JSON with base64 WAV audio + preview description + metadata.

    The ``audio_base64`` field contains a WAV file that the frontend should
    present in a preview player. The user must listen and click "Confirm"
    before the audio is delivered to the recipient.
    """

    temp_file_path = None

    try:
        # ── 1. Save uploaded file ────────────────────────────────────
        suffix = os.path.splitext(file.filename)[1] if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"[v1] Received file: {file.filename}, saved to {temp_file_path}")

        # ── 2. Extract audio features ────────────────────────────────
        loop = asyncio.get_event_loop()
        audio_features = await loop.run_in_executor(
            None, extract_basic_features, temp_file_path
        )
        logger.info(f"[v1] Extracted features: {audio_features}")

        # ── 3. Transcribe audio ──────────────────────────────────────
        transcript_text = await transcribe_audio(temp_file_path)

        if transcript_text and len(transcript_text.strip()) > 2:
            rag_query = transcript_text
            logger.info(f"[v1] Transcription: '{transcript_text}'")
        else:
            rag_query = "generic cat meow"
            logger.info("[v1] Transcription empty/short, using generic query for RAG.")

        # ── 4. RAG context ───────────────────────────────────────────
        rag_context = retrieve_context(rag_query, n_results=3)

        # ── 5. LLM analysis ─────────────────────────────────────────
        llm_result: CatTranslationResponse = analyze_intention(
            text=transcript_text if transcript_text else "Unknown sound",
            audio_features=audio_features,
            rag_context=rag_context,
        )
        logger.info(f"[v1] LLM result: {llm_result.emotion_category}")

        # ── 6–8. DSP synthesis + description + encoding ──────────────
        synthesis_response = await synthesize_and_describe(
            llm_result=llm_result,
            breed=breed,
            output_sr=output_sr,
        )

        if synthesis_response.synthesis_ok:
            logger.success("[v1] Full pipeline completed successfully")
        else:
            logger.warning(
                "[v1] LLM analysis succeeded but DSP synthesis was skipped/failed"
            )

        return synthesis_response

    except HTTPException as he:
        logger.error(f"[v1] HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"[v1] Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {str(e)}"
        )

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"[v1] Cleaned up temp file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(
                    f"[v1] Failed to delete temp file {temp_file_path}: {cleanup_error}"
                )
