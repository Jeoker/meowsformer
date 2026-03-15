import os
import shutil
import tempfile
import asyncio
from typing import TYPE_CHECKING, Optional
from fastapi import HTTPException
from app.services.audio_processor import convert_to_wav
from app.core.api_client import get_openai_client
from loguru import logger

if TYPE_CHECKING:
    from openai import OpenAI

_client: Optional["OpenAI"] = None


def _get_client() -> "OpenAI":
    global _client
    if _client is None:
        _client = get_openai_client()
    return _client


async def transcribe_audio(file_path: str) -> str:
    """
    Transcribes the audio file using OpenAI Whisper API.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        str: Transcribed text.
        
    Raises:
        HTTPException: If transcription fails or file is invalid.
    """
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty")
        
    # OpenAI Whisper file size limit (25MB)
    if file_size > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 25MB)")
    
    # Create temporary directory for conversion
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "temp_audio.wav")
        
        try:
            # Convert audio to ensure compatibility (16k mono wav)
            await convert_to_wav(file_path, temp_file_path)
            
            # Check if converted file exists and is not empty
            if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                 logger.error("Converted audio file is missing or empty")
                 raise HTTPException(status_code=500, detail="Audio conversion failed internally")
                 
            # Read converted audio file
            with open(temp_file_path, "rb") as audio_file:
                transcription = _get_client().audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                
            return transcription

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
