import asyncio
import subprocess
import os
import json
import math
import re
from typing import Dict, Any, Tuple
from fastapi import HTTPException
from loguru import logger

async def convert_to_wav(input_path: str, output_path: str) -> None:
    """
    Converts audio file to 16k mono wav format using ffmpeg.
    
    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the converted wav file.
        
    Raises:
        HTTPException: If conversion fails.
    """
    command = [
        "ffmpeg",
        "-y", # Overwrite output file
        "-i", input_path,
        "-ar", "16000", # Set sample rate to 16000Hz
        "-ac", "1", # Set audio channels to mono
        "-c:a", "pcm_s16le", # PCM 16-bit little-endian
        output_path
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
            logger.error(f"FFmpeg error: {error_msg}")
            raise HTTPException(status_code=500, detail="Audio conversion failed")
            
    except Exception as e:
        logger.error(f"Error during audio conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio conversion error: {str(e)}")

def get_audio_duration(file_path: str) -> float:
    """Helper to get audio duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode().strip()
        return float(output) if output else 0.0
    except Exception as e:
        logger.warning(f"Failed to get duration: {e}")
        return 0.0

def get_audio_volume(file_path: str) -> Tuple[float, float]:
    """Helper to get mean volume (dB) and calculate RMS amplitude."""
    # Using volumedetect filter
    cmd = [
        "ffmpeg",
        "-i", file_path,
        "-filter:a", "volumedetect",
        "-f", "null",
        "/dev/null"
    ]
    try:
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        _, stderr = process.communicate()
        stderr_output = stderr.decode()
        
        # Parse mean_volume: -25.5 dB
        match = re.search(r"mean_volume:\s+([-\d.]+)\s+dB", stderr_output)
        if match:
            mean_volume_db = float(match.group(1))
            # RMS amplitude approximation: 10^(dB/20)
            rms_amplitude = math.pow(10, mean_volume_db / 20)
            return mean_volume_db, rms_amplitude
    except Exception as e:
        logger.warning(f"Failed to extract volume: {e}")
        
    return -99.9, 0.0

def extract_basic_features(file_path: str) -> Dict[str, Any]:
    """
    Extracts basic audio features like duration and RMS volume.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        Dict[str, Any]: Dictionary containing duration (seconds) and volume (RMS).
    """
    duration = get_audio_duration(file_path)
    mean_volume_db, rms_amplitude = get_audio_volume(file_path)

    return {
        "duration_seconds": duration,
        "mean_volume_db": mean_volume_db,
        "rms_amplitude": rms_amplitude
    }
