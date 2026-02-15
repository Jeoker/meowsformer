import asyncio
import os
import shutil
import subprocess
from unittest.mock import MagicMock, patch
from app.services.audio_processor import extract_basic_features, convert_to_wav
from app.services.transcription_service import transcribe_audio

TEST_AUDIO_FILE = "tests/test_audio_input.wav"
CONVERTED_AUDIO_FILE = "tests/test_audio_output.wav"

def setup_test_files():
    # Generate a 1-second sine wave audio file using ffmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=1",
        "-ar", "44100", # Original sample rate different from target
        "-ac", "2",     # Stereo
        "-c:a", "pcm_s16le",
        TEST_AUDIO_FILE
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"Created test audio file: {TEST_AUDIO_FILE}")

def cleanup_test_files():
    if os.path.exists(TEST_AUDIO_FILE):
        os.remove(TEST_AUDIO_FILE)
    if os.path.exists(CONVERTED_AUDIO_FILE):
        os.remove(CONVERTED_AUDIO_FILE)
    print("Cleaned up test files.")

async def test_audio_processing():
    print("\n--- Testing Audio Processing ---")
    
    # Test 1: Feature Extraction
    features = extract_basic_features(TEST_AUDIO_FILE)
    print(f"Extracted features: {features}")
    
    assert features["duration_seconds"] >= 0.9, "Duration should be around 1.0s"
    assert features["rms_amplitude"] > 0, "RMS amplitude should be positive"
    assert features["mean_volume_db"] != -99.9, "Volume should be detected"
    print("✅ Feature extraction passed.")

    # Test 2: Conversion
    await convert_to_wav(TEST_AUDIO_FILE, CONVERTED_AUDIO_FILE)
    
    assert os.path.exists(CONVERTED_AUDIO_FILE), "Converted file should exist"
    
    # Verify properties of converted file
    converted_features = extract_basic_features(CONVERTED_AUDIO_FILE)
    # Check if sample rate is 16000 (we can't easily check SR with our extract function, but we can check if it works)
    # We can use ffprobe manually to check SR
    probe_cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "a:0", 
        "-show_entries", "stream=sample_rate", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        CONVERTED_AUDIO_FILE
    ]
    sr = subprocess.check_output(probe_cmd).decode().strip()
    assert sr == "16000", f"Sample rate should be 16000, got {sr}"
    print("✅ Audio conversion passed.")

async def test_transcription_service():
    print("\n--- Testing Transcription Service ---")
    
    # Mock the OpenAI client
    mock_transcription = MagicMock()
    mock_transcription.text = "This is a meow." # The API returns an object with text attribute, or just a string depending on response_format. 
    # In our code: response_format="text" returns a string directly? 
    # Let's check the code in transcription_service.py: 
    # transcription = client.audio.transcriptions.create(..., response_format="text")
    # If response_format is "text", it returns a string.
    
    mock_response = "This is a meow."

    with patch("app.services.transcription_service.client.audio.transcriptions.create", return_value=mock_response) as mock_create:
        
        # Test valid transcription
        result = await transcribe_audio(TEST_AUDIO_FILE)
        assert result == "This is a meow."
        print("✅ Transcription passed.")
        
        # Verify it called the API with a file
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args.kwargs['model'] == "whisper-1"
        assert 'file' in call_args.kwargs

async def main():
    try:
        setup_test_files()
        await test_audio_processing()
        await test_transcription_service()
    finally:
        cleanup_test_files()

if __name__ == "__main__":
    asyncio.run(main())
