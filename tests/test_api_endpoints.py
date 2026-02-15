import sys
from unittest.mock import MagicMock

# --- CRITICAL: Mock chromadb BEFORE importing app modules ---
mock_chromadb = MagicMock()
sys.modules["chromadb"] = mock_chromadb
sys.modules["chromadb.utils"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = MagicMock()

import unittest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Now we can safely import the router
from app.api.endpoints import router
from app.schemas.translation import CatTranslationResponse

# Setup FastAPI test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)

class TestAPIEndpoints(unittest.TestCase):

    @patch("app.api.endpoints.analyze_intention")
    @patch("app.api.endpoints.retrieve_context")
    @patch("app.api.endpoints.transcribe_audio", new_callable=AsyncMock)
    @patch("app.api.endpoints.extract_basic_features")
    def test_translate_endpoint(self, mock_extract, mock_transcribe, mock_retrieve, mock_analyze):
        # 1. Setup Mocks
        
        # Audio features
        mock_extract.return_value = {
            "duration_seconds": 1.5,
            "mean_volume_db": -20.0,
            "rms_amplitude": 0.1
        }
        
        # Transcription (Async Mock)
        mock_transcribe.return_value = "Meow meow"
        
        # RAG Context
        mock_retrieve.return_value = "Mock RAG Context"
        
        # LLM Response
        mock_response = CatTranslationResponse(
            sound_id="purr_happy_01",
            pitch_adjust=1.0,
            human_interpretation="I'm hungry!",
            emotion_category="Hungry",
            behavior_note="Short meow indicating demand."
        )
        mock_analyze.return_value = mock_response

        # 2. Prepare test file
        test_file_content = b"fake audio content"
        files = {"file": ("test_audio.wav", test_file_content, "audio/wav")}

        # 3. Call Endpoint
        response = client.post("/translate", files=files)
        
        # 4. Assertions
        if response.status_code != 200:
            print(f"Response Error: {response.text}")
            
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        
        self.assertEqual(json_response["sound_id"], "purr_happy_01")
        self.assertEqual(json_response["emotion_category"], "Hungry")
        
        # Verify call chain
        mock_extract.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_analyze.assert_called_once()
        
        print("✅ API Endpoint Test Passed")

if __name__ == "__main__":
    unittest.main()
