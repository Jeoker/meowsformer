from unittest.mock import MagicMock, patch
from app.services.llm_service import analyze_intention
from app.schemas.translation import CatTranslationResponse

def test_analyze_intention_mock():
    # Mock return value
    mock_response = CatTranslationResponse(
        sound_id="purr_happy_01",
        pitch_adjust=1.0,
        human_interpretation="I'm so content right now.",
        emotion_category="Happy",
        behavior_note="Relaxed purring indicates contentment."
    )

    # Patch the client.chat.completions.create method
    with patch("app.services.llm_service.client.chat.completions.create", return_value=mock_response) as mock_create:
        
        result = analyze_intention(
            text="cat looks happy",
            audio_features={"pitch": 440},
            rag_context="Purring usually means happiness."
        )

        assert result == mock_response
        assert result.sound_id == "purr_happy_01"
        assert result.emotion_category == "Happy"
        
        # Verify the call arguments
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["response_model"] == CatTranslationResponse
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        
        print("analyze_intention test passed!")

if __name__ == "__main__":
    test_analyze_intention_mock()
