from typing import Dict, Any, Optional
import json

import instructor
from app.core.api_client import get_instructor_client
from app.core.config import settings
from app.schemas.translation import CatTranslationResponse

_client: Optional[instructor.Instructor] = None


def _get_client() -> instructor.Instructor:
    global _client
    if _client is None:
        _client = get_instructor_client()
    return _client


def analyze_intention(text: str, audio_features: Dict[str, Any], rag_context: str) -> CatTranslationResponse:
    """
    Analyze the intention based on text, audio features, and RAG context to determine the cat's response.
    
    Args:
        text (str): The user's input text or intent.
        audio_features (dict): Extracted audio features.
        rag_context (str): Retrieved context from the vector database.
        
    Returns:
        CatTranslationResponse: Structured response containing translation details.
    """
    
    system_prompt = (
        "你是一个精通猫咪生物声学的翻译官。根据用户的文本意图、音频特征和提供的科学上下文，决定猫的反应。"
    )
    
    user_prompt = f"""
    用户文本意图: {text}
    
    音频特征:
    {json.dumps(audio_features, ensure_ascii=False, indent=2)}
    
    科学上下文 (RAG):
    {rag_context}
    
    请分析并在 JSON 中返回结果，确保 pitch_adjust 在 0.8 到 1.5 之间。
    """

    response = _get_client().chat.completions.create(
        model=settings.LLM_MODEL,
        response_model=CatTranslationResponse,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    
    return response
