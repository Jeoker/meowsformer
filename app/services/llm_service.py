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
    Map transcribed owner speech (plus acoustics and RAG) to synthesis-oriented
    cat vocalisation parameters — semantic translation, not a reactive reply.
    
    Args:
        text (str): Transcribed owner speech.
        audio_features (dict): Extracted audio features.
        rag_context (str): Retrieved context from the vector database.
        
    Returns:
        CatTranslationResponse: Structured response containing translation details.
    """
    
    system_prompt = (
        "你是一个精通猫咪生物声学的翻译官。根据主人的口述文本、音频特征与科学上下文，"
        "将人话的含义映射到应合成的猫叫声参数；你是在做语义翻译，而不是推断猫听完话后的现场反应。"
    )
    
    user_prompt = f"""
    主人口述（转录）: {text}
    
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
