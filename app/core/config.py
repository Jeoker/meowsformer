from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = "sk-placeholder"
    CHROMA_DB_PATH: str = "./db/chroma_db"
    DEBUG_MODE: bool = False

    # API provider selection — set API_PROVIDER=ai_builders in .env to switch
    API_PROVIDER: Literal["openai", "ai_builders"] = "openai"
    AI_BUILDER_TOKEN: str = ""
    AI_BUILDER_BASE_URL: str = "https://space.ai-builders.com/backend/v1"
    # Empty = auto: openai → gpt-4o, ai_builders → deepseek. Set in .env to override.
    LLM_MODEL: str = ""

    # Whisper ISO-639-1 code (e.g. zh, en). Empty = model auto-detect (can mis-detect short clips).
    WHISPER_LANGUAGE: str = ""

    @model_validator(mode="after")
    def _set_llm_model_default(self) -> "Settings":
        if not self.LLM_MODEL:
            default = "gpt-4o" if self.API_PROVIDER == "openai" else "deepseek"
            object.__setattr__(self, "LLM_MODEL", default)
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
