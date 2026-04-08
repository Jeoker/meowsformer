from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = "sk-placeholder"
    CHROMA_DB_PATH: str = "./db/chroma_db"
    DEBUG_MODE: bool = False

    LLM_MODEL: str = "gpt-4o"

    # Whisper ISO-639-1 code (e.g. zh, en). Empty = model auto-detect (can mis-detect short clips).
    WHISPER_LANGUAGE: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
