from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = "sk-placeholder"
    CHROMA_DB_PATH: str = "./db/chroma_db"
    DEBUG_MODE: bool = False

    # JWT Auth
    JWT_SECRET_KEY: str = "change-me-in-production-use-a-long-random-string"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
