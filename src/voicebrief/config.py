"""Typed application settings.

Everything is env-driven so the same image runs locally and in deploy. Host ports
deliberately avoid the defaults (5432/6333/6379/9000) so VoiceBrief can coexist with
other stacks on a developer machine.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "ollama", "echo"]
TTSEngine = Literal["kokoro", "piper", "null"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core
    env: str = Field(default="local", alias="VB_ENV")
    log_level: str = Field(default="INFO", alias="VB_LOG_LEVEL")
    timezone: str = Field(default="Asia/Kolkata", alias="VB_TIMEZONE")

    # Datastores
    postgres_dsn: str = Field(
        default="postgresql+psycopg://voicebrief:voicebrief@localhost:55432/voicebrief",
        alias="VB_POSTGRES_DSN",
    )
    qdrant_url: str = Field(default="http://localhost:56333", alias="VB_QDRANT_URL")
    redis_url: str = Field(default="redis://localhost:56379/0", alias="VB_REDIS_URL")

    s3_endpoint: str = Field(default="http://localhost:59000", alias="VB_S3_ENDPOINT")
    s3_access_key: str = Field(default="voicebrief", alias="VB_S3_ACCESS_KEY")
    s3_secret_key: SecretStr = Field(default=SecretStr("voicebrief"), alias="VB_S3_SECRET_KEY")
    s3_bucket: str = Field(default="voicebrief-media", alias="VB_S3_BUCKET")

    # LLM
    llm_provider: LLMProvider = Field(default="anthropic", alias="VB_LLM_PROVIDER")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_script_model: str = Field(default="claude-sonnet-5", alias="VB_LLM_SCRIPT_MODEL")
    llm_utility_model: str = Field(
        default="claude-haiku-4-5-20251001", alias="VB_LLM_UTILITY_MODEL"
    )
    ollama_url: str = Field(default="http://localhost:11434", alias="VB_OLLAMA_URL")
    ollama_script_model: str = Field(
        default="qwen2.5:7b-instruct-q4_K_M", alias="VB_OLLAMA_SCRIPT_MODEL"
    )
    daily_budget_inr: float = Field(default=50.0, alias="VB_DAILY_BUDGET_INR")

    # Embeddings
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="VB_EMBED_MODEL")
    embed_dim: int = Field(default=384, alias="VB_EMBED_DIM")

    # TTS
    tts_engine: TTSEngine = Field(default="kokoro", alias="VB_TTS_ENGINE")
    tts_voice_en: str = Field(default="af_heart", alias="VB_TTS_VOICE_EN")
    tts_voice_hi: str = Field(default="hf_alpha", alias="VB_TTS_VOICE_HI")

    # Source credentials (all optional; absence degrades, never breaks)
    github_token: SecretStr | None = Field(default=None, alias="GITHUB_TOKEN")
    reddit_client_id: SecretStr | None = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: SecretStr | None = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    youtube_api_key: SecretStr | None = Field(default=None, alias="YOUTUBE_API_KEY")

    # Ingestion guardrails — the budget story in §3 of the PRD lives here.
    max_items_per_run: int = Field(default=1500, alias="VB_MAX_ITEMS_PER_RUN")
    max_items_to_rank: int = Field(default=60, alias="VB_MAX_ITEMS_TO_RANK")
    max_stories_per_episode: int = Field(default=10, alias="VB_MAX_STORIES_PER_EPISODE")

    @property
    def user_agent(self) -> str:
        """Every outbound request identifies itself. Politeness is a design constraint."""
        return "VoiceBrief/0.1 (+https://github.com/ParthBiyani/Voice-Brief)"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
