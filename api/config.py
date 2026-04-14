from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    cors_origins: str = "*"

    pages_jsonl: Path = Path("data/raw/pages_fr.jsonl")

    # Chroma Cloud
    chroma_host: str = "api.trychroma.com"
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""
    chroma_collection: str = "visit_corsica_fr"

    rag_top_k: int = 6
    rag_max_distance: float = 0.85

    # Recherche web (DuckDuckGo + extraction)
    web_search_enabled: bool = True
    web_max_ddg: int = 4
    web_max_fetch: int = 2
    web_fetch_timeout: float = 5.0

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> Settings:
        if not self.pages_jsonl.is_absolute():
            object.__setattr__(self, "pages_jsonl", _PROJECT_ROOT / self.pages_jsonl)
        return self


def get_settings() -> Settings:
    return Settings()
