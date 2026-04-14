from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du dépôt (visit-corsica-chatbot/), pour .env et dossiers data/ quel que soit le cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Si ANTHROPIC_API_KEY est défini, Claude est utilisé en priorité pour la génération.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"

    cors_origins: str = "*"

    pages_jsonl: Path = Path("data/raw/pages_fr.jsonl")
    chroma_path: Path = Path("data/chroma")

    rag_top_k: int = 6
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chroma_collection: str = "visit_corsica_fr"

    # Recherche web (DuckDuckGo + extraction) — toujours en complément de l'index
    web_search_enabled: bool = True
    web_max_ddg: int = 4
    web_max_fetch: int = 2
    web_fetch_timeout: float = 5.0

    # Seuil de distance Chroma : au-delà, le chunk est considéré hors-sujet et ignoré
    # (cosine distance : 0 = identique, 2 = opposé ; ~0.8 est déjà assez éloigné)
    rag_max_distance: float = 0.85

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> Settings:
        if not self.chroma_path.is_absolute():
            object.__setattr__(self, "chroma_path", _PROJECT_ROOT / self.chroma_path)
        if not self.pages_jsonl.is_absolute():
            object.__setattr__(self, "pages_jsonl", _PROJECT_ROOT / self.pages_jsonl)
        return self


def get_settings() -> Settings:
    return Settings()
