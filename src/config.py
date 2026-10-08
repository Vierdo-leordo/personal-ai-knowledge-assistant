"""Central configuration, loaded from environment variables / a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        # Streamlit Community Cloud exposes secrets through st.secrets
        try:
            import streamlit as st  # noqa: WPS433

            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:
            pass
        return default
    return value.strip()


@dataclass
class Settings:
    # --- LLM ----------------------------------------------------------------
    # "auto" picks the first provider that has an API key, else "offline".
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto").lower())
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-3.8-flash"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    # How long to wait (seconds) when a free-tier rate limit says "retry in Xs"
    max_rate_limit_wait: int = field(default_factory=lambda: int(_env("MAX_RATE_LIMIT_WAIT", "65")))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.2")))

    # --- Embeddings ---------------------------------------------------------
    # "auto" -> sentence-transformers if installed, else gemini/openai if a key
    # exists, else the built-in hashing embedder (works fully offline).
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "auto").lower())
    sentence_transformer_model: str = field(
        default_factory=lambda: _env("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    )
    gemini_embedding_model: str = field(default_factory=lambda: _env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))
    openai_embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- Vector store -------------------------------------------------------
    # "auto" -> ChromaDB if installed, else the built-in NumPy store.
    vector_store: str = field(default_factory=lambda: _env("VECTOR_STORE", "auto").lower())

    # --- Chunking / retrieval ----------------------------------------------
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "900")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    history_turns: int = field(default_factory=lambda: int(_env("HISTORY_TURNS", "4")))

    # --- Paths ----------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))))
    vector_dir: Path = field(default_factory=lambda: Path(_env("VECTOR_DIR", str(PROJECT_ROOT / "vector_store"))))
    outputs_dir: Path = field(default_factory=lambda: Path(_env("OUTPUTS_DIR", str(PROJECT_ROOT / "outputs"))))

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def db_path(self) -> Path:
        return self.metadata_dir / "assistant.db"

    def ensure_dirs(self) -> None:
        for p in (self.uploads_dir, self.metadata_dir, self.embeddings_dir, self.vector_dir, self.outputs_dir):
            p.mkdir(parents=True, exist_ok=True)

    def provider_has_key(self, provider: str) -> bool:
        return bool(
            {
                "groq": self.groq_api_key,
                "gemini": self.gemini_api_key,
                "openai": self.openai_api_key,
                "anthropic": self.anthropic_api_key,
            }.get(provider)
        )

    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        for provider in ("groq", "gemini", "openai", "anthropic"):
            if self.provider_has_key(provider):
                return provider
        return "offline"


CATEGORIES = ["General", "Education", "Research", "Projects", "Notes"]


def get_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    s.ensure_dirs()
    return s
