"""
Centralized configuration for the Sustainability SME system.
Loads from environment variables / .env file with sensible defaults.
Also initializes LangSmith tracing environment.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ── LangSmith tracing setup (must be set before any langchain imports) ──
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "sustainability-sme"))
os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Ollama ──
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model_light: str = Field(default="gpt-oss:120b-cloud")
    ollama_model_heavy: str = Field(default="gpt-oss:120b-cloud")
    ollama_model_vision: str = Field(default="gpt-oss:120b-cloud")
    ollama_model_embed: str = Field(default="mxbai-embed-large")

    # ── Storage Paths ──
    faiss_persist_dir: str = Field(default="./data/faiss")
    sqlite_table_db: str = Field(default="./data/tables.db")
    checkpoint_db: str = Field(default="./data/checkpoints.db")
    data_dir: str = Field(default="./data")
    pdf_dir: str = Field(default="./data/pdfs")
    image_dir: str = Field(default="./data/images")

    # ── Chunking ──
    chunk_size_child: int = Field(default=400)
    chunk_size_parent: int = Field(default=1600)
    chunk_overlap: int = Field(default=50)

    # ── Retrieval ──
    retrieval_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=5)
    rerank_enabled: bool = Field(default=False)

    # ── RAPTOR ──
    raptor_cluster_size: int = Field(default=10)
    raptor_max_levels: int = Field(default=3)

    # ── Memory ──
    short_term_max_messages: int = Field(default=20)
    long_term_max_items: int = Field(default=200)

    # ── Context window budgeting ──
    max_context_tokens: int = Field(default=131072)
    output_tokens_reserved: int = Field(default=4096)
    prompt_overhead_tokens: int = Field(default=3000)
    ctx_share_text: float = Field(default=0.75)
    ctx_share_table: float = Field(default=0.12)
    ctx_share_map_reduce: float = Field(default=0.10)
    ctx_share_memory: float = Field(default=0.03)
    chars_per_token: float = Field(default=4.0)

    # ── Hallucination defense ──
    citation_coverage_threshold: float = Field(default=0.5)

    # ── Attribution verifier ──
    attribution_enabled: bool = Field(default=True)
    attribution_max_facts: int = Field(default=40)
    attribution_entity_fuzzy_ratio: float = Field(default=0.85)
    attribution_value_tolerance: float = Field(default=0.005)
    attribution_proximity_tokens: int = Field(default=15)

    # ── LangSmith ──
    langchain_tracing_v2: str = Field(default="false")
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="sustainability-sme")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore unexpected env vars

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.pdf_dir,
            self.image_dir,
            self.faiss_persist_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @property
    def is_langsmith_enabled(self) -> bool:
        """Check if LangSmith tracing is properly configured."""
        return (
            self.langchain_tracing_v2.lower() == "true"
            and len(self.langchain_api_key) > 0
            and self.langchain_api_key != "ls_your_key_here"
        )


# Singleton settings instance
settings = Settings()
settings.ensure_directories()
