"""
Embedding model wrapper using Ollama.
Provides batch embedding and caching support.
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_ollama import OllamaEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> OllamaEmbeddings:
    """Get the singleton embedding model instance."""
    logger.info("Initializing embedding model — model=%s", settings.ollama_model_embed)
    return OllamaEmbeddings(
        model=settings.ollama_model_embed,
        base_url=settings.ollama_base_url,
    )


def embed_text(text: str) -> list[float]:
    """Embed a single text string."""
    model = get_embedding_model()
    return model.embed_query(text)


def embed_texts(texts: list[str], max_chars: int = 1800) -> list[list[float]]:
    """
    Embed a batch of text strings.

    Truncates each input to max_chars (default 1800 ≈ 450 tokens) so that
    long table representations / image captions never exceed the embedding
    model's context window (mxbai-embed-large defaults to 512 tokens).
    """
    if not texts:
        return []
    model = get_embedding_model()

    safe_texts = [t if len(t) <= max_chars else t[:max_chars] for t in texts]
    truncated = sum(1 for a, b in zip(texts, safe_texts) if a != b)
    logger.debug(
        "Embedding batch of %d texts (truncated %d to %d chars)",
        len(texts), truncated, max_chars,
    )
    return model.embed_documents(safe_texts)
