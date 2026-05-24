"""
LLM abstraction layer with model routing.
Provides a unified interface to switch between light/heavy/vision models.
All calls go through Ollama. Includes retry logic and per-agent token accounting.
"""

import json
import logging
import re
import time
from contextvars import ContextVar
from typing import Any, Optional, Type
from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Per-agent token accounting ──────────────────────────────────────────────
# A ContextVar so concurrent requests (each in their own asyncio task) don't
# step on each other's counters. Each agent node resets the counter at start
# and reads it at end to attribute tokens to that agent's trace entry.

_token_counter: ContextVar[Optional[dict]] = ContextVar("llm_token_counter", default=None)


def reset_token_counter() -> None:
    """Start a fresh accumulator for the current async/thread context."""
    _token_counter.set({
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    })


def get_token_counter() -> dict:
    """Return a snapshot of the current counter (or an empty zeroed one)."""
    counter = _token_counter.get()
    if counter is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    return dict(counter)


def _record_tokens(message: Any) -> None:
    """Pull usage_metadata off an AIMessage and add it to the current counter."""
    counter = _token_counter.get()
    if counter is None or message is None:
        return
    meta = getattr(message, "usage_metadata", None) or {}
    try:
        counter["prompt_tokens"]     += int(meta.get("input_tokens", 0) or 0)
        counter["completion_tokens"] += int(meta.get("output_tokens", 0) or 0)
        counter["total_tokens"]      += int(meta.get("total_tokens", 0) or 0)
        counter["calls"]             += 1
    except (TypeError, ValueError):
        pass


def _unwrap_structured(d: dict) -> Any:
    """Companion to with_structured_output(include_raw=True): record tokens
    from the raw AIMessage and return just the parsed Pydantic object.

    Legacy helper — schema-less, raises on parsing_error. Newer code paths
    use _unwrap_structured_with_repair which falls back to extracting JSON
    from prose responses (common with gpt-oss:120b-cloud which often emits
    markdown wrappers around its JSON despite the schema constraint)."""
    if not isinstance(d, dict):
        return d
    raw = d.get("raw")
    _record_tokens(raw)
    if d.get("parsing_error"):
        raise d["parsing_error"]
    return d.get("parsed")


# ── Robust structured-output repair ──────────────────────────────────────────
# gpt-oss:120b-cloud frequently violates the json_schema constraint and emits
# markdown prose, code-fenced JSON, or a bare list where a wrapped object was
# expected. Without repair, every structured agent (planner, validator, judge,
# contradiction detector, fact extractor) silently degrades — the validation
# agent's prose gets thrown away, the judge defaults to a low verdict, etc.,
# biasing eval scores downward by 0.2-0.5 composite per question.
#
# This module sits between LangChain's structured-output runnable and the
# caller. On parsing failure it tries: (1) strip code fences, (2) extract the
# first {...} or [...] block via regex, (3) auto-wrap a bare list when the
# schema has exactly one list-typed field. If everything fails it falls back
# to LangChain's original error so callers' existing try/except still works.


def _is_list_type(annotation: Any) -> bool:
    """True if a Pydantic field annotation is list[X] / List[X] / typing.List[X]."""
    s = str(annotation)
    return s.startswith("list[") or s.startswith("List[") or s.startswith("typing.List[")


def _auto_wrap_list(data: list, schema: Type[BaseModel]) -> Optional[dict]:
    """If the model returned a bare JSON list but the schema expects a wrapper
    object with exactly one list-typed field, wrap into that field.

    e.g. model emits [{...}, {...}] for ContradictionReport whose schema is
    {contradictions: list[Contradiction]} → wrap into {"contradictions": [...]}.
    """
    fields = getattr(schema, "model_fields", {}) or {}
    list_fields = [name for name, f in fields.items() if _is_list_type(f.annotation)]
    if len(list_fields) == 1:
        return {list_fields[0]: data}
    # Fallback to common naming conventions
    for candidate in ("items", "results", "data", "records"):
        if candidate in fields and _is_list_type(fields[candidate].annotation):
            return {candidate: data}
    return None


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\{\[][\s\S]*?[\}\]])\s*```")


def _try_repair_json(text: str, schema: Type[BaseModel]) -> Optional[BaseModel]:
    """Extract JSON from a possibly prose-wrapped LLM response and validate
    against ``schema``. Returns None when no recoverable JSON was found —
    callers should fall through to their existing failure path.
    """
    if not text:
        return None

    candidates: list[str] = []

    # 1. Code-fenced JSON (```json {...} ```)
    for m in _CODE_FENCE_RE.finditer(text):
        candidates.append(m.group(1))

    # 2. First {...} block (greedy — covers nested objects)
    m = _JSON_OBJECT_RE.search(text)
    if m:
        candidates.append(m.group(0))

    # 3. First [...] block (for bare-array responses)
    m = _JSON_ARRAY_RE.search(text)
    if m:
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        # Try direct validation first
        try:
            return schema.model_validate(data)
        except Exception:
            pass
        # Bare list → try wrapping
        if isinstance(data, list):
            wrapped = _auto_wrap_list(data, schema)
            if wrapped is not None:
                try:
                    return schema.model_validate(wrapped)
                except Exception:
                    pass
    return None


def _make_unwrap_with_repair(schema: Type[BaseModel]):
    """Closure capturing the schema so the RunnableLambda can repair on the
    fly. Returns a function compatible with the |-pipe API used everywhere."""

    schema_name = schema.__name__

    def _unwrap(d: dict) -> Any:
        if not isinstance(d, dict):
            return d
        raw = d.get("raw")
        _record_tokens(raw)
        parsed = d.get("parsed")
        parsing_error = d.get("parsing_error")

        # Happy path
        if parsed is not None and not parsing_error:
            return parsed

        # Repair attempt
        if raw is not None:
            content = getattr(raw, "content", "") or ""
            repaired = _try_repair_json(content, schema)
            if repaired is not None:
                logger.info(
                    "Structured output repaired for %s — %d chars of prose extracted to valid JSON",
                    schema_name, len(content),
                )
                return repaired

        # Last resort: re-raise the original error so existing try/except
        # paths in callers (e.g. agents/validation.py) keep working.
        if parsing_error:
            raise parsing_error
        return parsed

    return _unwrap

# ── Model routing map ──
MODEL_ROUTING = {
    "light": settings.ollama_model_light,       # Fast: classification, SQL gen, validation
    "heavy": settings.ollama_model_heavy,       # Powerful: reasoning, synthesis
    "vision": settings.ollama_model_vision,     # Multimodal: image captioning (preprocessing)
    "embed": settings.ollama_model_embed,       # Embeddings (handled separately)
}


@lru_cache(maxsize=8)
def _detect_model_context_length(model_name: str) -> int:
    """Query Ollama /api/show for the model's max context_length.

    Ollama defaults num_ctx to 2048 if you don't pass it explicitly — even
    when the model itself supports 128K. We probe the model's true capacity
    once per process and clamp it to settings.max_context_tokens.
    """
    try:
        import requests
        resp = requests.post(
            f"{settings.ollama_base_url}/api/show",
            json={"model": model_name},
            timeout=5,
        )
        info = resp.json().get("model_info", {}) or {}
        for key, value in info.items():
            if key.endswith(".context_length") and isinstance(value, int):
                detected = int(value)
                clamped = min(detected, settings.max_context_tokens)
                logger.info(
                    "Context window for %s — model_max=%d, settings_cap=%d, using=%d",
                    model_name, detected, settings.max_context_tokens, clamped,
                )
                return clamped
    except Exception as e:
        logger.warning("Could not auto-detect context length for %s: %s", model_name, e)
    logger.info("Falling back to settings.max_context_tokens=%d for %s",
                settings.max_context_tokens, model_name)
    return settings.max_context_tokens


@lru_cache(maxsize=8)
def get_context_budgets(task_type: str = "heavy") -> dict[str, int]:
    """Compute per-slot character budgets sized to the active context window.

    Returns a dict of char limits for each dynamic prompt slot:
        text_context, table_context, map_reduce_context, memory_context.

    The math:
        num_ctx              = detected model context (clamped by settings)
        - output_reserved    = answer-generation budget
        - prompt_overhead    = static template + safety margin
        = available_tokens   → split by ctx_share_* fractions
        → multiplied by chars_per_token

    Cached per task_type so all callers share the same view of a given model.
    """
    model_name = MODEL_ROUTING.get(task_type, MODEL_ROUTING["light"])
    num_ctx = _detect_model_context_length(model_name)
    available_tokens = max(
        1500,  # absolute floor
        num_ctx - settings.output_tokens_reserved - settings.prompt_overhead_tokens,
    )
    cpt = max(1.0, float(settings.chars_per_token))
    available_chars = int(available_tokens * cpt)

    budgets = {
        "text_context":       int(available_chars * settings.ctx_share_text),
        "table_context":      int(available_chars * settings.ctx_share_table),
        "map_reduce_context": int(available_chars * settings.ctx_share_map_reduce),
        "memory_context":     int(available_chars * settings.ctx_share_memory),
    }
    logger.info(
        "Context budgets (task=%s, num_ctx=%d, avail=%d tok / %d chars) — "
        "text=%d, table=%d, map_reduce=%d, memory=%d",
        task_type, num_ctx, available_tokens, available_chars,
        budgets["text_context"], budgets["table_context"],
        budgets["map_reduce_context"], budgets["memory_context"],
    )
    return budgets


@lru_cache(maxsize=8)
def get_llm(
    task_type: str = "light",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> ChatOllama:
    """
    Get an LLM instance routed by task complexity.

    Args:
        task_type: "light" | "heavy" | "vision" — determines which model to use
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum output tokens

    Returns:
        ChatOllama instance configured for the task
    """
    model_name = MODEL_ROUTING.get(task_type, MODEL_ROUTING["light"])
    num_ctx = _detect_model_context_length(model_name)
    logger.info(
        "Initializing LLM — task=%s, model=%s, temp=%.1f, num_ctx=%d",
        task_type, model_name, temperature, num_ctx,
    )

    return ChatOllama(
        model=model_name,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_predict=max_tokens,
        num_ctx=num_ctx,
        # Keep alive for 5 minutes to avoid cold starts
        keep_alive="5m",
    )


def get_structured_llm(
    output_schema: Type[BaseModel],
    task_type: str = "light",
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Get an LLM that returns structured (Pydantic) output AND records tokens
    via the active token counter (see reset_token_counter / get_token_counter).

    Args:
        output_schema: Pydantic model class defining the expected output structure
        task_type: "light" | "heavy" — determines which model to use
        temperature: Sampling temperature

    Returns:
        Runnable whose .invoke(prompt) returns a parsed Pydantic instance.

    Implementation note: we ask LangChain for include_raw=True so we still get
    the raw AIMessage (with usage_metadata), then chain a small RunnableLambda
    that records tokens and unwraps the parsed schema. Callers see the same
    API as before (.invoke(...) returns the Pydantic obj).

    Example:
        structured_llm = get_structured_llm(QueryAnalysis, task_type="light")
        result: QueryAnalysis = structured_llm.invoke(prompt)
    """
    llm = get_llm(task_type=task_type, temperature=temperature)
    logger.info("Binding structured output — schema=%s", output_schema.__name__)
    # include_raw=True so we still see the AIMessage on parse failure and
    # can attempt JSON-repair extraction (see _make_unwrap_with_repair).
    base = llm.with_structured_output(output_schema, include_raw=True)
    return base | RunnableLambda(_make_unwrap_with_repair(output_schema))


def invoke_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    task_type: str = "light",
    temperature: float = 0.0,
    max_retries: int = 2,
) -> str:
    """
    Invoke an LLM with retry logic. Returns raw text response.

    Args:
        prompt: User prompt
        system_prompt: Optional system instructions
        task_type: "light" | "heavy" | "vision"
        temperature: Sampling temperature
        max_retries: Number of retries on failure

    Returns:
        LLM response as string
    """
    llm = get_llm(task_type=task_type, temperature=temperature)
    messages = []

    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    for attempt in range(max_retries + 1):
        try:
            start = time.time()
            response = llm.invoke(messages)
            elapsed = (time.time() - start) * 1000

            _record_tokens(response)
            logger.debug(
                "LLM response — task=%s, tokens=%s, time=%.0fms",
                task_type,
                getattr(response, "usage_metadata", "N/A"),
                elapsed,
            )
            return response.content

        except Exception as e:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, e)
            if attempt == max_retries:
                logger.error("LLM call exhausted retries — task=%s, error=%s", task_type, e)
                raise
            time.sleep(1.0 * (attempt + 1))  # Linear backoff

    return ""  # Unreachable, but satisfies type checker


def invoke_vision_llm(
    prompt: str,
    image_paths: list[str],
    max_retries: int = 2,
) -> str:
    """
    Invoke the vision model with images. Used during preprocessing only.

    Args:
        prompt: Text prompt describing what to extract/caption
        image_paths: List of local image file paths
        max_retries: Number of retries on failure

    Returns:
        Vision model response as string
    """
    import base64

    llm = get_llm(task_type="vision", temperature=0.0)

    # Build multimodal message content
    content = [{"type": "text", "text": prompt}]
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        except FileNotFoundError:
            logger.warning("Image not found: %s", img_path)

    messages = [HumanMessage(content=content)]

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
            _record_tokens(response)
            return response.content
        except Exception as e:
            logger.warning("Vision LLM failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, e)
            if attempt == max_retries:
                raise
            time.sleep(1.0 * (attempt + 1))

    return ""
