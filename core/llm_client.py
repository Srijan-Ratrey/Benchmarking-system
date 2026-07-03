"""Async LiteLLM wrapper that returns latency, cost, and parsed output per call."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import acompletion

from .config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    raw_text: str | None
    parsed_json: Any | None
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    error: str | None
    parse_error: bool


async def call_model(
    model_cfg: ModelConfig,
    system_prompt: str,
    user_prompt: str,
    response_format: dict[str, Any] | None,
    timeout_seconds: int,
    max_retries: int,
) -> ModelOutput:
    """Call the model once with retries. Returns timing and cost regardless of outcome."""

    kwargs: dict[str, Any] = {
        "model": model_cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "timeout": timeout_seconds,
    }
    if model_cfg.temperature is not None:
        kwargs["temperature"] = model_cfg.temperature
    if model_cfg.max_tokens is not None:
        kwargs["max_tokens"] = model_cfg.max_tokens
    if response_format:
        kwargs["response_format"] = response_format
    if model_cfg.extra_body:
        kwargs["extra_body"] = model_cfg.extra_body
    if model_cfg.safety_settings:
        kwargs["safety_settings"] = model_cfg.safety_settings

    last_error: str | None = None
    started = time.perf_counter()

    for attempt in range(max_retries):
        try:
            call_start = time.perf_counter()
            resp = await acompletion(**kwargs)
            latency_ms = (time.perf_counter() - call_start) * 1000.0

            raw_text = resp.choices[0].message.content
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
            usage = getattr(resp, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = (
                getattr(usage, "completion_tokens", None) if usage else None
            )
            cost = _extract_cost(resp)

            # Empty / whitespace content is a silent provider refusal (safety filter,
            # finish_reason=content_filter, etc.). Retries rarely help when the filter
            # is deterministic, but we try once or twice in case it's transient.
            if not raw_text or not str(raw_text).strip():
                err_label = (
                    "content_filter"
                    if finish_reason == "content_filter"
                    else f"empty_response (finish={finish_reason or 'unknown'})"
                )
                last_error = err_label
                logger.debug(
                    "model=%s attempt=%d %s",
                    model_cfg.name,
                    attempt,
                    err_label,
                )
                # Don't retry deterministic refusals more than twice — saves time + $
                attempts_for_empty = min(max_retries, 2)
                if attempt < attempts_for_empty - 1:
                    await asyncio.sleep(0.5 + random.uniform(0, 0.5))
                    continue
                return ModelOutput(
                    raw_text=raw_text,
                    parsed_json=None,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    error=err_label,
                    parse_error=False,
                )

            parsed: Any = None
            parse_error = False
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                parse_error = True

            return ModelOutput(
                raw_text=raw_text,
                parsed_json=parsed,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                error=None,
                parse_error=parse_error,
            )

        except Exception as e:
            err_str = str(e)
            last_error = err_str[:200]
            logger.debug("model=%s attempt=%d error=%s", model_cfg.name, attempt, err_str[:120])
            if attempt >= max_retries - 1:
                break
            delay = min(0.5 * (2**attempt) + random.uniform(0, 0.5), 30)
            if "429" in err_str.lower() or "rate" in err_str.lower():
                delay = min(delay * 3, 30)
            await asyncio.sleep(delay)

    latency_ms = (time.perf_counter() - started) * 1000.0
    return ModelOutput(
        raw_text=None,
        parsed_json=None,
        latency_ms=latency_ms,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=None,
        error=last_error,
        parse_error=False,
    )


def _extract_cost(resp: Any) -> float | None:
    hidden = getattr(resp, "_hidden_params", None) or {}
    cost = hidden.get("response_cost")
    if cost is not None:
        try:
            return float(cost)
        except (TypeError, ValueError):
            pass
    try:
        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return None
