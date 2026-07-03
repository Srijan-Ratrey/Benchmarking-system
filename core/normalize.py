"""Verdict extraction (dotted path) and canonical normalization."""

from __future__ import annotations

from typing import Any


def extract_verdict(parsed: Any, path: str) -> Any:
    if parsed is None or not path:
        return None
    current: Any = parsed
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def normalize_verdict(raw: Any, normalize_map: dict[str, str]) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        key = "true" if raw else "false"
    else:
        key = str(raw).strip().lower()
    if not key:
        return None
    return normalize_map.get(key)
