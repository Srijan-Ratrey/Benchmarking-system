"""CSV loader that turns a labeled dataset into Example objects."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any

from .config import RunConfig
from .normalize import normalize_verdict


@dataclass
class Example:
    id: str
    inputs: dict[str, str]
    gold_verdict_raw: str
    gold_verdict_normalized: str | None
    extra: dict[str, Any] = field(default_factory=dict)


def load_dataset(cfg: RunConfig) -> list[Example]:
    ds = cfg.dataset

    with open(ds.path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    _validate_columns(fields, ds.input_columns, ds.gold_column, ds.id_column)

    skip = set(ds.skip_columns)
    examples: list[Example] = []
    for i, row in enumerate(rows):
        example_id = (
            str(row[ds.id_column]).strip()
            if ds.id_column and row.get(ds.id_column)
            else str(i)
        )
        inputs = {col: (row.get(col) or "") for col in ds.input_columns}
        gold_raw = (row.get(ds.gold_column) or "").strip()
        gold_norm = normalize_verdict(gold_raw, cfg.verdict.normalize)
        extra = {
            k: v for k, v in row.items()
            if k not in skip
            and k not in ds.input_columns
            and k != ds.gold_column
            and k != ds.id_column
        }
        examples.append(Example(
            id=example_id,
            inputs=inputs,
            gold_verdict_raw=gold_raw,
            gold_verdict_normalized=gold_norm,
            extra=extra,
        ))

    if cfg.runtime.sample_size is not None:
        examples = examples[: cfg.runtime.sample_size]
    return examples


def _validate_columns(
    fields: list[str],
    input_columns: list[str],
    gold_column: str,
    id_column: str | None,
) -> None:
    missing: list[str] = []
    for col in input_columns:
        if col not in fields:
            missing.append(col)
    if gold_column not in fields:
        missing.append(gold_column)
    if id_column and id_column not in fields:
        missing.append(id_column)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns {missing}. Available: {fields}"
        )
