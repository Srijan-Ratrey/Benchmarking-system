"""Writers for raw.csv (per-row predictions) and summary.csv/json (per-model aggregates)."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import RunConfig
from .dataset import Example
from .llm_client import ModelOutput
from .metrics import ModelMetrics, compute_model_metrics
from .runner import CallResult, RunResult, split_pair_key

logger = logging.getLogger(__name__)


class RawCsvWriter:
    """Append-as-you-go writer keyed by example_id with one column block per model.

    Flushes are throttled to at most one rewrite every FLUSH_INTERVAL_SEC to
    keep the cost sub-O(N²) on large runs. Call close() at end of run to
    guarantee the final state is on disk.
    """

    FLUSH_INTERVAL_SEC = 2.0

    def __init__(
        self,
        path: Path,
        examples: list[Example],
        model_names: list[str],
        input_columns: list[str],
        extend_existing: bool = False,
    ) -> None:
        self.path = path
        self.input_columns = input_columns
        self._dirty = False
        self._last_flush_ts = 0.0

        if extend_existing:
            existing_fieldnames, existing_rows = _read_existing_raw(path)
            existing_models = _existing_model_names(existing_fieldnames)
            self._row_index = {row["id"]: i for i, row in enumerate(existing_rows)}
            self._rows = existing_rows
            self.model_names = existing_models + [
                m for m in model_names if m not in existing_models
            ]
            # Initialize empty column blocks for the new models on every row
            new_blocks = [m for m in model_names if m not in existing_models]
            for row in self._rows:
                for m in new_blocks:
                    for col in _model_column_suffixes():
                        row.setdefault(f"{m}{col}", "")
            self._fieldnames = self._build_fieldnames()
        else:
            self.model_names = model_names
            self._row_index = {ex.id: i for i, ex in enumerate(examples)}
            self._rows = []
            for ex in examples:
                row: dict[str, Any] = {
                    "id": ex.id,
                    "gold_verdict_raw": ex.gold_verdict_raw,
                    "gold_verdict": ex.gold_verdict_normalized,
                }
                for col in input_columns:
                    row[f"input_{col}"] = _truncate(ex.inputs.get(col, ""), 500)
                self._rows.append(row)
            self._fieldnames = self._build_fieldnames()
        self._flush()

    def _build_fieldnames(self) -> list[str]:
        fields = ["id", "gold_verdict_raw", "gold_verdict"]
        for col in self.input_columns:
            fields.append(f"input_{col}")
        for m in self.model_names:
            fields.extend([
                f"{m}_verdict",
                f"{m}_correct",
                f"{m}_latency_ms",
                f"{m}_cost_usd",
                f"{m}_prompt_tokens",
                f"{m}_completion_tokens",
                f"{m}_error",
                f"{m}_parse_error",
                f"{m}_raw_output",
            ])
        return fields

    def record(self, result: CallResult) -> None:
        idx = self._row_index.get(result.example_id)
        if idx is None:
            return
        row = self._rows[idx]
        m = result.model_name
        out = result.output
        row[f"{m}_verdict"] = result.predicted_verdict
        row[f"{m}_correct"] = (
            "" if result.correct is None else ("1" if result.correct else "0")
        )
        row[f"{m}_latency_ms"] = round(out.latency_ms, 2) if out.latency_ms else ""
        row[f"{m}_cost_usd"] = f"{out.cost_usd:.8f}" if out.cost_usd is not None else ""
        row[f"{m}_prompt_tokens"] = out.prompt_tokens if out.prompt_tokens is not None else ""
        row[f"{m}_completion_tokens"] = (
            out.completion_tokens if out.completion_tokens is not None else ""
        )
        row[f"{m}_error"] = out.error or ""
        row[f"{m}_parse_error"] = "1" if out.parse_error else ""
        row[f"{m}_raw_output"] = _truncate(out.raw_text or "", 2000)
        self._dirty = True
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        if time.monotonic() - self._last_flush_ts >= self.FLUSH_INTERVAL_SEC:
            self._flush()

    def close(self) -> None:
        """Final flush guaranteed to persist any buffered rows. Idempotent."""
        if self._dirty:
            self._flush()

    def _flush(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)
        self._dirty = False
        self._last_flush_ts = time.monotonic()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


_MODEL_COLUMN_SUFFIXES = (
    "_verdict", "_correct", "_latency_ms", "_cost_usd",
    "_prompt_tokens", "_completion_tokens", "_error", "_parse_error", "_raw_output",
)


def _model_column_suffixes() -> tuple[str, ...]:
    return _MODEL_COLUMN_SUFFIXES


def _read_existing_raw(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot extend — raw.csv not found at {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _existing_model_names(fieldnames: list[str]) -> list[str]:
    """Reconstruct model names from a raw.csv header by looking for *_verdict columns
    (excluding the reserved `gold_verdict`)."""
    seen: list[str] = []
    for col in fieldnames:
        if col.endswith("_verdict") and col != "gold_verdict":
            name = col[: -len("_verdict")]
            if name and name not in seen:
                seen.append(name)
    return seen


def existing_model_names_in_run(run_dir: Path) -> list[str]:
    """Public helper: list model names currently present in a run's raw.csv."""
    fieldnames, _ = _read_existing_raw(run_dir / "raw.csv")
    return _existing_model_names(fieldnames)


def load_call_results_from_raw(
    run_dir: Path, model_names: list[str]
) -> dict[str, list[CallResult]]:
    """Reconstruct per-model CallResult lists from a previously-written raw.csv.

    Used when extending an existing run so the existing models' metrics flow
    through `compute_model_metrics` unchanged.
    """

    _, rows = _read_existing_raw(run_dir / "raw.csv")
    out: dict[str, list[CallResult]] = {m: [] for m in model_names}

    for row in rows:
        for m in model_names:
            raw_text = row.get(f"{m}_raw_output") or None
            try:
                parsed = json.loads(raw_text) if raw_text else None
            except (TypeError, json.JSONDecodeError):
                parsed = None
            output = ModelOutput(
                raw_text=raw_text,
                parsed_json=parsed,
                latency_ms=_as_float(row.get(f"{m}_latency_ms")) or 0.0,
                prompt_tokens=_as_int(row.get(f"{m}_prompt_tokens")),
                completion_tokens=_as_int(row.get(f"{m}_completion_tokens")),
                cost_usd=_as_float(row.get(f"{m}_cost_usd")),
                error=(row.get(f"{m}_error") or None),
                parse_error=bool(row.get(f"{m}_parse_error")),
            )
            predicted = row.get(f"{m}_verdict") or None
            correct_raw = row.get(f"{m}_correct")
            if correct_raw in ("1", 1):
                correct: bool | None = True
            elif correct_raw in ("0", 0):
                correct = False
            else:
                correct = None
            prompt_name, base_model = split_pair_key(m)
            out[m].append(CallResult(
                example_id=row["id"],
                model_name=m,
                output=output,
                predicted_verdict=predicted,
                correct=correct,
                prompt_name=prompt_name,
                base_model_name=base_model or m,
            ))
    return out


def _as_float(v: Any) -> float | None:
    if v in (None, "", "nan"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    if v in (None, "", "nan"):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def write_summary(
    run_dir: Path,
    cfg: RunConfig,
    run_result: RunResult,
) -> dict[str, ModelMetrics]:
    gold_by_id = {ex.id: ex.gold_verdict_normalized for ex in run_result.examples}
    metrics: dict[str, ModelMetrics] = {}
    for model_name, results in run_result.results_by_model.items():
        metrics[model_name] = compute_model_metrics(model_name, results, gold_by_id)

    _write_summary_csv(run_dir / "summary.csv", metrics, cfg.base_model)
    _write_summary_json(run_dir / "summary.json", cfg, run_result, metrics)
    return metrics


def _write_summary_csv(
    path: Path, metrics: dict[str, ModelMetrics], base_model: str | None
) -> None:
    fields = [
        "prompt", "model", "pair", "is_base", "total", "refused", "scoreable", "correct",
        "accuracy", "coverage",
        "latency_p50_ms", "latency_p95_ms", "latency_mean_ms",
        "cost_total_usd", "cost_mean_usd", "cost_per_correct_usd",
        "prompt_tokens_total", "completion_tokens_total",
        "error_count", "parse_error_count",
        "precision", "recall", "f1",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, m in metrics.items():
            binary = m.confusion.get("binary", {}) if m.confusion else {}
            prompt_name, model_only = split_pair_key(name)
            writer.writerow({
                "prompt": prompt_name,
                "model": model_only,
                "pair": name,
                "is_base": "1" if base_model == model_only or base_model == name else "",
                "total": m.total,
                "refused": m.refused_count,
                "scoreable": m.scoreable,
                "correct": m.correct,
                "accuracy": _fmt(m.accuracy, 4),
                "coverage": _fmt(m.coverage, 4),
                "latency_p50_ms": _fmt(m.latency_p50_ms, 1),
                "latency_p95_ms": _fmt(m.latency_p95_ms, 1),
                "latency_mean_ms": _fmt(m.latency_mean_ms, 1),
                "cost_total_usd": _fmt(m.cost_total_usd, 6),
                "cost_mean_usd": _fmt(m.cost_mean_usd, 8),
                "cost_per_correct_usd": _fmt(m.cost_per_correct_usd, 8),
                "prompt_tokens_total": m.prompt_tokens_total,
                "completion_tokens_total": m.completion_tokens_total,
                "error_count": m.error_count,
                "parse_error_count": m.parse_error_count,
                "precision": _fmt(binary.get("precision"), 4),
                "recall": _fmt(binary.get("recall"), 4),
                "f1": _fmt(binary.get("f1"), 4),
            })


def _write_summary_json(
    path: Path,
    cfg: RunConfig,
    run_result: RunResult,
    metrics: dict[str, ModelMetrics],
) -> None:
    payload = {
        "run_name": cfg.run_name,
        "started_at": run_result.started_at,
        "finished_at": run_result.finished_at,
        "duration_seconds": round(run_result.finished_at - run_result.started_at, 2),
        "dataset": {
            "path": str(cfg.dataset.path),
            "total_examples": len(run_result.examples),
            "input_columns": cfg.dataset.input_columns,
            "gold_column": cfg.dataset.gold_column,
        },
        "base_model": cfg.base_model,
        "models": {name: asdict(m) for name, m in metrics.items()},
        "config_snapshot": cfg.raw,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _fmt(value: float | None, ndigits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{ndigits}f}"


def snapshot_config(config_path: Path, run_dir: Path) -> None:
    target = run_dir / "config.yaml"
    try:
        shutil.copy2(config_path, target)
    except Exception as e:
        logger.warning("Could not snapshot config to %s: %s", target, e)


def make_run_dir(base: Path, run_name: str) -> Path:
    ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())
    run_dir = base / f"{ts}-{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
