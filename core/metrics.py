"""Aggregate accuracy, latency, cost, and token counts per model."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .runner import CallResult


@dataclass
class ModelMetrics:
    model_name: str
    total: int                       # kept rows (refused rows excluded)
    refused_count: int               # rows where the provider returned empty content
    coverage: float                  # parseable verdicts / total
    accuracy: float | None           # correct / scoreable (gold + predicted both known)
    scoreable: int
    correct: int
    error_count: int
    parse_error_count: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_mean_ms: float | None
    cost_total_usd: float
    cost_mean_usd: float | None
    cost_per_correct_usd: float | None
    prompt_tokens_total: int
    completion_tokens_total: int
    confusion: dict[str, Any] = field(default_factory=dict)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_v = sorted(values)
    idx = int(round((pct / 100.0) * (len(sorted_v) - 1)))
    return sorted_v[idx]


def compute_model_metrics(
    model_name: str,
    results: list[CallResult],
    gold_by_id: dict[str, str | None],
) -> ModelMetrics:
    # Refused rows = provider returned empty / whitespace content. Exclude from
    # every aggregate so the leaderboard isn't diluted by silent safety blocks.
    kept = [
        r for r in results
        if r.output.raw_text and str(r.output.raw_text).strip()
    ]
    refused_count = len(results) - len(kept)
    total = len(kept)
    latencies: list[float] = []
    costs: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    error_count = 0
    parse_error_count = 0
    parseable = 0
    correct = 0
    scoreable = 0

    classes: set[str] = set()
    confusion_counts: dict[tuple[str, str], int] = {}

    for r in kept:
        out = r.output
        if out.error:
            error_count += 1
        if out.parse_error:
            parse_error_count += 1
        if out.latency_ms is not None:
            latencies.append(out.latency_ms)
        if out.cost_usd is not None:
            costs.append(out.cost_usd)
        if out.prompt_tokens:
            prompt_tokens += out.prompt_tokens
        if out.completion_tokens:
            completion_tokens += out.completion_tokens

        if r.predicted_verdict is not None:
            parseable += 1

        gold = gold_by_id.get(r.example_id)
        if r.predicted_verdict is not None and gold is not None:
            scoreable += 1
            classes.add(gold)
            classes.add(r.predicted_verdict)
            key = (gold, r.predicted_verdict)
            confusion_counts[key] = confusion_counts.get(key, 0) + 1
            if r.correct:
                correct += 1

    cost_total = sum(costs)
    confusion = _build_confusion(classes, confusion_counts) if classes else {}

    return ModelMetrics(
        model_name=model_name,
        total=total,
        refused_count=refused_count,
        coverage=(parseable / total) if total else 0.0,
        accuracy=(correct / scoreable) if scoreable else None,
        scoreable=scoreable,
        correct=correct,
        error_count=error_count,
        parse_error_count=parse_error_count,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_mean_ms=statistics.mean(latencies) if latencies else None,
        cost_total_usd=cost_total,
        cost_mean_usd=(cost_total / len(costs)) if costs else None,
        cost_per_correct_usd=(cost_total / correct) if correct else None,
        prompt_tokens_total=prompt_tokens,
        completion_tokens_total=completion_tokens,
        confusion=confusion,
    )


def _build_confusion(
    classes: set[str], counts: dict[tuple[str, str], int]
) -> dict[str, Any]:
    sorted_classes = sorted(classes)
    matrix: dict[str, dict[str, int]] = {
        gold: {pred: counts.get((gold, pred), 0) for pred in sorted_classes}
        for gold in sorted_classes
    }
    out: dict[str, Any] = {"classes": sorted_classes, "matrix": matrix}

    if len(sorted_classes) == 2:
        pos = sorted_classes[1]
        neg = sorted_classes[0]
        tp = counts.get((pos, pos), 0)
        fp = counts.get((neg, pos), 0)
        fn = counts.get((pos, neg), 0)
        tn = counts.get((neg, neg), 0)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )
        out["binary"] = {
            "positive_class": pos,
            "negative_class": neg,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return out
