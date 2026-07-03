"""Async benchmark runner: dispatches each model over the dataset with a worker pool."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from .config import ModelConfig, PromptConfig, RunConfig
from .dataset import Example
from .llm_client import ModelOutput, call_model
from .normalize import extract_verdict, normalize_verdict

logger = logging.getLogger(__name__)

PAIR_SEPARATOR = "__"
LEGACY_PROMPT_NAME = "default"


def pair_key(prompt_name: str, model_name: str) -> str:
    """Composite key used as the unit identifier across raw.csv, metrics, summaries.

    When the prompt is the implicit "default" (i.e. a config using the legacy
    singular `prompt:` block), we elide the prefix so raw.csv columns and
    Streamlit labels stay identical to pre-multi-prompt runs.
    """
    if not prompt_name or prompt_name == LEGACY_PROMPT_NAME:
        return model_name
    return f"{prompt_name}{PAIR_SEPARATOR}{model_name}"


def split_pair_key(key: str) -> tuple[str, str]:
    """Inverse of pair_key. Returns ('', key) if the key has no prompt prefix."""
    if PAIR_SEPARATOR in key:
        prompt, _, model = key.partition(PAIR_SEPARATOR)
        return prompt, model
    return "", key


@dataclass
class CallResult:
    example_id: str
    model_name: str  # composite "{prompt}__{model}" key
    output: ModelOutput
    predicted_verdict: str | None
    correct: bool | None
    prompt_name: str = ""
    base_model_name: str = ""


@dataclass
class RunResult:
    started_at: float
    finished_at: float
    examples: list[Example]
    results_by_model: dict[str, list[CallResult]] = field(default_factory=dict)


class _Shutdown:
    def __init__(self) -> None:
        self.flag = False

    def trigger(self, *_: Any) -> None:
        if self.flag:
            raise SystemExit(1)
        self.flag = True
        logger.warning("Shutdown requested — finishing in-flight calls and saving progress")


def _format_user_prompt(template: str, example: Example) -> str:
    result = template
    for col, val in example.inputs.items():
        result = result.replace("{" + col + "}", val)
    return result


def _build_result(
    example: Example,
    prompt_cfg: PromptConfig,
    model_cfg: ModelConfig,
    cfg: RunConfig,
    output: ModelOutput,
) -> CallResult:
    predicted_raw = extract_verdict(output.parsed_json, cfg.verdict.extract_path)
    predicted = normalize_verdict(predicted_raw, cfg.verdict.normalize)
    correct: bool | None
    if predicted is None or example.gold_verdict_normalized is None:
        correct = None
    else:
        correct = predicted == example.gold_verdict_normalized
    return CallResult(
        example_id=example.id,
        model_name=pair_key(prompt_cfg.name, model_cfg.name),
        output=output,
        predicted_verdict=predicted,
        correct=correct,
        prompt_name=prompt_cfg.name,
        base_model_name=model_cfg.name,
    )


async def _run_one_pair(
    prompt_cfg: PromptConfig,
    model_cfg: ModelConfig,
    examples: list[Example],
    cfg: RunConfig,
    on_result: Any,
    shutdown: _Shutdown,
) -> list[CallResult]:
    queue: asyncio.Queue[Example] = asyncio.Queue()
    for ex in examples:
        queue.put_nowait(ex)

    results: list[CallResult] = []
    lock = asyncio.Lock()
    total = len(examples)
    started = time.perf_counter()
    log_every = max(1, total // 20)
    label = pair_key(prompt_cfg.name, model_cfg.name)

    async def worker() -> None:
        while True:
            try:
                example = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                if shutdown.flag:
                    return
                user_prompt = _format_user_prompt(prompt_cfg.user_template, example)
                output = await call_model(
                    model_cfg=model_cfg,
                    system_prompt=prompt_cfg.system,
                    user_prompt=user_prompt,
                    response_format=cfg.response_format,
                    timeout_seconds=cfg.runtime.timeout_seconds,
                    max_retries=cfg.runtime.max_retries,
                )
                result = _build_result(example, prompt_cfg, model_cfg, cfg, output)
                async with lock:
                    results.append(result)
                    if on_result:
                        on_result(result)
                    done = len(results)
                    if done % log_every == 0 or done == total:
                        elapsed = time.perf_counter() - started
                        speed = done / elapsed if elapsed > 0 else 0
                        logger.info(
                            "[%s] %d/%d (%.1f rps)", label, done, total, speed
                        )
            except Exception as e:
                logger.exception("Worker error for %s: %s", label, e)
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker()) for _ in range(cfg.runtime.concurrency)
    ]
    await asyncio.gather(*workers, return_exceptions=True)

    elapsed = time.perf_counter() - started
    logger.info(
        "[%s] done %d/%d in %.1fs", label, len(results), total, elapsed
    )
    return results


async def run_benchmark(
    cfg: RunConfig,
    examples: list[Example],
    on_result: Any = None,
    models: list[ModelConfig] | None = None,
    prompts: list[PromptConfig] | None = None,
) -> RunResult:
    """Benchmark each (prompt, model) pair over the dataset.

    Inner workers run in parallel; pairs run sequentially. Results are keyed in
    `results_by_model` by the composite `"{prompt}__{model}"` pair key.
    """

    shutdown = _Shutdown()
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, shutdown.trigger)
    except (NotImplementedError, RuntimeError):
        signal.signal(signal.SIGINT, shutdown.trigger)

    selected_models = models if models is not None else cfg.models
    selected_prompts = prompts if prompts is not None else cfg.prompts
    result = RunResult(started_at=time.time(), finished_at=0.0, examples=examples)

    for prompt_cfg in selected_prompts:
        for model_cfg in selected_models:
            if shutdown.flag:
                logger.warning(
                    "Skipping remaining pair: %s",
                    pair_key(prompt_cfg.name, model_cfg.name),
                )
                break
            label = pair_key(prompt_cfg.name, model_cfg.name)
            logger.info("=== running pair: %s (%s) ===", label, model_cfg.model)
            per_pair = await _run_one_pair(
                prompt_cfg=prompt_cfg,
                model_cfg=model_cfg,
                examples=examples,
                cfg=cfg,
                on_result=on_result,
                shutdown=shutdown,
            )
            result.results_by_model[label] = per_pair
        if shutdown.flag:
            break

    result.finished_at = time.time()
    return result
