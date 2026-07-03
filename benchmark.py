"""CLI entry point for the benchmark harness.

Usage:
    python benchmarks/benchmark.py --config benchmarks/configs/example.yaml
    python benchmarks/benchmark.py --config <cfg> --sample 5 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import litellm
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.core.config import (  # noqa: E402
    RunConfig,
    diff_for_extension,
    load_config,
    load_snapshot_config,
)
from benchmarks.core.dataset import load_dataset  # noqa: E402
from benchmarks.core.report import (  # noqa: E402
    RawCsvWriter,
    existing_model_names_in_run,
    load_call_results_from_raw,
    make_run_dir,
    snapshot_config,
    write_summary,
)
from benchmarks.core.runner import RunResult, pair_key, run_benchmark  # noqa: E402

logger = logging.getLogger("benchmarks")


def _setup_logging(run_dir: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        run_dir / "run.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def _setup_litellm() -> None:
    load_dotenv()
    api_base = os.environ.get("LITELLM_API_BASE")
    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        print("ERROR: LITELLM_API_KEY not set in environment / .env", file=sys.stderr)
        sys.exit(2)
    if api_base:
        litellm.api_base = api_base
    litellm.api_key = api_key
    litellm.suppress_debug_info = True


def _resolve_models(cfg: RunConfig, names_csv: str | None):
    if not names_csv:
        return cfg.models
    wanted = [n.strip() for n in names_csv.split(",") if n.strip()]
    by_name = {m.name: m for m in cfg.models}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise SystemExit(f"Unknown model(s) requested via --models: {missing}")
    return [by_name[n] for n in wanted]


def _resolve_prompts(cfg: RunConfig, names_csv: str | None):
    if not names_csv:
        return cfg.prompts
    wanted = [n.strip() for n in names_csv.split(",") if n.strip()]
    by_name = {p.name: p for p in cfg.prompts}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise SystemExit(f"Unknown prompt(s) requested via --prompts: {missing}")
    return [by_name[n] for n in wanted]


def _print_plan(cfg: RunConfig, example_count: int, models, prompts) -> None:
    print("=== Benchmark plan ===")
    print(f"run_name:       {cfg.run_name}")
    print(f"dataset:        {cfg.dataset.path}")
    print(f"  input_cols:   {cfg.dataset.input_columns}")
    print(f"  gold_col:     {cfg.dataset.gold_column}")
    print(f"  id_col:       {cfg.dataset.id_column}")
    print(f"examples:       {example_count}")
    print(f"prompts ({len(prompts)}): {[p.name for p in prompts]}")
    print(f"models ({len(models)}):")
    for m in models:
        marker = " [base]" if cfg.base_model == m.name else ""
        print(f"  - {m.name}: {m.model}{marker}")
    pair_count = len(prompts) * len(models)
    print(
        f"pairs:          {pair_count}  "
        f"({len(prompts)} prompts × {len(models)} models, "
        f"{pair_count * example_count} calls total)"
    )
    print(f"concurrency:    {cfg.runtime.concurrency}")
    print(f"verdict path:   {cfg.verdict.extract_path}")
    print(f"normalize map:  {cfg.verdict.normalize}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM benchmark harness")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--name", help="Override run_name")
    p.add_argument("--sample", type=int, help="Override runtime.sample_size")
    p.add_argument("--models", help="Comma-separated subset of model names to run")
    p.add_argument("--prompts", help="Comma-separated subset of prompt names to run")
    p.add_argument(
        "--dry-run", action="store_true", help="Validate + print plan, no API calls"
    )
    p.add_argument("--runs-dir", default=str(REPO_ROOT / "benchmarks" / "runs"))
    p.add_argument(
        "--add-to-run",
        metavar="RUN_DIR",
        help="Extend an existing run's raw.csv + summary with the new model(s) "
             "from --models. The new YAML must match the run's snapshot on dataset/"
             "prompt/verdict/response_format/sample_size or the command will refuse.",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="(With --add-to-run) overwrite the named model(s) if they already "
             "have a column block in raw.csv. Use when re-running a model whose "
             "previous attempt errored out.",
    )
    return p.parse_args()


async def _async_main(cfg: RunConfig, args: argparse.Namespace) -> int:
    if args.add_to_run:
        return await _async_extend(cfg, args)

    examples = load_dataset(cfg)
    models = _resolve_models(cfg, args.models)
    prompts = _resolve_prompts(cfg, args.prompts)

    if args.dry_run:
        _print_plan(cfg, len(examples), models, prompts)
        print("\n--dry-run: no API calls made.")
        return 0

    _setup_litellm()
    run_dir = make_run_dir(Path(args.runs_dir), cfg.run_name)
    _setup_logging(run_dir)
    snapshot_config(Path(args.config).resolve(), run_dir)

    pair_names = [pair_key(p.name, m.name) for p in prompts for m in models]
    logger.info("Run dir: %s", run_dir)
    logger.info("Examples: %d | Pairs: %s", len(examples), pair_names)

    raw_writer = RawCsvWriter(
        path=run_dir / "raw.csv",
        examples=examples,
        model_names=pair_names,
        input_columns=cfg.dataset.input_columns,
    )

    try:
        run_result = await run_benchmark(
            cfg=cfg,
            examples=examples,
            on_result=raw_writer.record,
            models=models,
            prompts=prompts,
        )
    finally:
        raw_writer.close()

    metrics = write_summary(run_dir, cfg, run_result)
    _print_summary(metrics, cfg.base_model)
    logger.info("Run complete. Outputs in: %s", run_dir)
    return 0


async def _async_extend(cfg: RunConfig, args: argparse.Namespace) -> int:
    """Add one or more new models to an existing run's raw.csv + summary."""
    run_dir = Path(args.add_to_run).resolve()
    snapshot_path = run_dir / "config.yaml"
    raw_path = run_dir / "raw.csv"
    if not snapshot_path.exists() or not raw_path.exists():
        print(
            f"ERROR: target run dir is missing config.yaml or raw.csv: {run_dir}",
            file=sys.stderr,
        )
        return 2

    # Snapshot's dataset.path is relative to the original configs/ dir;
    # resolve from the new YAML's dir as a fallback.
    new_config_dir = Path(args.config).resolve().parent
    snapshot_cfg = load_snapshot_config(snapshot_path, fallback_dataset_dir=new_config_dir)

    diffs = diff_for_extension(snapshot_cfg, cfg)
    if diffs:
        print(
            "ERROR: cannot extend — the new YAML differs from the run's snapshot on these fields:",
            file=sys.stderr,
        )
        for path, old, new in diffs:
            print(f"  {path}: snapshot={old!r}  new={new!r}", file=sys.stderr)
        print(
            "\nFix the YAML so dataset/prompt/verdict/response_format/sample_size all match,"
            " then re-run. (Only the models list is allowed to differ.)",
            file=sys.stderr,
        )
        return 2

    if not args.models:
        print(
            "ERROR: --add-to-run requires --models <name[,name...]> to pick the new model(s).",
            file=sys.stderr,
        )
        return 2

    new_models = _resolve_models(cfg, args.models)
    existing_models = existing_model_names_in_run(run_dir)
    new_model_names = {m.name for m in new_models}
    clash = [m.name for m in new_models if m.name in existing_models]
    if clash and not args.replace:
        print(
            f"ERROR: model(s) already present in raw.csv (refuse to overwrite): {clash}\n"
            f"Pass --replace to overwrite them.",
            file=sys.stderr,
        )
        return 2
    # Models that stay untouched (historical) vs. models that we will (re)write.
    historical_models = [m for m in existing_models if m not in new_model_names]

    if args.dry_run:
        print("=== Extension plan ===")
        print(f"target run:     {run_dir}")
        print(f"existing models: {existing_models}")
        print(f"adding models:   {[m.name for m in new_models]}")
        print(f"dataset:         {snapshot_cfg.dataset.path}")
        print(f"sample_size:     {snapshot_cfg.runtime.sample_size}")
        print("\n--dry-run: no API calls made.")
        return 0

    _setup_litellm()
    _setup_logging(run_dir)
    logger.info("=== EXTENDING run: %s ===", run_dir)
    logger.info(
        "Existing models: %s | Adding: %s",
        existing_models,
        [m.name for m in new_models],
    )

    # The recorded sample_size in the snapshot may be the YAML default (e.g. 100)
    # even when the user overrode it via --sample at run time. Align to the actual
    # raw.csv row count so the dataset reload reproduces the same example IDs.
    import csv as _csv
    with open(raw_path, newline="", encoding="utf-8") as _f:
        raw_row_count = sum(1 for _ in _csv.DictReader(_f))
    if raw_row_count != snapshot_cfg.runtime.sample_size:
        logger.info(
            "Overriding sample_size %s → %s to match existing raw.csv",
            snapshot_cfg.runtime.sample_size,
            raw_row_count,
        )
        snapshot_cfg.runtime.sample_size = raw_row_count

    examples = load_dataset(snapshot_cfg)
    logger.info("Examples loaded: %d", len(examples))

    new_pair_names = [
        pair_key(p.name, m.name)
        for p in snapshot_cfg.prompts
        for m in new_models
    ]
    raw_writer = RawCsvWriter(
        path=raw_path,
        examples=examples,
        model_names=new_pair_names,
        input_columns=snapshot_cfg.dataset.input_columns,
        extend_existing=True,
    )

    # Run only the new model(s). Use the snapshot's runtime/prompt/response_format
    # but with the new YAML's model list (extra_body, safety_settings, etc).
    extend_cfg = RunConfig(
        run_name=snapshot_cfg.run_name,
        dataset=snapshot_cfg.dataset,
        prompts=snapshot_cfg.prompts,
        verdict=snapshot_cfg.verdict,
        models=new_models,
        base_model=snapshot_cfg.base_model,
        runtime=cfg.runtime,
        response_format=snapshot_cfg.response_format,
        raw=snapshot_cfg.raw,
    )
    try:
        fresh_run = await run_benchmark(
            cfg=extend_cfg,
            examples=examples,
            on_result=raw_writer.record,
            models=new_models,
            prompts=snapshot_cfg.prompts,
        )
    finally:
        raw_writer.close()

    # Reconstruct ONLY historical models (i.e., excluding ones we just re-ran)
    # from the merged raw.csv, and combine with fresh results.
    historical = load_call_results_from_raw(run_dir, historical_models)
    merged = RunResult(
        started_at=fresh_run.started_at,
        finished_at=fresh_run.finished_at,
        examples=examples,
    )
    for name, results in historical.items():
        merged.results_by_model[name] = results
    for name, results in fresh_run.results_by_model.items():
        merged.results_by_model[name] = results

    metrics = write_summary(run_dir, snapshot_cfg, merged)
    _print_summary(metrics, snapshot_cfg.base_model)
    logger.info("Extension complete. Updated outputs in: %s", run_dir)
    return 0


def _print_summary(metrics, base_model: str | None) -> None:
    print("\n=== Summary ===")
    header = f"{'model':<24} {'acc':>7} {'cov':>6} {'p50ms':>8} {'p95ms':>8} {'$total':>10} {'$/correct':>11}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        marker = "*" if name == base_model else " "
        print(
            f"{marker} {name:<22} "
            f"{_fmt(m.accuracy, 4):>7} "
            f"{_fmt(m.coverage, 3):>6} "
            f"{_fmt(m.latency_p50_ms, 1):>8} "
            f"{_fmt(m.latency_p95_ms, 1):>8} "
            f"{_fmt(m.cost_total_usd, 4):>10} "
            f"{_fmt(m.cost_per_correct_usd, 6):>11}"
        )


def _fmt(value, ndigits):
    if value is None:
        return "-"
    return f"{value:.{ndigits}f}"


def main() -> int:
    args = parse_args()
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    if args.name:
        cfg.run_name = args.name
    if args.sample is not None:
        cfg.runtime.sample_size = args.sample

    return asyncio.run(_async_main(cfg, args))


if __name__ == "__main__":
    sys.exit(main())
