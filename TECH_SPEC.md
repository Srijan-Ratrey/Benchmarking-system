# Tech Spec — LLM Benchmark Harness

**Location:** `watchdog/benchmarks/`
**Status:** Implemented and in use (8+ runs as of 2026-05)
**Owner:** Two Four Labs

---

## 1. Purpose

A generic, config-driven harness for benchmarking LLMs on labeled classification tasks (e.g. conversation quality PASS/FAIL, memory-eval verdicts). Point it at a labeled CSV and a list of models; it runs every (prompt × model) pair through the same dataset via the LiteLLM proxy and reports per-pair **accuracy**, **coverage**, **latency (p50/p95/mean)**, **cost**, and **precision/recall/F1** (binary tasks).

Design goals:

- **Dataset-agnostic** — all task specifics (columns, prompt, verdict schema, label mapping) live in YAML; core code never changes per task.
- **No judge LLM** — accuracy is exact-match between the model's extracted verdict and the gold label. Deterministic and cheap.
- **Crash-safe** — results are appended to disk as calls complete; Ctrl+C preserves finished rows.
- **Extensible runs** — new models can be added to a completed run without re-running existing models.

## 2. Architecture

```
benchmarks/
├── benchmark.py          # CLI entry point (argparse + asyncio)
├── app.py                # Streamlit web UI (688 LOC)
├── core/
│   ├── config.py         # YAML loading, validation, snapshot diffing
│   ├── dataset.py        # CSV → Example objects
│   ├── llm_client.py     # Async LiteLLM wrapper (retries, cost, latency)
│   ├── runner.py         # Async worker-pool runner per (prompt, model) pair
│   ├── normalize.py      # Verdict extraction (dotted path) + label normalization
│   ├── metrics.py        # Per-model aggregates + confusion matrix
│   └── report.py         # raw.csv / summary.csv / summary.json writers
├── configs/              # One YAML per benchmark definition
├── data/                 # Labeled CSVs (gitignored / local)
├── runs/                 # One folder per run: <timestamp>-<run_name>/
└── scripts/
    └── build_balanced_csv.py   # Stratified 50/50 PASS-FAIL slice builder
```

### Data flow

```
YAML config ──► load_config() ──► RunConfig (validated dataclasses)
CSV dataset ──► load_dataset() ──► list[Example] (gold label pre-normalized)
                     │
                     ▼
run_benchmark(): for each (prompt × model) pair, sequentially:
    worker pool (asyncio.Queue, N=concurrency workers)
        └─► call_model() via LiteLLM ──► ModelOutput
            └─► extract_verdict() + normalize_verdict() ──► CallResult
                    └─► RawCsvWriter.record()  (throttled flush, ≤1 rewrite / 2s)
                     ▼
write_summary() ──► summary.csv + summary.json (metrics per pair)
```

## 3. Core components

### 3.1 Config (`core/config.py`)

Dataclasses: `DatasetConfig`, `PromptConfig`, `VerdictConfig`, `ModelConfig`, `RuntimeConfig`, `RunConfig`.

Validation at load time:

- Required keys with type checks (`run_name`, `dataset`, `verdict`, `models`; exactly one of `prompt` / `prompts`).
- Every `input_column` must appear as a `{placeholder}` in each `user_template`.
- Prompt names must match `[a-zA-Z0-9_-]+`, no `__` (reserved as the pair-key separator), no duplicates.
- Model names must be unique; `base_model` must reference a declared model.
- `dataset.path` resolved relative to the YAML file; file must exist.
- Verdict normalize map keys lowercased at load (case-insensitive matching).

`diff_for_extension()` compares a run's config snapshot against a new YAML and returns mismatched fields — used to gate `--add-to-run` (only the models list may differ).

### 3.2 Dataset (`core/dataset.py`)

- CSV via `csv.DictReader`, UTF-8 with `errors="replace"`.
- Row id = `id_column` value, else row index.
- Gold label normalized at load through the same `verdict.normalize` map as predictions.
- Non-input, non-gold columns kept in `Example.extra` (minus `skip_columns`).
- `sample_size` = head-N truncation (hence the pre-balanced CSV script — see §7).

### 3.3 LLM client (`core/llm_client.py`)

Single async call path via `litellm.acompletion` against the LiteLLM proxy (`LITELLM_API_BASE`; auth via `LITELLM_API_KEY` from `.env`).

Per call, returns `ModelOutput`: raw text, parsed JSON, latency ms, prompt/completion tokens, cost USD, error, parse_error flag.

Behavior details:

- **Retries:** up to `max_retries` with exponential backoff (`0.5 · 2^attempt` + jitter, cap 30s); 3× longer delay on 429/rate-limit errors.
- **Silent refusals:** empty/whitespace content (e.g. `finish_reason=content_filter`) is treated as a refusal, retried at most twice (deterministic filters rarely change), and recorded with error label `content_filter` or `empty_response (finish=…)`.
- **Cost:** read from LiteLLM's `_hidden_params.response_cost`, falling back to `litellm.completion_cost()`.
- **JSON parsing:** strict `json.loads` on raw text; failure sets `parse_error=True` (not a hard error).
- Per-model knobs forwarded: `temperature`, `max_tokens`, `extra_body` (e.g. `reasoning: {enabled: false}` for Grok), `safety_settings`; top-level `response_format` (JSON schema, strict mode) applied to all models.

### 3.4 Runner (`core/runner.py`)

- **Unit of work = (prompt, model) pair**, identified by composite key `"{prompt}__{model}"` (`pair_key`). Legacy single-prompt configs elide the prefix so old runs/columns stay compatible.
- Pairs run **sequentially**; within a pair, an `asyncio.Queue` is drained by `runtime.concurrency` workers (default 10).
- Progress logged every ~5% with rows/sec.
- **Graceful shutdown:** first SIGINT sets a flag — in-flight calls finish and are persisted; second SIGINT hard-exits.
- Correctness per row: `correct = (normalized prediction == normalized gold)`; `None` if either side is unmappable.

### 3.5 Verdict extraction (`core/normalize.py`)

- `extract_verdict`: walks a dotted path (e.g. `verdict` or `result.verdict`) through the parsed JSON.
- `normalize_verdict`: bools → `"true"`/`"false"`, else lowercase/strip, then map through `verdict.normalize`; unmapped → `None`.

### 3.6 Metrics (`core/metrics.py`)

Per pair (`ModelMetrics`):

| Metric | Definition |
|---|---|
| `total` | kept rows (refusals excluded) |
| `refused_count` | rows with empty provider content — excluded from all aggregates so safety blocks don't dilute the leaderboard |
| `coverage` | parseable verdicts / total |
| `accuracy` | correct / scoreable (both gold and prediction mappable) |
| `latency_p50/p95/mean_ms` | nearest-rank percentiles over successful call latencies |
| `cost_total/mean/per_correct_usd` | LiteLLM-reported cost; per-correct = total / correct |
| `prompt/completion_tokens_total` | summed usage |
| `error_count`, `parse_error_count` | transport errors vs. malformed JSON |
| `confusion` | full matrix; for 2-class tasks also tp/fp/fn/tn + precision/recall/F1 |

Key scoring principle: **malformed JSON or unmappable verdicts lower coverage instead of counting as wrong**; provider refusals are excluded entirely and tracked separately.

### 3.7 Reporting (`core/report.py`)

Each run writes `runs/<YYYY-MM-DDTHH-MM-SS>-<run_name>/`:

| File | Contents |
|---|---|
| `config.yaml` | Snapshot of the YAML used |
| `raw.csv` | One row per example; per-pair column block: `{pair}_verdict/_correct/_latency_ms/_cost_usd/_prompt_tokens/_completion_tokens/_error/_parse_error/_raw_output` (inputs truncated to 500 chars, raw output to 2000) |
| `summary.csv` | One row per pair: counts, accuracy, coverage, latency percentiles, costs, tokens, precision/recall/F1 |
| `summary.json` | Same aggregates nested + run metadata + config snapshot |
| `run.log` | Rotating log (10 MB × 5) |

`RawCsvWriter` keeps all rows in memory and rewrites the file on a throttled schedule (≥2s between flushes) so interrupts lose at most a few seconds of work while staying sub-O(N²) on large runs.

### 3.8 Run extension (`--add-to-run`)

Adds new models to a finished run without re-running existing ones:

1. Load the run's `config.yaml` snapshot; refuse unless the new YAML matches on dataset/prompts/verdict/response_format/sample_size (`diff_for_extension`).
2. Align `sample_size` to the actual `raw.csv` row count (the snapshot may record the pre-`--sample` default).
3. Run only the new models; `RawCsvWriter(extend_existing=True)` appends new column blocks in-place.
4. Historical models' `CallResult`s are reconstructed from `raw.csv` (`load_call_results_from_raw`) and merged so `summary.*` is regenerated over everything.
5. `--replace` allows overwriting a model whose earlier attempt errored.

## 4. CLI

```bash
python benchmarks/benchmark.py --config benchmarks/configs/example.yaml [flags]
```

| Flag | Purpose |
|---|---|
| `--config` | Path to YAML (required) |
| `--name` | Override `run_name` |
| `--sample N` | First N rows only |
| `--models a,b` | Subset of model names |
| `--prompts a,b` | Subset of prompt names |
| `--dry-run` | Validate + print plan (pairs, call count); no API calls |
| `--runs-dir` | Override runs output dir |
| `--add-to-run DIR` | Extend an existing run (requires `--models`) |
| `--replace` | With `--add-to-run`, overwrite existing model columns |

Exit codes: 0 success, 2 config/validation error.

## 5. Web UI (`app.py`, Streamlit)

```bash
./venv/bin/streamlit run benchmarks/app.py
```

Three sidebar modes:

- **Browse runs** — leaderboard (metric picker: accuracy, F1, latency p50, cost/correct) with Altair charts; per-row predictions with "disagreements only" / "errors only" filters; config snapshot and `run.log` tail.
- **Launch run** — pick a YAML, override run name / sample / model subset, stream harness log live via subprocess; dry-run option.
- **Configs** — view every YAML in `configs/`.

Summary loading is cached keyed on file mtimes.

## 6. Configuration reference

```yaml
run_name: my-benchmark

dataset:
  path: ../data/foo.csv          # relative to this YAML
  input_columns: [conversation]  # each must appear as {placeholder} in user_template
  gold_column: verdict
  id_column: conversation_id     # optional
  skip_columns: []               # dropped from Example.extra

# Either a single prompt:
prompt:
  system: |
    ...
  user_template: |
    {conversation}
# ...or multiple (A/B comparison; each pair keyed "{prompt}__{model}"):
# prompts:
#   - name: v4
#     system: ...
#     user_template: ...
#   - name: v5
#     ...

response_format:                 # optional; forwarded to every model
  type: json_schema
  json_schema: {name: ..., strict: true, schema: {...}}

verdict:
  extract_path: verdict          # dotted path into model JSON
  normalize:                     # case-insensitive → canonical label
    pass: PASS
    "1": PASS
    fail: FAIL
    "0": FAIL

models:
  - name: grok-4-fast            # display label (unique)
    model: litellm_proxy/openrouter/x-ai/grok-4-fast
    temperature: 0.0
    max_tokens: 1024
    extra_body: {reasoning: {enabled: false}}   # optional
    # safety_settings: [...]                    # optional (Gemini)

base_model: grok-4-fast          # optional; starred in summaries

runtime:
  concurrency: 10
  max_retries: 3
  timeout_seconds: 60
  sample_size: null              # null = full dataset; head-N otherwise
```

Existing configs: `example.yaml` (roleplay quality, 3 models), `memory_eval.yaml` / `memory_eval_smoke.yaml`, `v5_prompt.yaml`, `claude_prompt_deepseekv4_flash.yaml`, `prompt_comparison_deepseek_v4_flash.yaml` (multi-prompt A/B), `toy.yaml`.

## 7. Data preparation

`scripts/build_balanced_csv.py` builds a 100-row stratified slice (50 PASS + 50 FAIL, from the `score` column) out of `data/memory_eval_sample.csv` → `data/test_100_balanced.csv`. Needed because harness sampling is head-N, not random/stratified.

## 8. Dependencies & environment

- Python 3.10+ (uses `X | None` unions), repo venv.
- `litellm`, `python-dotenv`, `pyyaml`; UI adds `streamlit`, `pandas`, `altair`.
- Env: `LITELLM_API_KEY` (required), `LITELLM_API_BASE` (proxy endpoint).
- All model access goes through the LiteLLM proxy — no direct provider SDKs; cost accounting comes from the proxy.

## 9. Known limitations / future work

- **Sampling is head-N** — no seed/stratified sampling in the harness itself; balance must be baked into the CSV (§7).
- **Pairs run sequentially** — total wall time scales with prompts × models; cross-pair parallelism is a possible speedup.
- **Single-turn only** — one system + one user message per call; no multi-turn or tool-use benchmarks.
- **Exact-match scoring only** — no judge-LLM or partial-credit scoring; fine for classification, not for free-form outputs.
- **`raw.csv` truncation** — inputs at 500 chars, raw output at 2000; full outputs are not persisted.
- **In-memory raw writer** — whole result table held in RAM; fine at current scale (~1k rows × few pairs), would need a rethink for very large datasets.
- **No statistical significance testing** — leaderboard deltas are point estimates; bootstrap CIs would help for small samples.
- **No CI/tests** — the harness has no automated test suite.
