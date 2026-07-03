# Benchmarks

Generic LLM benchmark harness. Point it at a labeled CSV and a list of models;
it runs every model through the same prompt via the LiteLLM proxy and reports
per-model **accuracy**, **latency**, and **cost**.

Accuracy is computed as exact-match between the model's verdict (extracted
from its JSON output) and the gold verdict column in the CSV — no judge LLM
is called.

## Web UI

A small Streamlit UI ships in `app.py`:

```bash
./venv/bin/streamlit run benchmarks/app.py
```

It has three modes (sidebar):
- **Browse runs** — leaderboard table + accuracy / latency / cost charts, per-row predictions with "disagreements only" and "errors only" filters, plus the saved config snapshot and tail of `run.log`.
- **Launch run** — pick a YAML config, optionally override `run_name` / sample size / model subset, stream the harness log live. Dry-run option for validation without API calls.
- **Configs** — quick view of every YAML in `configs/`.

## Quick start

```bash
# 1. Make sure LITELLM_API_KEY (and optionally LITELLM_API_BASE) are in .env
# 2. Author a YAML config (copy configs/example.yaml and edit)
# 3. Dry-run to validate
python benchmarks/benchmark.py --config benchmarks/configs/example.yaml --dry-run

# 4. Real run (full dataset)
python benchmarks/benchmark.py --config benchmarks/configs/example.yaml

# Small test: first 5 rows, one model
python benchmarks/benchmark.py \
    --config benchmarks/configs/example.yaml \
    --sample 5 --models grok-4-fast
```

Each run writes to `benchmarks/runs/<timestamp>-<run_name>/`:

| File           | What                                                                 |
|----------------|----------------------------------------------------------------------|
| `config.yaml`  | Snapshot of the YAML used for this run                               |
| `raw.csv`      | One row per example, with verdict / latency / cost / tokens / error per model |
| `summary.csv`  | One row per model with accuracy, latency p50/p95, total + per-correct cost |
| `summary.json` | Same aggregates nested + config snapshot + run metadata              |
| `run.log`      | Rotating log of the run                                              |

`raw.csv` is appended after every completed call, so a Ctrl+C leaves all
finished rows intact.

## Authoring a config

The YAML drives everything; the code itself is dataset-agnostic.

```yaml
dataset:
  path: ../data/foo.csv            # relative to the YAML file
  input_columns: [conversation]    # columns piped into the prompt template
  gold_column: verdict             # ground-truth column
  id_column: conversation_id       # optional, used as row id in raw.csv
  skip_columns: [internal_notes]   # columns to drop entirely

prompt:
  system: |
    ...
  user_template: |
    Evaluate this conversation:
    {conversation}                  # every input_column must appear as {placeholder}
    Return JSON {"verdict": "PASS"|"FAIL", ...}

verdict:
  extract_path: verdict             # dotted key in the model's JSON output
  normalize:                        # case-insensitive map → canonical label
    pass: PASS
    "1": PASS
    fail: FAIL
    "0": FAIL

models:
  - name: my-model                  # display label
    model: litellm_proxy/...        # LiteLLM model id
    temperature: 0.0
    max_tokens: 1024
    extra_body: {}
```

### Per-model knobs

`temperature`, `max_tokens`, and `extra_body` are optional per-model overrides.
`response_format` (top-level) is forwarded to every model.

### Verdict mapping

`verdict.normalize` is case-insensitive. The harness only counts a row as
"correct" when both the gold and model verdicts normalize to a known label.
Rows where the model returns malformed JSON or an unmappable verdict are
tracked as `parse_error` / lower `coverage` rather than counted as wrong.

## CLI flags

| Flag           | Purpose                                                       |
|----------------|---------------------------------------------------------------|
| `--config`     | Path to YAML                                                  |
| `--name`       | Override `run_name`                                           |
| `--sample N`   | Use only the first N rows (overrides `runtime.sample_size`)   |
| `--models a,b` | Run only the listed model names                               |
| `--dry-run`    | Validate config and print the plan; no API calls              |
| `--runs-dir`   | Override where run subfolders are written                     |
