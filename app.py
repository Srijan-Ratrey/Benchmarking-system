"""Streamlit UI for the benchmark harness.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
BENCH_DIR = REPO_ROOT
CONFIGS_DIR = BENCH_DIR / "configs"
RUNS_DIR = BENCH_DIR / "runs"
PYTHON_BIN = sys.executable

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from core.runner import pair_key, split_pair_key  # noqa: E402

st.set_page_config(page_title="LLM Benchmarks", layout="wide")


LEGACY_PROMPT_LABEL = "default"
METRIC_CHOICES: dict[str, dict] = {
    "Accuracy": {"column": "accuracy", "higher_is_better": True, "fmt": "{:.3f}"},
    "F1": {"column": "f1", "higher_is_better": True, "fmt": "{:.3f}"},
    "Latency p50 (ms)": {"column": "latency_p50_ms", "higher_is_better": False, "fmt": "{:.0f}"},
    "Cost per correct ($)": {"column": "cost_per_correct_usd", "higher_is_better": False, "fmt": "{:.6f}"},
}


def _list_configs() -> list[Path]:
    if not CONFIGS_DIR.exists():
        return []
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def _list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        (p for p in RUNS_DIR.iterdir() if p.is_dir()),
        reverse=True,
    )


@st.cache_data(show_spinner=False)
def _load_summary_cached(
    run_dir_str: str, csv_mtime: float, json_mtime: float
) -> tuple[pd.DataFrame | None, dict | None]:
    run_dir = Path(run_dir_str)
    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    df = pd.read_csv(summary_csv) if summary_csv.exists() else None
    js = json.loads(summary_json.read_text()) if summary_json.exists() else None
    return df, js


def _load_summary(run_dir: Path) -> tuple[pd.DataFrame | None, dict | None]:
    csv_mtime = _safe_mtime(run_dir / "summary.csv")
    json_mtime = _safe_mtime(run_dir / "summary.json")
    return _load_summary_cached(str(run_dir), csv_mtime, json_mtime)


@st.cache_data(show_spinner=False)
def _load_raw_cached(run_dir_str: str, mtime: float) -> pd.DataFrame | None:
    raw = Path(run_dir_str) / "raw.csv"
    if not raw.exists():
        return None
    return pd.read_csv(raw)


def _load_raw(run_dir: Path) -> pd.DataFrame | None:
    return _load_raw_cached(str(run_dir), _safe_mtime(run_dir / "raw.csv"))


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _pairs_from_summary(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return summary.csv rows enriched with normalized `prompt`, `model`, `pair` cols.

    Works for both new multi-prompt summaries (already split) and legacy ones where
    only a `model` column existed.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if not {"prompt", "model", "pair"}.issubset(out.columns):
        keys = out["model"].astype(str)
        splits = keys.map(split_pair_key)
        out["prompt"] = splits.map(lambda t: t[0])
        out["model"] = splits.map(lambda t: t[1] or "")
        out["pair"] = keys
    else:
        out["prompt"] = out["prompt"].fillna("").astype(str)
        out["model"] = out["model"].astype(str)
        out["pair"] = out["pair"].astype(str)
    out["prompt"] = out["prompt"].replace("", LEGACY_PROMPT_LABEL)
    return out


def _pivot_metric(pairs_df: pd.DataFrame, column: str) -> pd.DataFrame:
    if pairs_df.empty or column not in pairs_df.columns:
        return pd.DataFrame()
    return pairs_df.pivot(index="prompt", columns="model", values=column)


def _prompt_order(pairs_df: pd.DataFrame, summary_js: dict | None) -> list[str]:
    """Use config snapshot prompt order when available, else first-seen order."""
    if summary_js:
        snap = summary_js.get("config_snapshot") or {}
        snap_prompts = snap.get("prompts") or []
        ordered = [str(p.get("name", "")) for p in snap_prompts if isinstance(p, dict)]
        ordered = [p for p in ordered if p]
        if ordered:
            known = set(pairs_df["prompt"].astype(str))
            return [p for p in ordered if p in known] + [
                p for p in pairs_df["prompt"].astype(str).drop_duplicates() if p not in ordered
            ]
    return list(pairs_df["prompt"].astype(str).drop_duplicates())


def _model_names_from_raw(df: pd.DataFrame) -> list[str]:
    return [
        c[: -len("_verdict")]
        for c in df.columns
        if c.endswith("_verdict") and c != "gold_verdict"
    ]


def _pairs_from_raw(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return list of (pair_key, prompt, model) parsed from raw.csv verdict columns."""
    out: list[tuple[str, str, str]] = []
    for key in _model_names_from_raw(df):
        prompt, model = split_pair_key(key)
        out.append((key, prompt or LEGACY_PROMPT_LABEL, model or key))
    return out


def _render_pivot(pairs_df: pd.DataFrame, metric_label: str) -> None:
    spec = METRIC_CHOICES[metric_label]
    column = spec["column"]
    higher = spec["higher_is_better"]
    pivot = _pivot_metric(pairs_df, column)
    if pivot.empty:
        st.caption(f"`{column}` not present in this run's summary.")
        return

    fmt_str = spec["fmt"]

    def highlight_best(row: pd.Series) -> list[str]:
        non_null = row.dropna()
        if non_null.empty:
            return ["" for _ in row]
        winner = non_null.idxmax() if higher else non_null.idxmin()
        return [
            "font-weight: 700; background-color: rgba(255, 215, 64, 0.35);"
            if col == winner else ""
            for col in row.index
        ]

    cmap = "Blues" if higher else "Blues_r"
    styled = (
        pivot.style
        .format(fmt_str, na_rep="—")
        .background_gradient(cmap=cmap, axis=None)
        .apply(highlight_best, axis=1)
    )
    st.dataframe(styled, width="stretch")
    st.caption(
        f"Cell = `{column}`. ★-equivalent highlight marks the winning model per prompt. "
        f"Heatmap shading: {'higher' if higher else 'lower'} is better."
    )


def _render_grouped_bars(pairs_df: pd.DataFrame, multi_prompt: bool) -> None:
    chart_metrics = [
        ("Accuracy", "accuracy"),
        ("Latency p50 (ms)", "latency_p50_ms"),
        ("Cost per correct ($)", "cost_per_correct_usd"),
    ]
    cols = st.columns(3)
    for (label, column), holder in zip(chart_metrics, cols):
        with holder:
            st.caption(label)
            if column not in pairs_df.columns:
                st.write("—")
                continue
            chart_df = pairs_df[["prompt", "model", column]].copy()
            chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")
            chart_df = chart_df.dropna(subset=[column])
            if chart_df.empty:
                st.write("—")
                continue
            enc = {
                "x": alt.X("model:N", title=""),
                "y": alt.Y(f"{column}:Q", title=label),
                "tooltip": ["prompt", "model", column],
            }
            if multi_prompt:
                enc["color"] = alt.Color("prompt:N", legend=alt.Legend(orient="bottom"))
                enc["xOffset"] = alt.XOffset("prompt:N")
            chart = alt.Chart(chart_df).mark_bar().encode(**enc).properties(height=240)
            st.altair_chart(chart, width="stretch")


def _render_leaderboard_section(
    summary_df: pd.DataFrame | None, summary_js: dict | None
) -> None:
    st.subheader("Leaderboard")
    if summary_df is None or summary_df.empty:
        st.warning("summary.csv not found — run may be incomplete.")
        return

    pairs_df = _pairs_from_summary(summary_df)
    prompts = _prompt_order(pairs_df, summary_js)
    models = list(pairs_df["model"].drop_duplicates())
    multi_prompt = len(prompts) > 1
    multi_model = len(models) > 1

    if multi_prompt and multi_model:
        metric_label = st.radio(
            "Pivot metric",
            list(METRIC_CHOICES.keys()),
            horizontal=True,
            key="leaderboard_pivot_metric",
        )
        _render_pivot(pairs_df, metric_label)
        st.divider()

    detail_cols = [
        "prompt", "model", "is_base", "total", "refused", "accuracy", "coverage",
        "latency_p50_ms", "latency_p95_ms",
        "cost_total_usd", "cost_per_correct_usd",
        "prompt_tokens_total", "completion_tokens_total",
        "error_count", "parse_error_count",
        "precision", "recall", "f1",
    ]
    detail_cols = [c for c in detail_cols if c in pairs_df.columns]

    if multi_prompt:
        for i, prompt in enumerate(prompts):
            sub = pairs_df[pairs_df["prompt"] == prompt][detail_cols].copy()
            if "accuracy" in sub.columns:
                sub = sub.sort_values("accuracy", ascending=False)
            with st.expander(f"Prompt: {prompt}  ({len(sub)} model{'s' if len(sub) != 1 else ''})", expanded=(i == 0)):
                st.dataframe(sub, width="stretch", hide_index=True)
    else:
        # single-prompt: legacy-style flat table without the prompt column noise
        flat = pairs_df[[c for c in detail_cols if c != "prompt"]].copy()
        if "accuracy" in flat.columns:
            flat = flat.sort_values("accuracy", ascending=False)
        st.dataframe(flat, width="stretch", hide_index=True)

    _render_grouped_bars(pairs_df, multi_prompt=multi_prompt)


def _render_confusion_section(
    pairs_df: pd.DataFrame, summary_js: dict | None
) -> None:
    """One confusion-matrix card per pair, grouped by prompt when multi-prompt."""
    if not summary_js:
        return
    models_data = summary_js.get("models") or {}
    if not models_data:
        return

    st.subheader("Confusion matrices (gold vs predicted)")
    st.caption(
        "Rows = gold verdict from the CSV, columns = model's predicted verdict. "
        "Heatmap shading shows row counts; the diagonal is where the model agrees with gold."
    )

    pair_lookup = {row["pair"]: row for _, row in pairs_df.iterrows()}
    prompts = _prompt_order(pairs_df, summary_js)
    multi_prompt = len(prompts) > 1

    for prompt in prompts:
        prompt_pairs = pairs_df[pairs_df["prompt"] == prompt]
        items: list[tuple[str, dict]] = []
        for _, row in prompt_pairs.iterrows():
            pair = row["pair"]
            data = models_data.get(pair) or models_data.get(row["model"])
            if data is None:
                continue
            card_label = row["model"] if multi_prompt else pair
            items.append((card_label, data))
        if not items:
            continue
        if multi_prompt:
            st.markdown(f"### Prompt: `{prompt}`")
        for chunk_start in range(0, len(items), 2):
            cols = st.columns(2)
            for offset, (label, data) in enumerate(items[chunk_start : chunk_start + 2]):
                with cols[offset]:
                    _render_one_confusion(label, data)

    # Pairs that weren't matched by name (shouldn't happen, but be safe).
    unmatched = [k for k in models_data.keys() if k not in pair_lookup]
    if unmatched:
        st.caption(f"Unmatched summary entries: {unmatched}")


def _render_one_confusion(model_name: str, model_summary: dict) -> None:
    confusion = (model_summary or {}).get("confusion") or {}
    classes = confusion.get("classes") or []
    matrix = confusion.get("matrix") or {}
    binary = confusion.get("binary") or {}

    st.markdown(f"**{model_name}**")
    if not classes or not matrix:
        st.info("No scoreable rows for this model (gold or prediction missing).")
        return

    cells = [
        {"gold": g, "pred": p, "count": int(matrix.get(g, {}).get(p, 0))}
        for g in classes
        for p in classes
    ]
    cell_df = pd.DataFrame(cells)

    heat = alt.Chart(cell_df).mark_rect().encode(
        x=alt.X("pred:N", title="Predicted", sort=classes),
        y=alt.Y("gold:N", title="Gold", sort=classes),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["gold", "pred", "count"],
    )
    labels = alt.Chart(cell_df).mark_text(
        baseline="middle", fontSize=16, fontWeight="bold"
    ).encode(
        x=alt.X("pred:N", sort=classes),
        y=alt.Y("gold:N", sort=classes),
        text="count:Q",
        color=alt.condition(
            alt.datum.count > (cell_df["count"].max() / 2 if len(cell_df) else 0),
            alt.value("white"),
            alt.value("black"),
        ),
    )
    st.altair_chart((heat + labels).properties(height=180), width="stretch")

    metric_cols = st.columns(4)
    accuracy = model_summary.get("accuracy")
    metric_cols[0].metric("Accuracy", _fmt_metric(accuracy, 3))
    if binary:
        metric_cols[1].metric("Precision", _fmt_metric(binary.get("precision"), 3))
        metric_cols[2].metric("Recall", _fmt_metric(binary.get("recall"), 3))
        metric_cols[3].metric("F1", _fmt_metric(binary.get("f1"), 3))
        st.caption(
            f"Positive class: `{binary.get('positive_class')}` · "
            f"TP={binary.get('tp')} · FP={binary.get('fp')} · "
            f"FN={binary.get('fn')} · TN={binary.get('tn')}"
        )
    else:
        st.caption(f"{len(classes)} classes — binary precision/recall not applicable.")

    scoreable = model_summary.get("scoreable") or 0
    total = model_summary.get("total") or 0
    refused = model_summary.get("refused_count", 0) or 0
    parse_errors = model_summary.get("parse_error_count", 0) or 0
    inputs = total + refused
    unparseable = max(total - scoreable, 0)

    parts = [f"{inputs} inputs", f"{scoreable} scored"]
    if unparseable:
        label = f"{unparseable} unparseable"
        if parse_errors:
            label += f" ({parse_errors} JSON parse error{'s' if parse_errors != 1 else ''})"
        parts.append(label)
    if refused:
        parts.append(f"{refused} refused by provider")
    st.caption(" · ".join(parts) + ".")


def _fmt_metric(value, ndigits: int) -> str:
    if value is None:
        return "—"
    return f"{value:.{ndigits}f}"


def _render_per_row_section(
    raw_df: pd.DataFrame | None, pairs_df: pd.DataFrame
) -> None:
    st.subheader("Per-row predictions")
    if raw_df is None or raw_df.empty:
        st.info("raw.csv not found.")
        return

    raw_pairs = _pairs_from_raw(raw_df)
    if not raw_pairs:
        st.info("No model verdict columns found in raw.csv.")
        return

    models = sorted({p[2] for p in raw_pairs})
    prompts_by_model: dict[str, list[str]] = {}
    for key, prompt, model in raw_pairs:
        prompts_by_model.setdefault(model, []).append(prompt)

    summary_prompts = _prompt_order(pairs_df, None) if not pairs_df.empty else []
    multi_prompt_run = len(summary_prompts) > 1 if summary_prompts else any(
        len(set(ps)) > 1 for ps in prompts_by_model.values()
    )

    top = st.columns([2, 3, 3])
    with top[0]:
        selected_model = st.selectbox("Model", models, key="per_row_model")
    available_prompts = prompts_by_model.get(selected_model, [])
    # Preserve summary's prompt order when possible.
    if summary_prompts:
        available_prompts = [p for p in summary_prompts if p in available_prompts] + [
            p for p in available_prompts if p not in summary_prompts
        ]
    with top[1]:
        if multi_prompt_run and len(available_prompts) > 1:
            selected_prompts = st.multiselect(
                "Prompts",
                available_prompts,
                default=available_prompts,
                key="per_row_prompts",
            )
        else:
            selected_prompts = available_prompts
    with top[2]:
        filter_options = ["All rows", "Errors only"]
        if multi_prompt_run and len(selected_prompts) >= 2:
            filter_options = ["All rows", "Prompts disagree", "All prompts wrong", "Errors only"]
        filter_mode = st.radio(
            "Filter", filter_options, horizontal=True, key="per_row_filter"
        )

    flag_cols = st.columns(3)
    with flag_cols[0]:
        show_inputs = st.checkbox("Show inputs", value=False, key="per_row_show_inputs")
    with flag_cols[1]:
        show_latency = st.checkbox("Show latency", value=False, key="per_row_show_latency")
    with flag_cols[2]:
        show_raw = st.checkbox("Include raw outputs", value=False, key="per_row_show_raw")

    if not selected_prompts:
        st.info("Pick at least one prompt.")
        return

    base_cols = ["id", "gold_verdict_raw", "gold_verdict"]
    base_cols = [c for c in base_cols if c in raw_df.columns]
    table = raw_df[base_cols].copy()
    if show_inputs:
        input_cols = [c for c in raw_df.columns if c.startswith("input_")]
        for c in input_cols:
            table[c] = raw_df[c].astype(str).str.slice(0, 200)

    verdict_cols: list[str] = []
    correct_cols: list[str] = []
    error_cols: list[str] = []
    source_verdict_cols: list[str] = []
    for prompt in selected_prompts:
        key = pair_key(prompt, selected_model)
        v_col = f"{key}_verdict"
        c_col = f"{key}_correct"
        l_col = f"{key}_latency_ms"
        e_col = f"{key}_error"
        r_col = f"{key}_raw_output"
        label = prompt if multi_prompt_run else selected_model
        if v_col in raw_df.columns:
            new_v = f"{label}__verdict"
            table[new_v] = raw_df[v_col]
            verdict_cols.append(new_v)
            source_verdict_cols.append(v_col)
        if c_col in raw_df.columns:
            new_c = f"{label}__correct"
            table[new_c] = raw_df[c_col]
            correct_cols.append(new_c)
        if show_latency and l_col in raw_df.columns:
            table[f"{label}__latency_ms"] = raw_df[l_col]
        if e_col in raw_df.columns:
            new_e = f"{label}__error"
            table[new_e] = raw_df[e_col].fillna("").astype(str)
            error_cols.append(new_e)
        if show_raw and r_col in raw_df.columns:
            table[f"{label}__raw_output"] = raw_df[r_col]

    multi_verdict = len(verdict_cols) >= 2
    if multi_verdict and multi_prompt_run:
        table["prompts_agree"] = table[verdict_cols].nunique(axis=1) <= 1

    n_total = len(table)

    if filter_mode == "Prompts disagree" and "prompts_agree" in table.columns:
        table = table[~table["prompts_agree"]]
    elif filter_mode == "All prompts wrong" and correct_cols:
        wrong_mask = (table[correct_cols].astype(str) == "0").all(axis=1)
        table = table[wrong_mask]
    elif filter_mode == "Errors only" and error_cols:
        err_mask = table[error_cols].apply(lambda s: s.str.len() > 0).any(axis=1)
        table = table[err_mask]

    extras = ""
    if multi_verdict and multi_prompt_run:
        total_disagree = int(
            (raw_df[source_verdict_cols].nunique(axis=1) > 1).sum()
        )
        extras = f" · {total_disagree} rows where prompts disagree"

    st.caption(f"Showing {len(table)} of {n_total} rows{extras}")
    st.dataframe(table, width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------

st.sidebar.title("LLM Benchmarks")
mode = st.sidebar.radio("Mode", ["Browse runs", "Launch run", "Configs"], index=0)
st.sidebar.markdown(f"`{RUNS_DIR.relative_to(REPO_ROOT)}`")


# ============================================================================
# Browse runs
# ============================================================================
if mode == "Browse runs":
    runs = _list_runs()
    if not runs:
        st.info("No runs found yet. Switch to **Launch run** to create one.")
        st.stop()

    run_labels = [p.name for p in runs]
    selected_label = st.sidebar.selectbox("Run", run_labels)
    run_dir = next(p for p in runs if p.name == selected_label)

    st.title(run_dir.name)
    summary_df, summary_js = _load_summary(run_dir)
    pairs_df = _pairs_from_summary(summary_df)

    meta_cols = st.columns(4)
    if summary_js:
        examples = summary_js["dataset"]["total_examples"]
        duration = summary_js.get("duration_seconds", "-")
    else:
        examples = "-"
        duration = "-"
    prompt_count = pairs_df["prompt"].nunique() if not pairs_df.empty else 0
    model_count = pairs_df["model"].nunique() if not pairs_df.empty else 0
    pair_count = len(pairs_df) if not pairs_df.empty else 0
    meta_cols[0].metric("Examples", examples)
    meta_cols[1].metric("Prompts", prompt_count)
    meta_cols[2].metric("Models", model_count)
    meta_cols[3].metric(
        "Pairs · Duration", f"{pair_count} · {duration}s" if duration != "-" else f"{pair_count}"
    )

    _render_leaderboard_section(summary_df, summary_js)
    _render_confusion_section(pairs_df, summary_js)

    raw_df = _load_raw(run_dir)
    _render_per_row_section(raw_df, pairs_df)

    with st.expander("Config snapshot"):
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            st.code(cfg_path.read_text(), language="yaml")
        else:
            st.write("No config snapshot.")

    with st.expander("Run log (last 200 lines)"):
        log_path = run_dir / "run.log"
        if log_path.exists():
            lines = log_path.read_text(errors="replace").splitlines()
            st.code("\n".join(lines[-200:]))
        else:
            st.write("No log file.")


# ============================================================================
# Launch run
# ============================================================================
elif mode == "Launch run":
    st.title("Launch a benchmark run")
    configs = _list_configs()
    if not configs:
        st.error(f"No YAML configs found in {CONFIGS_DIR.relative_to(REPO_ROOT)}")
        st.stop()

    cfg_choice = st.selectbox(
        "Config", [c.name for c in configs], index=0
    )
    cfg_path = next(c for c in configs if c.name == cfg_choice)

    with st.expander("Preview config", expanded=False):
        st.code(cfg_path.read_text(), language="yaml")

    col1, col2 = st.columns(2)
    with col1:
        run_name = st.text_input("Override run_name", value="")
    with col2:
        sample = st.number_input(
            "Sample size (0 = use config default)",
            min_value=0,
            value=0,
            step=1,
            help="0 = use the YAML's sample_size. Enter a positive number to override (e.g. 1000 for the full memory_eval dataset).",
        )

    col3, col4 = st.columns(2)
    with col3:
        models_csv = st.text_input("Models subset (comma-separated)", value="")
    with col4:
        prompts_csv = st.text_input("Prompts subset (comma-separated)", value="")

    dry_run = st.checkbox("Dry run (validate, no API calls)", value=False)

    if st.button("Run benchmark", type="primary"):
        cmd = [
            PYTHON_BIN,
            str(BENCH_DIR / "benchmark.py"),
            "--config", str(cfg_path),
        ]
        if run_name.strip():
            cmd += ["--name", run_name.strip()]
        if sample > 0:
            cmd += ["--sample", str(sample)]
        if models_csv.strip():
            cmd += ["--models", models_csv.strip()]
        if prompts_csv.strip():
            cmd += ["--prompts", prompts_csv.strip()]
        if dry_run:
            cmd += ["--dry-run"]

        st.write("**Command:** `" + " ".join(cmd) + "`")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log_box = st.empty()
        buf: list[str] = []

        with st.spinner("Running..."):
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            last_render = 0.0
            for line in proc.stdout:
                buf.append(line.rstrip())
                now = time.time()
                if now - last_render > 0.3:
                    log_box.code("\n".join(buf[-400:]))
                    last_render = now
            proc.wait()
            log_box.code("\n".join(buf[-400:]))

        if proc.returncode == 0:
            st.success("Run finished.")
            if not dry_run:
                st.info("Switch to **Browse runs** to view results.")
        else:
            st.error(f"Run failed with exit code {proc.returncode}")


# ============================================================================
# Configs viewer
# ============================================================================
else:
    st.title("Configs")
    configs = _list_configs()
    if not configs:
        st.warning("No YAML configs found.")
        st.stop()
    for cfg in configs:
        with st.expander(cfg.name, expanded=False):
            st.code(cfg.read_text(), language="yaml")
