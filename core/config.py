"""YAML config loading and validation for the benchmark harness."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    path: Path
    input_columns: list[str]
    gold_column: str
    id_column: str | None = None
    skip_columns: list[str] = field(default_factory=list)


@dataclass
class PromptConfig:
    system: str
    user_template: str
    name: str = "default"


@dataclass
class VerdictConfig:
    extract_path: str
    normalize: dict[str, str]


@dataclass
class ModelConfig:
    name: str
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    safety_settings: list[dict[str, Any]] | None = None


@dataclass
class RuntimeConfig:
    concurrency: int = 10
    max_retries: int = 3
    timeout_seconds: int = 60
    sample_size: int | None = None


@dataclass
class RunConfig:
    run_name: str
    dataset: DatasetConfig
    prompts: list[PromptConfig]
    verdict: VerdictConfig
    models: list[ModelConfig]
    base_model: str | None
    runtime: RuntimeConfig
    response_format: dict[str, Any] | None
    raw: dict[str, Any]

    @property
    def prompt(self) -> PromptConfig:
        """Legacy single-prompt accessor — returns the first prompt."""
        return self.prompts[0]


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")

    return _build_config(raw, config_dir=path.parent)


def load_snapshot_config(snapshot_path: Path, fallback_dataset_dir: Path) -> RunConfig:
    """Load a run's `config.yaml` snapshot. Snapshots live in `runs/<ts>/` but the
    YAML inside still has its original relative paths (e.g. `../data/foo.csv`).
    We patch `dataset.path` to resolve under `fallback_dataset_dir` so the load
    succeeds without needing the original configs/ tree to still exist."""
    with open(snapshot_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("Snapshot config root must be a mapping")

    ds = raw.get("dataset")
    if isinstance(ds, dict):
        raw_path = ds.get("path")
        if raw_path and not Path(raw_path).is_absolute():
            candidate = (snapshot_path.parent / raw_path).resolve()
            if not candidate.exists():
                # Fall back to looking in the configs/ dir of the active run.
                ds["path"] = str((fallback_dataset_dir / Path(raw_path)).resolve())

    return _build_config(raw, config_dir=snapshot_path.parent)


def diff_for_extension(
    snapshot_cfg: RunConfig, new_cfg: RunConfig
) -> list[tuple[str, Any, Any]]:
    """Return [(field_path, snapshot_value, new_value), ...] for every field that
    must match when extending a run with a new model. Empty list = compatible."""

    sn_ds, new_ds = snapshot_cfg.dataset, new_cfg.dataset
    sn_v, new_v = snapshot_cfg.verdict, new_cfg.verdict

    pairs: list[tuple[str, Any, Any]] = [
        # Compare by basename: snapshots store dataset.path relative to the run
        # folder while the new YAML resolves relative to configs/. The same logical
        # file ends up with different absolute paths.
        ("dataset.path (basename)", sn_ds.path.name, new_ds.path.name),
        ("dataset.input_columns", sn_ds.input_columns, new_ds.input_columns),
        ("dataset.gold_column", sn_ds.gold_column, new_ds.gold_column),
        ("dataset.id_column", sn_ds.id_column, new_ds.id_column),
        ("dataset.skip_columns", sorted(sn_ds.skip_columns), sorted(new_ds.skip_columns)),
        ("verdict.extract_path", sn_v.extract_path, new_v.extract_path),
        ("verdict.normalize", sn_v.normalize, new_v.normalize),
        ("response_format", snapshot_cfg.response_format, new_cfg.response_format),
        ("runtime.sample_size", snapshot_cfg.runtime.sample_size, new_cfg.runtime.sample_size),
    ]

    sn_prompts = {p.name: p for p in snapshot_cfg.prompts}
    new_prompts = {p.name: p for p in new_cfg.prompts}
    pairs.append(("prompts.names", sorted(sn_prompts), sorted(new_prompts)))
    for name in sorted(sn_prompts.keys() & new_prompts.keys()):
        pairs.append(
            (f"prompts[{name}].system", sn_prompts[name].system, new_prompts[name].system)
        )
        pairs.append(
            (
                f"prompts[{name}].user_template",
                sn_prompts[name].user_template,
                new_prompts[name].user_template,
            )
        )
    return [(path, a, b) for path, a, b in pairs if a != b]


def _build_config(raw: dict[str, Any], config_dir: Path) -> RunConfig:
    _require(raw, "run_name", str)
    _require(raw, "dataset", dict)
    _require(raw, "verdict", dict)
    _require(raw, "models", list)

    if "prompt" in raw and "prompts" in raw:
        raise ConfigError("Use either 'prompt' (single) or 'prompts' (list), not both")
    if "prompt" not in raw and "prompts" not in raw:
        raise ConfigError("config.prompt or config.prompts is required")

    dataset = _build_dataset(raw["dataset"], config_dir)
    prompts = _build_prompts(raw, dataset.input_columns)
    verdict = _build_verdict(raw["verdict"])
    models = _build_models(raw["models"])
    runtime = _build_runtime(raw.get("runtime", {}))

    base_model = raw.get("base_model")
    if base_model and base_model not in {m.name for m in models}:
        raise ConfigError(
            f"base_model '{base_model}' not found in models: {[m.name for m in models]}"
        )

    response_format = raw.get("response_format")
    if response_format is not None and not isinstance(response_format, dict):
        raise ConfigError("response_format must be a mapping if provided")

    return RunConfig(
        run_name=raw["run_name"],
        dataset=dataset,
        prompts=prompts,
        verdict=verdict,
        models=models,
        base_model=base_model,
        runtime=runtime,
        response_format=response_format,
        raw=raw,
    )


def _build_dataset(raw: dict[str, Any], config_dir: Path) -> DatasetConfig:
    _require(raw, "path", str)
    _require(raw, "input_columns", list)
    _require(raw, "gold_column", str)

    path = Path(raw["path"])
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    if not path.exists():
        raise ConfigError(f"Dataset CSV not found: {path}")

    input_columns = [str(c) for c in raw["input_columns"]]
    if not input_columns:
        raise ConfigError("dataset.input_columns must list at least one column")

    return DatasetConfig(
        path=path,
        input_columns=input_columns,
        gold_column=str(raw["gold_column"]),
        id_column=str(raw["id_column"]) if raw.get("id_column") else None,
        skip_columns=[str(c) for c in raw.get("skip_columns", [])],
    )


def _build_prompt(
    raw: dict[str, Any], input_columns: list[str], name: str = "default"
) -> PromptConfig:
    _require(raw, "user_template", str, ctx=f"prompt[{name}]")
    system = str(raw.get("system", ""))
    user_template = str(raw["user_template"])

    placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", user_template))
    missing = [c for c in input_columns if c not in placeholders]
    if missing:
        raise ConfigError(
            f"prompt[{name}].user_template is missing placeholders for input columns: {missing}"
        )

    return PromptConfig(system=system, user_template=user_template, name=name)


_PROMPT_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _build_prompts(
    raw: dict[str, Any], input_columns: list[str]
) -> list[PromptConfig]:
    if "prompts" in raw:
        items = raw["prompts"]
        if not isinstance(items, list) or not items:
            raise ConfigError("prompts must be a non-empty list")
        prompts: list[PromptConfig] = []
        seen: set[str] = set()
        for i, entry in enumerate(items):
            if not isinstance(entry, dict):
                raise ConfigError(f"prompts[{i}] must be a mapping")
            _require(entry, "name", str, ctx=f"prompts[{i}]")
            name = entry["name"]
            if not _PROMPT_NAME_RE.match(name):
                raise ConfigError(
                    f"prompts[{i}].name must match [a-zA-Z0-9_-]+, got '{name}'"
                )
            if "__" in name:
                raise ConfigError(
                    f"prompts[{i}].name must not contain '__' (reserved separator)"
                )
            if name in seen:
                raise ConfigError(f"Duplicate prompt name '{name}'")
            seen.add(name)
            prompts.append(_build_prompt(entry, input_columns, name=name))
        return prompts

    return [_build_prompt(raw["prompt"], input_columns, name="default")]


def _build_verdict(raw: dict[str, Any]) -> VerdictConfig:
    _require(raw, "extract_path", str)
    _require(raw, "normalize", dict)
    normalize = {str(k).strip().lower(): str(v) for k, v in raw["normalize"].items()}
    if not normalize:
        raise ConfigError("verdict.normalize must map at least one value")
    return VerdictConfig(extract_path=str(raw["extract_path"]), normalize=normalize)


def _build_models(raw: list[Any]) -> list[ModelConfig]:
    if not raw:
        raise ConfigError("models list must contain at least one entry")

    seen: set[str] = set()
    models: list[ModelConfig] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"models[{i}] must be a mapping")
        _require(entry, "name", str, ctx=f"models[{i}]")
        _require(entry, "model", str, ctx=f"models[{i}]")
        name = entry["name"]
        if name in seen:
            raise ConfigError(f"Duplicate model name '{name}'")
        seen.add(name)
        models.append(ModelConfig(
            name=name,
            model=entry["model"],
            temperature=entry.get("temperature"),
            max_tokens=entry.get("max_tokens"),
            extra_body=entry.get("extra_body") or {},
            safety_settings=entry.get("safety_settings"),
        ))
    return models


def _build_runtime(raw: dict[str, Any]) -> RuntimeConfig:
    sample = raw.get("sample_size")
    if sample is not None and (not isinstance(sample, int) or sample <= 0):
        raise ConfigError("runtime.sample_size must be a positive integer or null")
    return RuntimeConfig(
        concurrency=int(raw.get("concurrency", 10)),
        max_retries=int(raw.get("max_retries", 3)),
        timeout_seconds=int(raw.get("timeout_seconds", 60)),
        sample_size=sample,
    )


def _require(raw: dict[str, Any], key: str, type_: type, ctx: str = "config") -> None:
    if key not in raw:
        raise ConfigError(f"{ctx}.{key} is required")
    if not isinstance(raw[key], type_):
        raise ConfigError(
            f"{ctx}.{key} must be {type_.__name__}, got {type(raw[key]).__name__}"
        )
