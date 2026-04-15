"""
Centralized configuration loader.
Reads config.yaml and provides typed access to model hyperparameters.
"""

import os
from pathlib import Path
from functools import lru_cache

import yaml


def _find_config() -> Path:
    """Walk up from cwd and src/ to find config.yaml."""
    for base in [Path.cwd(), Path(__file__).parent.parent]:
        p = base / "config.yaml"
        if p.exists():
            return p
    raise FileNotFoundError("config.yaml not found")


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    path = _find_config()
    with open(path) as f:
        return yaml.safe_load(f)


def reload():
    """Force reload config (useful after editing config.yaml)."""
    _load_raw.cache_clear()


def get_model_config() -> dict:
    """Return model loading config: path, n_ctx, n_gpu_layers."""
    raw = _load_raw()
    return {
        "path": raw["model"]["path"],
        "n_ctx": raw["model"].get("n_ctx", 2048),
        "n_gpu_layers": raw["model"].get("n_gpu_layers", 0),
    }


def get_cpu_percent() -> int:
    """Return CPU percent from config, env var, or default 50."""
    raw = _load_raw()
    env_val = os.environ.get("CPU_PERCENT")
    if env_val:
        return int(env_val)
    return raw.get("cpu", {}).get("percent", 50)


def get_generation_params(stage: str | None = None) -> dict:
    """
    Return generation parameters for a given stage.

    Priority: stages.<stage> overrides > defaults
    If stage is None, returns defaults only.

    Returns dict with keys: max_tokens, temperature, repeat_penalty, top_p
    """
    raw = _load_raw()
    defaults = dict(raw.get("defaults", {}))

    if stage and stage in raw.get("stages", {}):
        stage_cfg = raw["stages"][stage]
        defaults.update({k: v for k, v in stage_cfg.items() if v is not None})

    return defaults


def get_evaluator_params(fast: bool = False) -> dict:
    """Return generation params for the project evaluator."""
    raw = _load_raw()
    key = "evaluator_fast" if fast else "evaluator"
    cfg = raw.get(key, {})
    # Merge with defaults for any missing keys
    defaults = dict(raw.get("defaults", {}))
    defaults.update({k: v for k, v in cfg.items() if v is not None and k != "n_ctx"})
    return defaults


def get_evaluator_n_ctx(fast: bool = False) -> int:
    """Return n_ctx override for evaluator if specified."""
    raw = _load_raw()
    key = "evaluator_fast" if fast else "evaluator"
    return raw.get(key, {}).get("n_ctx", raw["model"].get("n_ctx", 2048))
