"""
Model loader utility — shared across CLI and API.
"""

import os
from pathlib import Path


def find_model(model_path: str | None = None) -> Path:
    """Find the best available GGUF model."""
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Model not found: {model_path}")

    models_dir = Path("models/gguf")
    for name in [
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ]:
        p = models_dir / name
        if p.exists():
            return p
    raise FileNotFoundError("No GGUF model found in models/gguf/")


def load_model(model_path: Path, n_threads: int = 0):
    """Load a GGUF model with llama-cpp-python. Reads config from config.yaml."""
    from llama_cpp import Llama
    from src.config import get_model_config, get_cpu_percent

    cfg = get_model_config()
    if n_threads <= 0:
        cpu_pct = get_cpu_percent()
        total = os.cpu_count() or 4
        n_threads = max(1, round(total * cpu_pct / 100))

    return Llama(
        model_path=str(model_path),
        n_ctx=cfg["n_ctx"],
        n_threads=n_threads,
        n_gpu_layers=cfg["n_gpu_layers"],
        verbose=False,
    )


def load_remote_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Load a remote model via HuggingFace Inference API. Requires HF_TOKEN env var."""
    from src.pipeline.remote_model import RemoteModel
    return RemoteModel(model=model_name)
