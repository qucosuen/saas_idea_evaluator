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
    """Load a GGUF model with llama-cpp-python."""
    from llama_cpp import Llama
    threads = n_threads if n_threads > 0 else max(2, (os.cpu_count() or 4) // 2)
    return Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=threads,
        verbose=False,
    )
