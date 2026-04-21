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


def load_remote_model(model_name: str | None = None, backend: str = None):
    """Load a remote model via Groq or HuggingFace API. Reads model from config.yaml if not provided."""
    import yaml
    from src.pipeline.remote_model import RemoteModel

    config_path = Path("config.yaml")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f).get("model", {})

    if model_name is None:
        if backend == "huggingface" or config.get("remote_backend") == "huggingface":
            model_name = config.get("hf_model", "meta-llama/Llama-3.3-70b-Instruct")
        else:
            model_name = config.get("remote_model", "llama-3.3-70b-versatile")

    return RemoteModel(model=model_name, backend=backend)


def load_remote_model_with_fallback(
    model_name: str | None = None,
    provider_order: list[str] | None = None,
):
    """
    Load a remote model with automatic provider selection and fallback on throttling.

    Args:
        model_name: Optional model name override (only used for groq, hf uses hf_model from config)
        provider_order: List of providers in fallback order (default: groq -> huggingface)

    Returns:
        MultiProviderModel instance with automatic fallback support
    """
    from src.pipeline.remote_model import MultiProviderModel

    return MultiProviderModel(
        provider_order=provider_order,
    )
