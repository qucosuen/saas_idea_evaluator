"""
Remote model backend using Groq or HuggingFace API.
Free tier Groq: 30 RPM, 6000 TPM, 500K tokens/day.
HuggingFace: Free inference API with rate limits.

Setup:
  Groq: export GROQ_API_KEY=your_key_here
  HuggingFace: export HF_TOKEN=your_key_here
"""

import json
import os
from pathlib import Path

import yaml


DEFAULT_PROVIDER_ORDER = ["groq", "huggingface"]


class ProviderProfile:
    """Profile data for a remote inference provider."""

    def __init__(self, backend: str, profile_path: str | Path | None = None):
        self.backend = backend
        self.profile_path = profile_path or Path("results/provider_profiles.json")
        self._data = None
        self.load()

    def load(self):
        """Load provider profile from JSON file."""
        if not self.profile_path.exists():
            self._data = None
            return
        try:
            with open(self.profile_path) as f:
                data = json.load(f)
            self._data = data.get("providers", {}).get(self.backend)
        except Exception:
            self._data = None

    @property
    def is_available(self) -> bool:
        return self._data is not None and self._data.get("available", False)

    @property
    def latency_avg_ms(self) -> float:
        return (
            self._data.get("latency_avg_ms", float("inf"))
            if self._data
            else float("inf")
        )

    @property
    def error_rate(self) -> float:
        return self._data.get("error_rate", 1.0) if self._data else 1.0

    @property
    def throttle_count(self) -> int:
        return self._data.get("throttle_count", 0) if self._data else 0

    @property
    def score(self) -> float:
        """Lower is better. Combines latency and error rate."""
        return self.latency_avg_ms * (1 + self.error_rate)

    def __repr__(self):
        return f"ProviderProfile({self.backend}, latency={self.latency_avg_ms}ms, error_rate={self.error_rate}, score={self.score:.0f})"


class ProviderSelector:
    """Selects best remote provider with automatic fallback on throttling."""

    def __init__(
        self,
        provider_order: list[str] | None = None,
        profile_path: str | Path | None = None,
    ):
        self.provider_order = provider_order or DEFAULT_PROVIDER_ORDER
        self.profile_path = profile_path or Path("results/provider_profiles.json")
        self._profiles = {}
        self._current_index = 0
        self._load_profiles()

    def _load_profiles(self):
        """Load all provider profiles."""
        for backend in self.provider_order:
            self._profiles[backend] = ProviderProfile(backend, self.profile_path)

    def select_best(self) -> str | None:
        """Select the best available provider based on profile scores."""
        available = []
        for backend in self.provider_order:
            profile = self._profiles[backend]
            if profile.is_available:
                available.append((backend, profile.score))

        if not available:
            return None

        available.sort(key=lambda x: x[1])
        return available[0][0]

    def get_provider(self) -> str | None:
        """Get the current provider (best by default)."""
        return self.select_best()

    def is_throttled(self, error: Exception) -> bool:
        """Check if error indicates rate limiting or unavailable service."""
        error_str = str(error).lower()
        throttle_indicators = [
            "rate limit",
            "throttle",
            "too many requests",
            "429",
            "rate_limit_exceeded",
            "requests per minute",
            "tpm limit",
            "rpm limit",
            "402",
            "payment required",
            "depleted",
            "insufficient credits",
        ]
        return any(ind in error_str for ind in throttle_indicators)

    def fallback(self) -> bool:
        """Fallback to next provider in order. Returns True if fallback successful."""
        if self._current_index + 1 >= len(self.provider_order):
            return False

        self._current_index += 1
        return True

    def get_current_backend(self) -> str | None:
        """Get the current backend name."""
        if self._current_index < len(self.provider_order):
            return self.provider_order[self._current_index]
        return None

    def reset(self):
        """Reset to first provider."""
        self._current_index = 0


class RemoteModel:
    """Drop-in replacement for llama_cpp using Groq or HuggingFace API."""

    def __init__(
        self, model: str | None = None, token: str | None = None, backend: str = None
    ):
        config = self._load_config()
        self.backend = (backend or config.get("remote_backend", "groq")).lower()

        if model is None:
            if self.backend == "huggingface":
                model = config.get("hf_model", "meta-llama/Llama-3.3-70b-Instruct")
            else:
                model = config.get("remote_model", "llama-3.3-70b-versatile")
        self.model = model

        if token:
            self.token = token
        elif self.backend == "groq":
            self.token = os.environ.get("GROQ_API_KEY", "")
            if not self.token:
                self.token = self._load_groq_key_from_config()
        else:
            self.token = os.environ.get("HF_TOKEN", "")
            if not self.token:
                self.token = self._load_hf_token_from_config()
            if not self.token:
                from huggingface_hub.hf_api import HfFolder

                token = HfFolder.get_token()
                if token:
                    self.token = token

        if not self.token:
            raise ValueError(
                f"API key not set for backend '{self.backend}'. "
                "Set GROQ_API_KEY or HF_TOKEN in config.yaml or environment."
            )

        if self.backend == "groq":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.token, base_url="https://api.groq.com/openai/v1"
            )
        else:
            from huggingface_hub import InferenceClient

            self.client = InferenceClient(provider="auto", api_key=self.token)

    def _load_config(self) -> dict:
        try:
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    return yaml.safe_load(f).get("model", {})
        except Exception:
            pass
        return {}

    def _load_groq_key_from_config(self) -> str:
        try:
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                return cfg.get("model", {}).get("groq_api_key", "")
        except Exception:
            pass
        return ""

    def _load_hf_token_from_config(self) -> str:
        try:
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                return cfg.get("model", {}).get("hf_token", "")
        except Exception:
            pass
        return ""

    def _load_model_from_config(self) -> str:
        try:
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                return cfg.get("model", {}).get("remote_model", "")
        except Exception:
            pass
        return ""

    def create_chat_completion(self, messages: list[dict], **kwargs) -> dict:
        """
        Match the llama_cpp create_chat_completion interface.
        Returns: {"choices": [{"message": {"content": str}}], "usage": {"completion_tokens": int}}
        """
        print(f"[RemoteModel] Using model: {self.model} (backend: {self.backend})")
        if self.backend == "groq":
            api_kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": min(kwargs.get("max_tokens", 500) * 3, 8192),
                "temperature": max(kwargs.get("temperature", 0.1), 0.01),
                "top_p": kwargs.get("top_p", 1.0),
            }
            resp = self.client.chat.completions.create(**api_kwargs)
            content = resp.choices[0].message.content or ""
            tokens = (
                resp.usage.completion_tokens if resp.usage else len(content.split())
            )
        else:
            from huggingface_hub import ChatCompletionOutput

            messages_hf = [
                {"role": m["role"], "content": m["content"]} for m in messages
            ]
            resp: ChatCompletionOutput = self.client.chat.completions.create(
                model=self.model,
                messages=messages_hf,
                max_tokens=kwargs.get("max_tokens", 500),
                temperature=kwargs.get("temperature", 0.1),
            )
            content = resp.choices[0].message.content or ""
            tokens = (
                resp.usage.completion_tokens if resp.usage else len(content.split())
            )

        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"completion_tokens": tokens},
        }


class MultiProviderModel:
    """
    Wrapper around RemoteModel that automatically selects the best provider
    and falls back on throttling errors.

    Usage:
        model = MultiProviderModel()  # Auto-selects best provider
        response = model.create_chat_completion(messages=[...])
    """

    def __init__(
        self,
        provider_order: list[str] | None = None,
        profile_path: str | Path | None = None,
        model: str | None = None,
        token: str | None = None,
    ):
        self.selector = ProviderSelector(
            provider_order=provider_order or DEFAULT_PROVIDER_ORDER,
            profile_path=profile_path,
        )
        self.model = model
        self.token = token
        self._current_model: RemoteModel | None = None
        self._init_current_model()

    def _init_current_model(self):
        """Initialize the RemoteModel for the current provider."""
        backend = self.selector.get_current_backend()
        if backend:
            model = self._get_model_for_backend(backend)
            self._current_model = RemoteModel(
                model=model,
                token=self.token,
                backend=backend,
            )
        else:
            self._current_model = None

    def _get_model_for_backend(self, backend: str) -> str:
        """Get the appropriate model name for a given backend from config."""
        config = self._load_config()
        if backend == "huggingface":
            return config.get("hf_model", "meta-llama/Llama-3.3-70b-Instruct")
        return config.get("remote_model", "llama-3.3-70b-versatile")

    def _load_config(self) -> dict:
        """Load config from config.yaml."""
        try:
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    return yaml.safe_load(f).get("model", {})
        except Exception:
            pass
        return {}

    @property
    def backend(self) -> str:
        return self.selector.get_current_backend() or "none"

    def create_chat_completion(self, messages: list[dict], **kwargs) -> dict:
        """
        Create chat completion with automatic fallback on throttling.
        """
        if not self._current_model:
            raise RuntimeError("No available remote model provider")

        try:
            return self._current_model.create_chat_completion(messages, **kwargs)
        except Exception as e:
            if self.selector.is_throttled(e):
                print(f"[RemoteModel] Throttled on {self.backend}: {e}")
                if self.selector.fallback():
                    print(
                        f"[RemoteModel] Falling back to: {self.selector.get_current_backend()}"
                    )
                    self._init_current_model()
                    if self._current_model:
                        return self._current_model.create_chat_completion(
                            messages, **kwargs
                        )
            raise
