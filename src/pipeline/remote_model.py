"""
Remote model backend using HuggingFace Inference API.
Free, no GPU needed, OpenAI-compatible chat completions.

Setup: export HF_TOKEN=hf_your_token_here
Get a token at: https://huggingface.co/settings/tokens
(needs "Make calls to Inference Providers" permission)
"""

import os
import time


class RemoteModel:
    """Drop-in replacement for llama_cpp.Llama using HuggingFace Inference API."""

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", token: str | None = None):
        self.model = model
        self.token = token or os.environ.get("HF_TOKEN", "")

        # Fallback: read from cached huggingface-cli login
        if not self.token:
            for path in ["~/.cache/huggingface/token", "~/.huggingface/token"]:
                expanded = os.path.expanduser(path)
                if os.path.exists(expanded):
                    self.token = open(expanded).read().strip()
                    break

        if not self.token:
            raise ValueError(
                "HF_TOKEN not set. Run: .venv/bin/huggingface-cli login"
            )

        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=self.token)

    def create_chat_completion(self, messages: list[dict], **kwargs) -> dict:
        """
        Match the llama_cpp create_chat_completion interface.
        Accepts: messages, max_tokens, temperature, repeat_penalty, top_p
        Returns: {"choices": [{"message": {"content": str}}], "usage": {"completion_tokens": int}}
        """
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(kwargs.get("max_tokens", 500) * 3, 4000),  # remote can handle more
            "temperature": max(kwargs.get("temperature", 0.1), 0.01),
            "top_p": kwargs.get("top_p", 1.0),
        }

        resp = self.client.chat.completions.create(**api_kwargs)

        content = resp.choices[0].message.content or ""
        tokens = resp.usage.completion_tokens if resp.usage else len(content.split())

        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"completion_tokens": tokens},
        }
