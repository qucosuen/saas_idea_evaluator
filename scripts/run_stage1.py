"""
Run Stage 1 only (workflow generator) for a job.
This is for debugging/testing the prompt without running full pipeline.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.prompts import STAGE1_SYSTEM, STAGE1_USER
from src.pipeline.validators import validate_stage1
from src.config import get_model_config


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 only")
    parser.add_argument("--job", default="DevOps Engineer", help="Job title")
    parser.add_argument(
        "--remote",
        action="store_true",
        default=True,
        help="Use HF Inference API (default)",
    )
    parser.add_argument(
        "--local", action="store_true", help="Use local model instead of remote"
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if cached")
    args = parser.parse_args()

    job = args.job

    if args.local:
        from src.pipeline.model import find_model, load_model

        path = find_model()
        print(f"Loading {path.name}...")
        model = load_model(path)
    else:
        from src.pipeline.model import load_remote_model
        from src.pipeline.remote_model import RemoteModel

        model = load_remote_model()

    print(f"Running Stage 1 for: {job}")

    start = time.time()
    system_msg = STAGE1_SYSTEM.replace("{job}", job)
    user_msg = STAGE1_USER.replace("{job}", job)

    try:
        resp = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=2500,
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    latency_ms = (time.time() - start) * 1000
    content = resp["choices"][0]["message"]["content"]
    tokens = resp["usage"]["completion_tokens"]

    print(f"Latency: {latency_ms:.0f}ms, Tokens: {tokens}")

    try:
        parsed = validate_stage1(content)
        print(f"\n✓ Valid - {len(parsed)} tasks parsed")

        result = {"job": job, "stage": "stage1", "tasks": [asdict(t) for t in parsed]}
        output_path = Path("results/stage1_output.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_path}")

    except Exception as e:
        print(f"\n✗ Validation error: {e}")
        with open("results/debug_stage1_raw.json", "w") as f:
            f.write(content)


if __name__ == "__main__":
    main()
