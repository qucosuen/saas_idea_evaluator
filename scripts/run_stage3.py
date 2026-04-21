"""
Run Stage 3 (Solutions) - takes Stage 2 output and proposes automation solutions.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.prompts import STAGE3_SYSTEM, STAGE3_USER
from src.pipeline.validators import validate_stage3


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 only")
    parser.add_argument(
        "--input", default="results/stage2_output.json", help="Stage 2 output file"
    )
    parser.add_argument("--remote", action="store_true", help="Use remote model")
    args = parser.parse_args()

    with open(args.input) as f:
        stage2 = json.load(f)

    job = stage2["job"]
    problems = stage2["problems"]

    problems_text = json.dumps(problems)

    if args.remote:
        from src.pipeline.model import load_remote_model

        model = load_remote_model()
    else:
        from src.pipeline.model import find_model, load_model

        path = find_model()
        print(f"Loading {path.name}...")
        model = load_model(path)

    print(f"Running Stage 3 for: {job}")

    start = time.time()
    system_msg = STAGE3_SYSTEM
    user_msg = STAGE3_USER.replace("{problems}", problems_text)

    try:
        resp = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=3000,
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    latency_ms = (time.time() - start) * 1000
    content = resp["choices"][0]["message"]["content"]
    tokens = resp["usage"]["completion_tokens"]

    print(f"Latency: {latency_ms:.0f}ms, Tokens: {tokens}")

    try:
        parsed = validate_stage3(content)
        print(f"\n✓ Valid - {len(parsed)} solutions parsed")

        result = {
            "job": job,
            "stage": "stage3",
            "solutions": [asdict(s) for s in parsed],
        }
        output_path = Path("results/stage3_output.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_path}")

    except Exception as e:
        print(f"\n✗ Validation error: {e}")
        with open("results/debug_stage3_raw.json", "w") as f:
            f.write(content)


if __name__ == "__main__":
    main()
