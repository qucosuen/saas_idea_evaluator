"""
Run Stage 4 (Evaluation) - takes Stage 3 output and evaluates solutions.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.prompts import STAGE4_SYSTEM, STAGE4_USER
from src.pipeline.validators import validate_stage4


def main():
    parser = argparse.ArgumentParser(description="Run Stage 4 only")
    parser.add_argument(
        "--input", default="results/stage3_output.json", help="Stage 3 output file"
    )
    parser.add_argument("--remote", action="store_true", help="Use remote model")
    args = parser.parse_args()

    with open(args.input) as f:
        stage3 = json.load(f)

    job = stage3["job"]
    solutions = stage3["solutions"]

    solutions_text = json.dumps(solutions)

    if args.remote:
        from src.pipeline.model import load_remote_model

        model = load_remote_model()
    else:
        from src.pipeline.model import find_model, load_model

        path = find_model()
        print(f"Loading {path.name}...")
        model = load_model(path)

    print(f"Running Stage 4 for: {job}")

    start = time.time()
    system_msg = STAGE4_SYSTEM
    user_msg = STAGE4_USER.replace("{solutions}", solutions_text)

    try:
        resp = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=2000,
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    latency_ms = (time.time() - start) * 1000
    content = resp["choices"][0]["message"]["content"]
    tokens = resp["usage"]["completion_tokens"]

    print(f"Latency: {latency_ms:.0f}ms, Tokens: {tokens}")

    try:
        parsed = validate_stage4(content)
        print(f"\n✓ Valid - {len(parsed)} evaluations parsed")

        result = {
            "job": job,
            "stage": "stage4",
            "evaluations": [asdict(e) for e in parsed],
        }
        output_path = Path("results/stage4_output.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_path}")

    except Exception as e:
        print(f"\n✗ Validation error: {e}")
        with open("results/debug_stage4_raw.json", "w") as f:
            f.write(content)


if __name__ == "__main__":
    main()
