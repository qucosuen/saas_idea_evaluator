"""
Run Stage 2 (Problem Analyzer) - takes Stage 1 output and identifies problems and current solutions for each step.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.prompts import (
    STAGE2A_SYSTEM,
    STAGE2A_USER,
    STAGE2B_SYSTEM,
    STAGE2B_USER,
)
from src.pipeline.validators import validate_stage2a, validate_stage2b
from src.pipeline.schemas import StepAnalysis


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 only")
    parser.add_argument(
        "--input", default="results/stage1_output.json", help="Stage 1 output file"
    )
    parser.add_argument(
        "--remote", action="store_true", default=True, help="Use remote model (default)"
    )
    parser.add_argument("--local", action="store_true", help="Use local model")
    parser.add_argument(
        "--stage2b-only", action="store_true", help="Run only Stage 2B (current solutions)"
    )
    parser.add_argument(
        "--stage2a-only", action="store_true", help="Run only Stage 2A (problems)"
    )
    parser.add_argument(
        "--output", default=None, help="Output file path"
    )
    args = parser.parse_args()

    # Load Stage 1 output
    with open(args.input) as f:
        stage1 = json.load(f)

    job = stage1["job"]
    tasks = stage1["tasks"]

    # Build workflow description from Stage 1 output
    workflow_text = json.dumps(tasks)

    if args.local:
        from src.pipeline.model import find_model, load_model

        path = find_model()
        print(f"Loading {path.name}...")
        model = load_model(path)
    else:
        from src.pipeline.model import load_remote_model_with_fallback

        model = load_remote_model_with_fallback()
        print(f"Using remote model with fallback: {model.backend}")

    print(f"Running Stage 2 for: {job}")

    # Stage 2B: Current Solutions first
    start = time.time()
    system_msg = STAGE2B_SYSTEM
    user_msg = STAGE2B_USER.format(workflow=workflow_text)

    try:
        resp = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=4000,
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    latency_ms = (time.time() - start) * 1000
    content = resp["choices"][0]["message"]["content"]
    tokens = resp["usage"]["completion_tokens"]

    print(f"Stage 2B (Current Solutions): {latency_ms:.0f}ms, {tokens} tokens")

    try:
        solutions = validate_stage2b(content)
        print(f"✓ Valid - {len(solutions)} solutions parsed")
    except Exception as e:
        print(f"\n✗ Stage 2B Validation error: {e}")
        with open("results/debug_stage2b_raw.json", "w") as f:
            f.write(content)
        return

    if args.stage2b_only:
        analyses = [
            StepAnalysis(
                task_number=sol.task_number,
                step_number=sol.step_number,
                step_description=sol.step_description,
                current_solution=sol.current_solution,
                problem="",
            )
            for sol in solutions
        ]
        result = {
            "job": job,
            "stage": "stage2b",
            "step_analyses": [asdict(a) for a in analyses],
        }
        output_path = Path("results/stage2b_output.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_path}")
        return

    # Stage 2A: Problems second
    start = time.time()
    system_msg = STAGE2A_SYSTEM
    user_msg = STAGE2A_USER.format(workflow=workflow_text)

    try:
        resp = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=4000,
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    latency_ms = (time.time() - start) * 1000
    content = resp["choices"][0]["message"]["content"]
    tokens = resp["usage"]["completion_tokens"]

    print(f"Stage 2A (Problems): {latency_ms:.0f}ms, {tokens} tokens")

    try:
        problems = validate_stage2a(content)
        print(f"✓ Valid - {len(problems)} problems parsed")
    except Exception as e:
        print(f"\n✗ Stage 2A Validation error: {e}")
        with open("results/debug_stage2a_raw.json", "w") as f:
            f.write(content)
        return

    if args.stage2a_only:
        analyses = []
        for p in sorted(problems, key=lambda x: (x.task_number, x.step_number)):
            analyses.append(
                StepAnalysis(
                    task_number=p.task_number,
                    step_number=p.step_number,
                    step_description=p.step_description,
                    current_solution="",
                    problem=p.problem,
                )
            )
        result = {
            "job": job,
            "stage": "stage2a",
            "step_analyses": [asdict(a) for a in analyses],
        }
        output_path = Path("results/stage2a_output.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_path}")
        return

    # Merge solutions and problems into StepAnalysis
    problem_map = {(p.task_number, p.step_number): p.problem for p in problems}
    analyses = []
    for sol in solutions:
        problem = problem_map.get((sol.task_number, sol.step_number), "")
        analyses.append(
            StepAnalysis(
                task_number=sol.task_number,
                step_number=sol.step_number,
                step_description=sol.step_description,
                current_solution=sol.current_solution,
                problem=problem,
            )
        )

    result = {
        "job": job,
        "stage": "stage2",
        "step_analyses": [asdict(a) for a in analyses],
    }
    output_path = Path("results/stage2_output.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
