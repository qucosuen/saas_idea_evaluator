#!/usr/bin/env python3
"""Run Stage 2 using existing Stage 1 results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.model import find_model, load_model
from src.pipeline.runner import PipelineRunner
from src.pipeline.validators import validate_stage2a, validate_stage2b
from src.pipeline.prompts import (
    STAGE2A_SYSTEM,
    STAGE2A_USER,
    STAGE2B_SYSTEM,
    STAGE2B_USER,
)
from src.config import get_generation_params


def main():
    stage1_path = Path(__file__).parent / "results" / "stage1_output.json"

    with open(stage1_path) as f:
        stage1_data = json.load(f)

    tasks = stage1_data.get("tasks", [])
    workflow_text = json.dumps(tasks, indent=2)

    print(f"Loaded stage 1 results with {len(tasks)} tasks")

    # Use local model directly (remote providers are rate limited)
    print("Using local model...")
    model_path = find_model()
    model = load_model(model_path)
    print(f"Using local model: {model_path.name}")

    runner = PipelineRunner(model, max_retries=1)

    params_a = get_generation_params("stage2a_problems")
    params_b = get_generation_params("stage2b_solutions")

    print("\n--- Running Stage 2A (Problems) ---")
    raw_a, ms_a, tok_a = runner._generate(
        STAGE2A_SYSTEM, STAGE2A_USER.format(workflow=workflow_text), "stage2a_problems"
    )
    print(f"Stage 2A: {ms_a:.0f}ms, {tok_a} tokens")

    try:
        parsed_a = validate_stage2a(raw_a)
        print(f"Stage 2A: Valid output with {len(parsed_a)} entries")
        output_a = {
            "stage": "stage2a",
            "valid": True,
            "latency_ms": ms_a,
            "tokens": tok_a,
            "results": [vars(p) for p in parsed_a],
        }
    except Exception as e:
        print(f"Stage 2A: Validation failed - {e}")
        output_a = {
            "stage": "stage2a",
            "valid": False,
            "latency_ms": ms_a,
            "tokens": tok_a,
            "raw": raw_a[:1000],
        }

    print("\n--- Running Stage 2B (Solutions) ---")
    raw_b, ms_b, tok_b = runner._generate(
        STAGE2B_SYSTEM, STAGE2B_USER.format(workflow=workflow_text), "stage2b_solutions"
    )
    print(f"Stage 2B: {ms_b:.0f}ms, {tok_b} tokens")

    try:
        parsed_b = validate_stage2b(raw_b)
        print(f"Stage 2B: Valid output with {len(parsed_b)} entries")
        output_b = {
            "stage": "stage2b",
            "valid": True,
            "latency_ms": ms_b,
            "tokens": tok_b,
            "results": [vars(p) for p in parsed_b],
        }
    except Exception as e:
        print(f"Stage 2B: Validation failed - {e}")
        output_b = {
            "stage": "stage2b",
            "valid": False,
            "latency_ms": ms_b,
            "tokens": tok_b,
            "raw": raw_b[:1000],
        }

    result = {
        "job": stage1_data.get("job", "unknown"),
        "stage2a": output_a,
        "stage2b": output_b,
        "total_latency_ms": ms_a + ms_b,
        "total_tokens": tok_a + tok_b,
    }

    output_path = Path(__file__).parent / "results" / "stage2_output.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Total latency: {ms_a + ms_b:.0f}ms, Total tokens: {tok_a + tok_b}")


if __name__ == "__main__":
    main()
