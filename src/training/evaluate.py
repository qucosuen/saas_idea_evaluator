"""
Phase 3: Evaluate Fine-Tuned Model
====================================
Runs the fine-tuned model against the test set and compares with baseline.

Usage:
  .venv/bin/python scripts/phase3_evaluate.py \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --lora-path models/project-evaluator-lora
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluator.inference import (
    DIMENSIONS, extract_json, validate_evaluation,
    load_hf_model, generate_hf, print_evaluation,
)


def evaluate_model(model, tokenizer, test_cases: list, label: str) -> dict:
    """Run evaluation on test cases and return metrics."""
    metrics = {
        "label": label,
        "json_valid": 0,
        "schema_compliant": 0,
        "total": len(test_cases),
        "overall_scores": [],
        "timings": [],
        "results": [],
    }

    for i, tc in enumerate(test_cases):
        desc = tc["description"]
        print(f"\n[{i+1}/{len(test_cases)}] {tc['id']}")

        start = time.time()
        raw = generate_hf(model, tokenizer, desc)
        elapsed = time.time() - start
        metrics["timings"].append(elapsed)

        data = extract_json(raw)
        result = {"id": tc["id"], "parsed": data is not None, "time": elapsed}

        if data:
            metrics["json_valid"] += 1
            valid, issues = validate_evaluation(data)
            result["schema_valid"] = valid
            result["evaluation"] = data

            if valid:
                metrics["schema_compliant"] += 1
            if "overall_score" in data:
                metrics["overall_scores"].append(data["overall_score"])

            print(f"  ✓ JSON valid, schema {'✓' if valid else '✗'}, "
                  f"overall={data.get('overall_score', '?')} ({elapsed:.1f}s)")
        else:
            result["schema_valid"] = False
            print(f"  ✗ Failed to parse JSON ({elapsed:.1f}s)")

        metrics["results"].append(result)

    return metrics


def compare_metrics(baseline: dict, finetuned: dict):
    """Print comparison between baseline and fine-tuned metrics."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {baseline['label']} vs {finetuned['label']}")
    print(f"{'='*60}")

    for key in ["json_valid", "schema_compliant"]:
        b = baseline[key]
        f = finetuned[key]
        t = baseline["total"]
        print(f"{key:20s}: {b}/{t} ({100*b/t:.0f}%) → {f}/{t} ({100*f/t:.0f}%)")

    if baseline["timings"] and finetuned["timings"]:
        b_avg = sum(baseline["timings"]) / len(baseline["timings"])
        f_avg = sum(finetuned["timings"]) / len(finetuned["timings"])
        print(f"{'avg_time':20s}: {b_avg:.1f}s → {f_avg:.1f}s")

    if baseline["overall_scores"] and finetuned["overall_scores"]:
        b_scores = baseline["overall_scores"]
        f_scores = finetuned["overall_scores"]
        print(f"{'score_range':20s}: [{min(b_scores):.1f}-{max(b_scores):.1f}] → "
              f"[{min(f_scores):.1f}-{max(f_scores):.1f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--test-data", default="data/test_descriptions.json")
    args = parser.parse_args()

    # Load test data
    with open(args.test_data) as f:
        test_cases = json.load(f)

    # Evaluate base model
    print("=== Evaluating BASE model ===")
    base_model, base_tok = load_hf_model(args.model)
    base_metrics = evaluate_model(base_model, base_tok, test_cases, "baseline")
    del base_model  # Free memory

    # Evaluate fine-tuned model
    print("\n\n=== Evaluating FINE-TUNED model ===")
    ft_model, ft_tok = load_hf_model(args.model, lora_path=args.lora_path)
    ft_metrics = evaluate_model(ft_model, ft_tok, test_cases, "fine-tuned")

    # Compare
    compare_metrics(base_metrics, ft_metrics)

    # Save
    output = Path("results/phase3_evaluation.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump({"baseline": base_metrics, "finetuned": ft_metrics},
                  f, indent=2, default=str)
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
