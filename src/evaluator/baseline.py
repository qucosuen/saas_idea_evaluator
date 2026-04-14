"""
Phase 1: Few-Shot Baseline Evaluation
======================================
Runs a small instruct model with few-shot examples to establish a baseline
for structured project evaluation output.

Usage:
  .venv/bin/python scripts/phase1_fewshot_baseline.py
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluator.inference import (
    DIMENSIONS,
    extract_json,
    validate_evaluation,
    load_hf_model,
    print_evaluation,
)

# ---------------------------------------------------------------------------
# Few-shot system prompt with 2 worked examples
# ---------------------------------------------------------------------------

FEWSHOT_SYSTEM_PROMPT = """You are an expert startup and project evaluator. Given a project description, evaluate it across 10 business-viability dimensions. For each dimension, provide a score from 1 to 10 and a one-sentence reason.

Respond ONLY with valid JSON. No markdown, no extra text.

Example 1:
User: Evaluate this project: A mobile app that lets dog owners find and book trusted pet sitters in their neighborhood with real-time GPS tracking.
Assistant: {"pain_severity":{"score":7,"reason":"Finding reliable pet care is stressful for dog owners"},"pain_frequency":{"score":6,"reason":"Needed for vacations, work trips, and busy days several times a year"},"existing_alternatives":{"score":4,"reason":"Rover and Wag already serve this market well"},"willingness_to_pay":{"score":7,"reason":"Pet owners regularly spend on pet care services"},"market_size":{"score":7,"reason":"Large pet ownership market globally"},"scalability":{"score":8,"reason":"Platform model scales with near-zero marginal cost"},"profitability_potential":{"score":6,"reason":"Commission-based revenue but competitive pricing pressure"},"defensibility":{"score":3,"reason":"Low switching costs and established competitors"},"time_to_value":{"score":8,"reason":"Users can book a sitter within minutes"},"founder_market_fit_requirement":{"score":3,"reason":"No deep domain expertise needed"},"overall_score":5.9,"summary":"Validated market with proven demand but faces strong incumbents. Differentiation through GPS tracking is minor. Would need a unique angle to compete."}

Example 2:
User: Evaluate this project: A Chrome extension that replaces all images on websites with pictures of cats.
Assistant: {"pain_severity":{"score":1,"reason":"This solves no real pain point"},"pain_frequency":{"score":1,"reason":"No recurring need for this"},"existing_alternatives":{"score":2,"reason":"Similar novelty extensions exist"},"willingness_to_pay":{"score":1,"reason":"Users expect novelty extensions to be free"},"market_size":{"score":2,"reason":"Tiny niche of novelty extension users"},"scalability":{"score":9,"reason":"Browser extension scales trivially"},"profitability_potential":{"score":1,"reason":"No viable monetization path"},"defensibility":{"score":1,"reason":"Trivially copyable in a weekend"},"time_to_value":{"score":9,"reason":"Instant entertainment value"},"founder_market_fit_requirement":{"score":1,"reason":"Anyone with basic web dev skills can build this"},"overall_score":2.8,"summary":"A fun novelty project with no business viability. No pain point, no willingness to pay, and trivially replicable. Fine as a hobby project but not a business."}

Now evaluate the following project. Respond with ONLY valid JSON, no other text."""


USER_TEMPLATE = "Evaluate this project: {description}"


def generate_fewshot(model, tokenizer, description: str, max_new_tokens: int = 768) -> str:
    """Generate evaluation using few-shot prompt."""
    import torch

    messages = [
        {"role": "system", "content": FEWSHOT_SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(description=description)},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_baseline(model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"):
    """Run Phase 1 few-shot baseline evaluation."""
    # Load test data
    test_file = Path(__file__).parent.parent / "data" / "test_descriptions.json"
    with open(test_file) as f:
        test_cases = json.load(f)

    # Load model
    print(f"=== Phase 1: Few-Shot Baseline ===")
    print(f"Model: {model_name}")
    print(f"Test cases: {len(test_cases)}")
    print()

    model, tokenizer = load_hf_model(model_name)

    # Run evaluations
    results = []
    metrics = {
        "json_valid": 0,
        "schema_compliant": 0,
        "total": len(test_cases),
        "scores_by_dimension": {dim: [] for dim in DIMENSIONS},
        "overall_scores": [],
        "timings": [],
    }

    for i, tc in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(test_cases)}] {tc['id']} ({tc['category']})")
        print(f"Description: {tc['description'][:80]}...")
        print(f"Expected: {tc['expected_quality']}")
        print(f"{'='*60}")

        # Try up to 2 attempts
        best_result = None
        for attempt in range(2):
            start = time.time()
            raw = generate_fewshot(model, tokenizer, tc["description"])
            elapsed = time.time() - start
            metrics["timings"].append(elapsed)

            print(f"\nAttempt {attempt+1} ({elapsed:.1f}s):")
            print(f"Raw output (first 300 chars): {raw[:300]}")

            data = extract_json(raw)
            if data is not None:
                best_result = data
                break
            else:
                print("  -> Failed to parse JSON")

        result_entry = {
            "id": tc["id"],
            "category": tc["category"],
            "expected_quality": tc["expected_quality"],
            "raw_output": raw[:500],
            "parsed": best_result is not None,
            "evaluation": best_result,
        }

        if best_result is not None:
            metrics["json_valid"] += 1
            valid, issues = validate_evaluation(best_result)
            result_entry["schema_valid"] = valid
            result_entry["schema_issues"] = issues

            if valid:
                metrics["schema_compliant"] += 1

            # Collect scores
            for dim in DIMENSIONS:
                if dim in best_result and isinstance(best_result[dim], dict):
                    score = best_result[dim].get("score")
                    if isinstance(score, (int, float)):
                        metrics["scores_by_dimension"][dim].append(score)

            if "overall_score" in best_result:
                metrics["overall_scores"].append(best_result["overall_score"])

            print_evaluation(best_result)
        else:
            result_entry["schema_valid"] = False
            result_entry["schema_issues"] = ["Failed to parse JSON"]
            print("  -> All attempts failed to produce valid JSON")

        results.append(result_entry)

    # Print summary
    print(f"\n\n{'='*60}")
    print("PHASE 1 BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"JSON valid:       {metrics['json_valid']}/{metrics['total']} ({100*metrics['json_valid']/metrics['total']:.0f}%)")
    print(f"Schema compliant: {metrics['schema_compliant']}/{metrics['total']} ({100*metrics['schema_compliant']/metrics['total']:.0f}%)")

    if metrics["timings"]:
        avg_time = sum(metrics["timings"]) / len(metrics["timings"])
        print(f"Avg generation:   {avg_time:.1f}s")

    if metrics["overall_scores"]:
        print(f"Overall scores:   {metrics['overall_scores']}")
        print(f"Score range:      {min(metrics['overall_scores']):.1f} - {max(metrics['overall_scores']):.1f}")

    # Score distribution per dimension
    print(f"\nScore distribution by dimension:")
    for dim in DIMENSIONS:
        scores = metrics["scores_by_dimension"][dim]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {dim:35s} avg={avg:.1f}  values={scores}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "phase1_baseline_results.json"
    with open(output_file, "w") as f:
        json.dump({"metrics": {k: v for k, v in metrics.items() if k != "scores_by_dimension"},
                    "score_distribution": {k: v for k, v in metrics["scores_by_dimension"].items()},
                    "results": results}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {output_file}")

    return metrics, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    args = parser.parse_args()
    run_baseline(args.model)
