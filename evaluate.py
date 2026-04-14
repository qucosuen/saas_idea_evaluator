#!/usr/bin/env python3
"""
Project Idea Evaluator — GGUF Q4_K_M Inference
================================================
Evaluates startup/project ideas using Qwen2.5-3B-Instruct (Q4_K_M quantized).

Usage:
  # Evaluate a single idea
  python evaluate.py "An app that helps freelancers track unpaid invoices"

  # Interactive mode — keep evaluating ideas one after another
  python evaluate.py

  # Output raw JSON instead of formatted
  python evaluate.py --json "An app that helps freelancers track unpaid invoices"

Requirements:
  pip install llama-cpp-python huggingface-hub
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GGUF_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODELS_DIR = Path(__file__).parent / "models" / "gguf"

SYSTEM_PROMPT = """\
You are an expert startup and project evaluator. Given a project description, \
evaluate it across 10 business-viability dimensions. For each dimension, provide \
a score from 1 to 10 and a one-sentence reason.

Dimensions:
- pain_severity: How painful is the problem? (1=mild, 10=critical)
- pain_frequency: How often do users face this? (1=rarely, 10=daily)
- existing_alternatives: Are current solutions lacking? (1=well-solved, 10=no good alternatives)
- willingness_to_pay: Will users pay? (1=expects free, 10=pays premium)
- market_size: How large is the addressable market? (1=tiny niche, 10=massive)
- scalability: Can it scale with low marginal cost? (1=hard, 10=near-zero cost)
- profitability_potential: Revenue/margin potential? (1=low, 10=high)
- defensibility: How hard to replicate? (1=trivially copyable, 10=strong moat)
- time_to_value: How fast do users see value? (1=months, 10=instant)
- founder_market_fit_requirement: Domain expertise needed? (1=anyone, 10=deep domain)

Also provide:
- overall_score: A weighted average (1-10)
- summary: A 2-3 sentence overall assessment

Respond ONLY with valid JSON. No markdown, no extra text."""

DIMENSIONS = [
    "pain_severity", "pain_frequency", "existing_alternatives",
    "willingness_to_pay", "market_size", "scalability",
    "profitability_potential", "defensibility", "time_to_value",
    "founder_market_fit_requirement",
]


# ---------------------------------------------------------------------------
# Model download & loading
# ---------------------------------------------------------------------------

def get_model_path() -> Path:
    """Download the GGUF model if not cached, return local path."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local = MODELS_DIR / GGUF_FILE
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    print(f"Downloading {GGUF_REPO}/{GGUF_FILE} (~2 GB)...")
    print("This is a one-time download. The model will be cached locally.")
    hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE, local_dir=str(MODELS_DIR))
    print(f"Saved to {local}")
    return local


def load_model(model_path: Path, threads: int = 0):
    """Load the GGUF model with llama-cpp-python."""
    from llama_cpp import Llama
    n_threads = threads if threads > 0 else (os.cpu_count() or 4)
    print(f"Loading model ({model_path.name}, {n_threads} threads)...")
    model = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=n_threads,
        n_gpu_layers=0,
        verbose=False,
    )
    print("Ready.\n")
    return model


# ---------------------------------------------------------------------------
# JSON extraction & validation
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Extract a JSON object from model output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"(\{.*\})"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def validate(data: dict) -> list[str]:
    """Return list of issues (empty = valid)."""
    issues = []
    for dim in DIMENSIONS:
        if dim not in data:
            issues.append(f"Missing: {dim}")
        elif not isinstance(data[dim], dict):
            issues.append(f"{dim} should have score and reason")
        else:
            s = data[dim].get("score")
            if not isinstance(s, (int, float)) or not 1 <= s <= 10:
                issues.append(f"{dim}.score out of range")
            if "reason" not in data[dim]:
                issues.append(f"{dim} missing reason")
    if "overall_score" not in data:
        issues.append("Missing overall_score")
    if "summary" not in data:
        issues.append("Missing summary")
    return issues


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def evaluate_project(model, description: str, retries: int = 2) -> dict:
    """Generate a structured evaluation for a project description."""
    for attempt in range(1, retries + 1):
        start = time.time()
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Evaluate this project: {description}"},
            ],
            max_tokens=768,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        elapsed = time.time() - start
        raw = response["choices"][0]["message"]["content"]

        data = extract_json(raw)
        if data is not None:
            issues = validate(data)
            data["_meta"] = {
                "time_seconds": round(elapsed, 1),
                "attempt": attempt,
                "valid": len(issues) == 0,
                "issues": issues if issues else None,
            }
            return data

    return {"error": "Failed to parse JSON", "raw": raw[:500], "time_seconds": round(elapsed, 1)}



# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_result(data: dict):
    """Pretty-print an evaluation result."""
    if "error" in data:
        print(f"\n  Error: {data['error']}")
        if "raw" in data:
            print(f"  Raw output: {data['raw'][:200]}")
        return

    print(f"\n{'=' * 60}")
    print("  PROJECT EVALUATION")
    print(f"{'=' * 60}")

    for dim in DIMENSIONS:
        if dim in data and isinstance(data[dim], dict):
            score = data[dim].get("score", "?")
            reason = data[dim].get("reason", "")
            label = dim.replace("_", " ").title()
            if isinstance(score, (int, float)):
                bar = "█" * int(score) + "░" * (10 - int(score))
                print(f"\n  {label}")
                print(f"    {bar}  {score}/10")
            else:
                print(f"\n  {label}: {score}")
            print(f"    {reason}")

    overall = data.get("overall_score")
    if overall is not None:
        bar = "█" * int(overall) + "░" * (10 - int(overall))
        print(f"\n{'─' * 60}")
        print(f"  Overall: {bar}  {overall}/10")

    summary = data.get("summary")
    if summary:
        print(f"\n  {summary}")

    meta = data.get("_meta", {})
    t = meta.get("time_seconds", "?")
    valid = "✓" if meta.get("valid") else "✗"
    print(f"\n  [{t}s, schema {valid}]")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a project idea using Qwen2.5-3B (GGUF Q4_K_M)",
    )
    parser.add_argument("description", nargs="?", help="Project description to evaluate")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0=auto)")
    parser.add_argument("--model", help="Path to a custom GGUF model file")
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model) if args.model else get_model_path()
    model = load_model(model_path, threads=args.threads)

    # Single evaluation or interactive
    if args.description:
        result = evaluate_project(model, args.description)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_result(result)
    else:
        print("Interactive mode. Type a project idea and press Enter.")
        print("Type 'quit' to exit.\n")
        while True:
            try:
                desc = input("Project > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not desc or desc.lower() in ("quit", "exit", "q"):
                if desc.lower() in ("quit", "exit", "q"):
                    break
                continue
            result = evaluate_project(model, desc)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print_result(result)


if __name__ == "__main__":
    main()
