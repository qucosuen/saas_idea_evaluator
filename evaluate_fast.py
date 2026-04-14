#!/usr/bin/env python3
"""
Ultra-Fast Project Evaluator — Optimized for <20s, <50% CPU
=============================================================
Techniques applied:
  1. Compact JSON schema — shorter keys = fewer output tokens (~40% reduction)
  2. Thread limiting — uses half of available cores to stay under 50% CPU
  3. Smaller context window — 1024 tokens (our prompts fit in ~600)
  4. Aggressive temperature — 0.0 (greedy decoding, no sampling overhead)
  5. Reduced max_tokens — 384 (compact schema fits in ~250 tokens)
  6. Supports Qwen2.5-1.5B for ~2x speedup over 3B when available

Usage:
  python evaluate_fast.py "An app that helps freelancers track unpaid invoices"
  python evaluate_fast.py                          # interactive mode
  python evaluate_fast.py --json "description"     # raw JSON output
  python evaluate_fast.py --model models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf "desc"

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

MODELS_DIR = Path(__file__).parent / "models" / "gguf"

# ---------------------------------------------------------------------------
# Compact schema — reduces output tokens by ~40%
# Instead of verbose keys like "pain_severity", we use "ps", "pf", etc.
# The model generates less text = faster inference.
# ---------------------------------------------------------------------------

COMPACT_SYSTEM = """\
You are a project evaluator. Score a project on 10 dimensions (1-10) with a short reason each.

Dimensions (use these exact keys):
ps=pain severity, pf=pain frequency, ea=existing alternatives (1=saturated,10=none),
wp=willingness to pay, ms=market size, sc=scalability,
pp=profitability potential, df=defensibility, tv=time to value,
fm=founder-market fit requirement (1=anyone,10=deep domain)

Also include: os=overall score (1-10), sm=summary (1-2 sentences).

Output ONLY valid JSON with short keys. Example:
{"ps":{"s":7,"r":"reason"},"pf":{"s":6,"r":"reason"},"ea":{"s":4,"r":"reason"},"wp":{"s":7,"r":"reason"},"ms":{"s":7,"r":"reason"},"sc":{"s":8,"r":"reason"},"pp":{"s":6,"r":"reason"},"df":{"s":3,"r":"reason"},"tv":{"s":8,"r":"reason"},"fm":{"s":3,"r":"reason"},"os":5.9,"sm":"summary"}"""

COMPACT_DIMS = ["ps", "pf", "ea", "wp", "ms", "sc", "pp", "df", "tv", "fm"]
DIM_NAMES = {
    "ps": "Pain Severity", "pf": "Pain Frequency",
    "ea": "Existing Alternatives", "wp": "Willingness to Pay",
    "ms": "Market Size", "sc": "Scalability",
    "pp": "Profitability Potential", "df": "Defensibility",
    "tv": "Time to Value", "fm": "Founder-Market Fit Req.",
}

# Also support full-key output as fallback
FULL_TO_COMPACT = {
    "pain_severity": "ps", "pain_frequency": "pf",
    "existing_alternatives": "ea", "willingness_to_pay": "wp",
    "market_size": "ms", "scalability": "sc",
    "profitability_potential": "pp", "defensibility": "df",
    "time_to_value": "tv", "founder_market_fit_requirement": "fm",
    "overall_score": "os", "summary": "sm",
}



def find_model() -> Path:
    """Find the best available GGUF model, preferring smaller for speed."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Prefer 1.5B (faster) over 3B
    candidates = [
        MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Download 3B as default
    from huggingface_hub import hf_hub_download
    print("Downloading Qwen2.5-3B-Instruct Q4_K_M (~2 GB, one-time)...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        local_dir=str(MODELS_DIR),
    )
    return MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"


def load_model(model_path: Path, threads: int = 0):
    """Load GGUF model with thread-limited config for <50% CPU."""
    from llama_cpp import Llama
    cpu_count = os.cpu_count() or 4
    # Use half the cores to stay under 50% CPU utilization
    n_threads = threads if threads > 0 else max(2, cpu_count // 2)
    print(f"Loading {model_path.name} ({n_threads}/{cpu_count} threads for <50% CPU)")
    model = Llama(
        model_path=str(model_path),
        n_ctx=1024,       # Smaller context = less memory, faster prefill
        n_threads=n_threads,
        n_gpu_layers=0,
        verbose=False,
    )
    return model, n_threads


def extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pat in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"(\{.*\})"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def normalize_keys(data: dict) -> dict:
    """Convert full keys to compact keys if model used verbose format."""
    normalized = {}
    for k, v in data.items():
        compact = FULL_TO_COMPACT.get(k, k)
        if compact in COMPACT_DIMS and isinstance(v, dict):
            # Normalize score/reason keys
            score = v.get("s") or v.get("score")
            reason = v.get("r") or v.get("reason", "")
            normalized[compact] = {"s": score, "r": reason}
        elif compact == "os":
            normalized["os"] = v
        elif compact == "sm":
            normalized["sm"] = v
        else:
            normalized[k] = v
    return normalized


def evaluate(model, description: str) -> dict:
    """Generate evaluation with compact schema."""
    start = time.time()
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": COMPACT_SYSTEM},
            {"role": "user", "content": f"Evaluate: {description}"},
        ],
        max_tokens=384,    # Compact JSON fits in ~250 tokens
        temperature=0.0,   # Greedy = fastest, deterministic
    )
    elapsed = time.time() - start
    raw = response["choices"][0]["message"]["content"]

    data = extract_json(raw)
    if data is None:
        return {"error": "JSON parse failed", "raw": raw[:300], "time": round(elapsed, 1)}

    data = normalize_keys(data)
    data["_time"] = round(elapsed, 1)
    return data



def print_result(data: dict):
    """Pretty-print evaluation."""
    if "error" in data:
        print(f"\n  Error: {data['error']}")
        if "raw" in data:
            print(f"  Raw: {data['raw'][:200]}")
        return

    t = data.get("_time", "?")
    print(f"\n{'=' * 60}")
    print(f"  PROJECT EVALUATION  ({t}s)")
    print(f"{'=' * 60}")

    for dim in COMPACT_DIMS:
        if dim in data and isinstance(data[dim], dict):
            score = data[dim].get("s", "?")
            reason = data[dim].get("r", "")
            name = DIM_NAMES.get(dim, dim)
            if isinstance(score, (int, float)):
                bar = "█" * int(score) + "░" * (10 - int(score))
                print(f"\n  {name}")
                print(f"    {bar}  {score}/10")
            else:
                print(f"\n  {name}: {score}")
            if reason:
                print(f"    {reason}")

    overall = data.get("os")
    if overall is not None:
        bar = "█" * int(overall) + "░" * (10 - int(overall))
        print(f"\n{'─' * 60}")
        print(f"  Overall: {bar}  {overall}/10")

    summary = data.get("sm")
    if summary:
        print(f"\n  {summary}")
    print(f"{'=' * 60}")


def to_full_json(data: dict) -> dict:
    """Convert compact keys back to full names for export."""
    COMPACT_TO_FULL = {v: k for k, v in FULL_TO_COMPACT.items()}
    result = {}
    for k, v in data.items():
        if k in COMPACT_TO_FULL and k in COMPACT_DIMS:
            full_key = COMPACT_TO_FULL[k]
            result[full_key] = {"score": v.get("s"), "reason": v.get("r", "")}
        elif k == "os":
            result["overall_score"] = v
        elif k == "sm":
            result["summary"] = v
        elif k == "_time":
            result["_generation_time_seconds"] = v
        else:
            result[k] = v
    return result


def main():
    parser = argparse.ArgumentParser(description="Fast project evaluator (<20s, <50% CPU)")
    parser.add_argument("description", nargs="?", help="Project description")
    parser.add_argument("--json", action="store_true", help="Output full JSON")
    parser.add_argument("--compact-json", action="store_true", help="Output compact JSON")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0=auto, half cores)")
    parser.add_argument("--model", help="Path to GGUF model file")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_model()
    model, threads = load_model(model_path, threads=args.threads)

    def run(desc):
        result = evaluate(model, desc)
        if args.json:
            print(json.dumps(to_full_json(result), indent=2, ensure_ascii=False))
        elif args.compact_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_result(result)

    if args.description:
        run(args.description)
    else:
        print("Interactive mode. Type a project idea, or 'quit' to exit.\n")
        while True:
            try:
                desc = input("Project > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if desc.lower() in ("quit", "exit", "q"):
                break
            if desc:
                run(desc)


if __name__ == "__main__":
    main()
