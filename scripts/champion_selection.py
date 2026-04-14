#!/usr/bin/env python3
"""
Champion Model Selection — Quality-Latency Composite Metric
=============================================================
Evaluates all candidate models on a combined metric that balances:
  1. Response quality (JSON validity, schema compliance, score calibration, reason quality)
  2. Response time (measured latency on local hardware)

Metric: QLE (Quality-Latency Efficiency)
  QLE = Quality_Score * Latency_Penalty

Where:
  Quality_Score (0-100) = weighted combination of:
    - JSON validity rate (0 or 1) × 25
    - Schema compliance (0 or 1) × 20
    - Score calibration: 1 - |predicted_overall - expected_overall| / 10 × 25
    - Reason specificity: avg reason length > 20 chars × 15
    - Score differentiation: std(dimension_scores) > 1.0 × 15

  Latency_Penalty (0-1) = 1 / (1 + latency_seconds / T_ref)
    where T_ref = 30s (reference time — penalty is 0.5 at 30s, 0.75 at 10s, 0.91 at 3s)

  QLE range: 0-100 (higher is better)

This metric penalizes slow models but doesn't eliminate them — a model that's
2x slower needs to be meaningfully better on quality to win.
"""

import json, time, re, os, sys, statistics
from pathlib import Path
from datetime import datetime

# Reference latency for penalty function (seconds)
T_REF = 30.0

TEST_CASES = [
    {
        "description": "An app that helps freelancers track unpaid invoices and auto-send payment reminders via email and SMS.",
        "expected_tier": "strong",
        "expected_overall_range": (6.5, 8.5),
    },
    {
        "description": "A browser extension that changes all website backgrounds to the color blue.",
        "expected_tier": "weak",
        "expected_overall_range": (1.5, 4.0),
    },
    {
        "description": "An AI-powered platform that automatically generates and files patent applications for inventors.",
        "expected_tier": "strong",
        "expected_overall_range": (6.5, 9.0),
    },
]

DIMS = ["pain_severity","pain_frequency","existing_alternatives","willingness_to_pay",
        "market_size","scalability","profitability_potential","defensibility",
        "time_to_value","founder_market_fit_requirement"]

# Also accept compact keys
COMPACT_MAP = {"ps":"pain_severity","pf":"pain_frequency","ea":"existing_alternatives",
    "wp":"willingness_to_pay","ms":"market_size","sc":"scalability",
    "pp":"profitability_potential","df":"defensibility","tv":"time_to_value",
    "fm":"founder_market_fit_requirement"}

SYSTEM = ("You are a project evaluator. Score on 10 dimensions (1-10) with a short reason each. "
    "Dimensions: pain_severity, pain_frequency, existing_alternatives, willingness_to_pay, "
    "market_size, scalability, profitability_potential, defensibility, time_to_value, "
    "founder_market_fit_requirement. Also: overall_score (1-10), summary. Output ONLY valid JSON.")


def extract_json(text):
    try: return json.loads(text)
    except: pass
    for pat in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"(\{.*\})"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except: continue
    return None

def normalize_keys(data):
    """Convert compact keys to full keys."""
    out = {}
    for k, v in data.items():
        full = COMPACT_MAP.get(k, k)
        if full in DIMS and isinstance(v, dict):
            score = v.get("score") or v.get("s")
            reason = v.get("reason") or v.get("r", "")
            out[full] = {"score": score, "reason": reason}
        elif full in DIMS and isinstance(v, (int, float)):
            out[full] = {"score": v, "reason": ""}
        elif k in ("overall_score", "os"):
            out["overall_score"] = v
        elif k in ("summary", "sm"):
            out["summary"] = v
        else:
            out[k] = v
    return out

def score_quality(data, expected_range):
    """Score a single evaluation on quality (0-100)."""
    if data is None:
        return 0.0, {"json_valid": False}

    data = normalize_keys(data)
    details = {"json_valid": True}

    # Schema compliance: all 10 dims present with score and reason
    present = sum(1 for d in DIMS if d in data and isinstance(data[d], dict) and "score" in data[d])
    schema_ok = present == 10 and "overall_score" in data
    details["schema_compliance"] = schema_ok
    details["dims_present"] = present

    # Score calibration: is overall_score in expected range?
    overall = data.get("overall_score", 5.0)
    if isinstance(overall, (int, float)):
        lo, hi = expected_range
        mid = (lo + hi) / 2
        distance = abs(overall - mid) / 10.0
        calibration = max(0, 1.0 - distance * 2)  # 0 at distance=0.5, 1 at distance=0
    else:
        calibration = 0.0
    details["overall_score"] = overall
    details["calibration"] = round(calibration, 3)

    # Reason specificity: average reason length
    reasons = []
    for d in DIMS:
        if d in data and isinstance(data[d], dict):
            r = data[d].get("reason", "")
            if isinstance(r, str):
                reasons.append(len(r))
    avg_reason_len = statistics.mean(reasons) if reasons else 0
    reason_quality = min(1.0, avg_reason_len / 40.0)  # 1.0 at 40+ chars
    details["avg_reason_length"] = round(avg_reason_len, 1)
    details["reason_quality"] = round(reason_quality, 3)

    # Score differentiation: std of dimension scores
    scores = []
    for d in DIMS:
        if d in data and isinstance(data[d], dict):
            s = data[d].get("score")
            if isinstance(s, (int, float)):
                scores.append(s)
    score_std = statistics.stdev(scores) if len(scores) > 1 else 0
    differentiation = min(1.0, score_std / 2.5)  # 1.0 at std >= 2.5
    details["score_std"] = round(score_std, 2)
    details["differentiation"] = round(differentiation, 3)

    # Weighted composite
    quality = (
        (1.0 if True else 0.0) * 25 +  # JSON valid (already True if we got here)
        (1.0 if schema_ok else 0.0) * 20 +
        calibration * 25 +
        reason_quality * 15 +
        differentiation * 15
    )
    details["quality_score"] = round(quality, 1)
    return quality, details

def latency_penalty(seconds):
    """Penalty function: 1/(1 + t/T_ref). Returns 0-1."""
    return 1.0 / (1.0 + seconds / T_REF)

def evaluate_model(model_path, model_name):
    """Run all test cases and compute QLE."""
    from llama_cpp import Llama
    n_threads = max(2, (os.cpu_count() or 4) // 2)
    model = Llama(model_path=str(model_path), n_ctx=1024, n_threads=n_threads, verbose=False)

    results = []
    for tc in TEST_CASES:
        # Warm-up run (discard)
        model.create_chat_completion(
            messages=[{"role":"system","content":"Say hi."},{"role":"user","content":"Hi"}],
            max_tokens=5, temperature=0.0)

        start = time.time()
        resp = model.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Evaluate this project: {tc['description']}"},
            ],
            max_tokens=400, temperature=0.0,
        )
        elapsed = time.time() - start
        raw = resp["choices"][0]["message"]["content"]
        tokens = resp["usage"]["completion_tokens"]

        data = extract_json(raw)
        quality, details = score_quality(data, tc["expected_overall_range"])
        penalty = latency_penalty(elapsed)
        qle = quality * penalty

        results.append({
            "description": tc["description"][:60] + "...",
            "expected_tier": tc["expected_tier"],
            "latency_seconds": round(elapsed, 1),
            "tokens": tokens,
            "quality_score": round(quality, 1),
            "latency_penalty": round(penalty, 3),
            "qle": round(qle, 1),
            "details": details,
        })

    del model

    avg_quality = statistics.mean(r["quality_score"] for r in results)
    avg_latency = statistics.mean(r["latency_seconds"] for r in results)
    avg_penalty = latency_penalty(avg_latency)
    avg_qle = avg_quality * avg_penalty

    return {
        "model": model_name,
        "avg_quality": round(avg_quality, 1),
        "avg_latency_seconds": round(avg_latency, 1),
        "avg_latency_penalty": round(avg_penalty, 3),
        "avg_qle": round(avg_qle, 1),
        "test_results": results,
    }


def main():
    models_dir = Path("models/gguf")
    candidates = [
        ("Qwen2.5-0.5B-Instruct", models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        ("Qwen2.5-1.5B-Instruct", models_dir / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        ("Qwen2.5-3B-Instruct", models_dir / "qwen2.5-3b-instruct-q4_k_m.gguf"),
    ]

    print(f"Champion Selection — QLE (Quality-Latency Efficiency)")
    print(f"Hardware: i7-1185G7, {os.cpu_count()} cores, {max(2,(os.cpu_count() or 4)//2)} threads")
    print(f"T_ref: {T_REF}s (latency penalty = 0.5 at {T_REF}s)")
    print(f"Test cases: {len(TEST_CASES)}")

    all_results = []
    for name, path in candidates:
        if not path.exists():
            print(f"\nSkipping {name} (not found)")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        result = evaluate_model(path, name)
        all_results.append(result)

        for tr in result["test_results"]:
            print(f"\n  {tr['description']}")
            print(f"    Quality: {tr['quality_score']}/100  Latency: {tr['latency_seconds']}s  Penalty: {tr['latency_penalty']:.3f}  QLE: {tr['qle']}")
            d = tr["details"]
            print(f"    Schema: {'✓' if d.get('schema_compliance') else '✗'}  Calibration: {d.get('calibration',0):.2f}  Reasons: {d.get('avg_reason_length',0):.0f} chars  Differentiation: {d.get('differentiation',0):.2f}")

        print(f"\n  AVERAGE: Quality={result['avg_quality']}/100  Latency={result['avg_latency_seconds']}s  QLE={result['avg_qle']}")

    # Champion selection
    print(f"\n\n{'='*60}")
    print(f"  CHAMPION SELECTION")
    print(f"{'='*60}")
    print(f"\n{'Model':<28} {'Quality':>8} {'Latency':>8} {'Penalty':>8} {'QLE':>8}")
    print(f"{'-'*60}")
    for r in sorted(all_results, key=lambda x: x["avg_qle"], reverse=True):
        print(f"{r['model']:<28} {r['avg_quality']:>7.1f} {r['avg_latency_seconds']:>7.1f}s {r['avg_latency_penalty']:>7.3f} {r['avg_qle']:>7.1f}")

    champion = max(all_results, key=lambda x: x["avg_qle"])
    print(f"\n  CHAMPION: {champion['model']} (QLE = {champion['avg_qle']})")

    # Save results
    output = {
        "metric": "QLE (Quality-Latency Efficiency)",
        "formula": "QLE = Quality_Score × Latency_Penalty, where Latency_Penalty = 1/(1 + t/T_ref)",
        "T_ref_seconds": T_REF,
        "quality_components": {
            "json_validity": {"weight": 25, "description": "Output parses as valid JSON"},
            "schema_compliance": {"weight": 20, "description": "All 10 dimensions present with score and reason"},
            "score_calibration": {"weight": 25, "description": "Overall score falls within expected range for the idea quality"},
            "reason_specificity": {"weight": 15, "description": "Average reason length >= 40 characters"},
            "score_differentiation": {"weight": 15, "description": "Standard deviation of dimension scores >= 2.5"},
        },
        "hardware": {"cpu": "Intel i7-1185G7", "threads": max(2,(os.cpu_count() or 4)//2), "backend": "llama.cpp GGUF Q4_K_M"},
        "timestamp": datetime.utcnow().isoformat(),
        "champion": champion["model"],
        "champion_qle": champion["avg_qle"],
        "results": all_results,
    }

    out_path = Path("results/champion_selection.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()
