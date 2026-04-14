#!/usr/bin/env python3
"""
Benchmark small models from the SlimLM leaderboard on local hardware.
Downloads GGUF versions and measures response time.

Usage: .venv/bin/python scripts/benchmark_small_models.py
"""
import json, os, time, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path("models/gguf")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {"name": "Qwen2.5-0.5B-Instruct", "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
     "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf", "params": "0.5B"},
    {"name": "Qwen2.5-1.5B-Instruct", "repo": None,
     "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf", "params": "1.5B"},
    {"name": "Qwen2.5-3B-Instruct", "repo": None,
     "file": "qwen2.5-3b-instruct-q4_k_m.gguf", "params": "3B"},
]

PROMPTS = [
    {"label": "short_question", "system": "Answer briefly.",
     "user": "Who was the first president of the USA?", "expected_tokens": 20},
    {"label": "project_eval_minimal", "system": "Score this project 1-10 on: pain, market, scalability. Output JSON only.",
     "user": "Evaluate: An app that helps freelancers track unpaid invoices.", "expected_tokens": 80},
    {"label": "project_eval_full",
     "system": "You are a project evaluator. Score on 10 dimensions (1-10) with a short reason each. Dimensions: ps=pain severity, pf=pain frequency, ea=existing alternatives, wp=willingness to pay, ms=market size, sc=scalability, pp=profitability, df=defensibility, tv=time to value, fm=founder-market fit. Also: os=overall score, sm=summary. Output ONLY valid JSON.",
     "user": "Evaluate: An app that helps freelancers track unpaid invoices and auto-send payment reminders.",
     "expected_tokens": 300},
]


def download_model(m):
    path = MODELS_DIR / m["file"]
    if path.exists():
        print(f"  Already cached: {path.name} ({path.stat().st_size / 1e9:.1f} GB)")
        return path
    if m["repo"] is None:
        print(f"  SKIP (no repo configured)")
        return path if path.exists() else None
    print(f"  Downloading {m['repo']}/{m['file']}...")
    hf_hub_download(repo_id=m["repo"], filename=m["file"], local_dir=str(MODELS_DIR))
    print(f"  Done: {path.stat().st_size / 1e9:.1f} GB")
    return path


def benchmark_model(model_path, model_name, params, n_threads):
    from llama_cpp import Llama
    print(f"\n{'='*60}")
    print(f"  {model_name} ({params}, {n_threads} threads)")
    print(f"  {model_path.name} ({model_path.stat().st_size / 1e9:.2f} GB)")
    print(f"{'='*60}")

    model = Llama(model_path=str(model_path), n_ctx=1024, n_threads=n_threads, verbose=False)

    results = []
    for prompt in PROMPTS:
        print(f"\n  [{prompt['label']}] (expect ~{prompt['expected_tokens']} tokens)")
        times = []
        for run in range(2):
            start = time.time()
            resp = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                max_tokens=prompt["expected_tokens"] + 50,
                temperature=0.0,
            )
            elapsed = time.time() - start
            output = resp["choices"][0]["message"]["content"]
            tok_count = resp["usage"]["completion_tokens"]
            tps = tok_count / elapsed if elapsed > 0 else 0
            times.append(elapsed)
            print(f"    Run {run+1}: {elapsed:.1f}s ({tok_count} tokens, {tps:.1f} tok/s)")

        avg = sum(times) / len(times)
        best = min(times)
        results.append({
            "prompt": prompt["label"],
            "avg_seconds": round(avg, 2),
            "best_seconds": round(best, 2),
            "tokens": tok_count,
            "tokens_per_second": round(tok_count / best, 1),
        })

    del model
    return results


def main():
    n_threads = max(2, (os.cpu_count() or 4) // 2)  # Half cores for <50% CPU
    print(f"CPU: {os.cpu_count()} cores, using {n_threads} threads")
    print(f"Models dir: {MODELS_DIR}")

    # Download all models
    print("\n--- Downloading models ---")
    for m in MODELS:
        print(f"\n{m['name']}:")
        download_model(m)

    # Benchmark
    all_results = {}
    for m in MODELS:
        path = MODELS_DIR / m["file"]
        if not path.exists():
            print(f"\nSkipping {m['name']} (not downloaded)")
            continue
        results = benchmark_model(path, m["name"], m["params"], n_threads)
        all_results[m["name"]] = {
            "params": m["params"],
            "file_size_gb": round(path.stat().st_size / 1e9, 2),
            "threads": n_threads,
            "benchmarks": results,
        }

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  BENCHMARK SUMMARY (i7-1185G7, {n_threads}/{os.cpu_count()} threads)")
    print(f"{'='*80}")
    print(f"{'Model':<28} {'Params':>6} {'Size':>6} {'Short Q':>8} {'3-dim':>8} {'Full 10-dim':>11} {'tok/s':>6}")
    print(f"{'-'*80}")
    for name, data in all_results.items():
        b = {r["prompt"]: r for r in data["benchmarks"]}
        short = b.get("short_question", {}).get("best_seconds", "-")
        mini = b.get("project_eval_minimal", {}).get("best_seconds", "-")
        full = b.get("project_eval_full", {}).get("best_seconds", "-")
        tps = b.get("short_question", {}).get("tokens_per_second", "-")
        short_s = f"{short:.1f}s" if isinstance(short, float) else short
        mini_s = f"{mini:.1f}s" if isinstance(mini, float) else mini
        full_s = f"{full:.1f}s" if isinstance(full, float) else full
        tps_s = f"{tps:.0f}" if isinstance(tps, float) else tps
        print(f"{name:<28} {data['params']:>6} {data['file_size_gb']:>5.1f}G {short_s:>8} {mini_s:>8} {full_s:>11} {tps_s:>6}")

    # Save results
    output = Path("results/benchmark_small_models.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump({
            "hardware": {
                "cpu": "Intel i7-1185G7 @ 3.00GHz",
                "cores": os.cpu_count(),
                "threads_used": n_threads,
                "ram_gb": 31,
            },
            "models": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
