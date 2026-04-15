"""
Benchmark Stage 1 format score and duplication score only.

Usage:
  # Default: top 5 worst duplication jobs from last benchmark
  python scripts/benchmark_format_dup.py

  # Custom jobs via CLI
  python scripts/benchmark_format_dup.py "Library Assistant" "Janitor" "Accountant"

  # From a file (one job per line)
  python scripts/benchmark_format_dup.py --file jobs.txt

  # All 30 benchmark jobs
  python scripts/benchmark_format_dup.py --all
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama
from src.pipeline.prompts import STAGE1_SYSTEM, STAGE1_USER
from src.benchmark.metrics import score_stage1_workflow
from src.config import get_model_config, get_generation_params, get_cpu_percent

DATASET_PATH = "src/benchmark/dataset.json"
LAST_BENCHMARK = "results/benchmark/stage1_benchmark.json"

# Top 5 worst duplication jobs (default)
DEFAULT_JOBS = [
    "Library Assistant",
    "Janitor",
    "Sales Representative",
    "Real Estate Agent",
    "University Registrar",
]


def load_expected_patterns(jobs: list[str]) -> dict[str, list[str]]:
    """Load expected_patterns from dataset for the given jobs."""
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    lookup = {d["job"]: d["expected_patterns"] for d in dataset}
    return {j: lookup.get(j, []) for j in jobs}


def get_top_dup_jobs(n: int = 5) -> list[str]:
    """Get top N jobs by duplication from last benchmark results."""
    if not Path(LAST_BENCHMARK).exists():
        return DEFAULT_JOBS[:n]
    with open(LAST_BENCHMARK) as f:
        data = json.load(f)
    ranked = sorted(data["detailed_scores"], key=lambda s: s["total_duplicates"], reverse=True)
    return [s["job"] for s in ranked[:n]]


def run(jobs: list[str]):
    patterns = load_expected_patterns(jobs)
    model_cfg = get_model_config()
    gen_params = get_generation_params("stage1_workflow")
    cpu_pct = get_cpu_percent()

    total_cpus = os.cpu_count() or 4
    n_threads = max(1, round(total_cpus * cpu_pct / 100))
    print(f"Model: {model_cfg['path']} ({n_threads} threads)")
    print(f"Gen params: {gen_params}")
    print(f"Jobs:  {len(jobs)}\n")

    model = Llama(
        model_path=model_cfg["path"],
        n_ctx=model_cfg["n_ctx"],
        n_threads=n_threads,
        n_gpu_layers=model_cfg["n_gpu_layers"],
        verbose=False,
    )
    model.create_chat_completion(
        messages=[{"role": "user", "content": "Hi"}], max_tokens=5, temperature=0.0
    )

    results = []
    for i, job in enumerate(jobs):
        start = time.perf_counter()
        resp = model.create_chat_completion(
            messages=[
                {"role": "system", "content": STAGE1_SYSTEM},
                {"role": "user", "content": STAGE1_USER.format(job=job)},
            ],
            **gen_params,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        output = resp["choices"][0]["message"]["content"]

        # Post-process: clean mild duplication like "Step 1: Step 1: content"
        output = re.sub(r"(Step\s+\d+\s*:\s*)(Step\s+\d+\s*:\s*)+", r"\1", output)

        scores = score_stage1_workflow(output, patterns.get(job, []))

        results.append({
            "job": job,
            "format_score": scores["format_score"],
            "step_count": scores["step_count"],
            "duplication_penalty": scores["duplication_penalty"],
            "total_duplicates": scores["total_duplicates"],
            "emptiness_penalty": scores["emptiness_penalty"],
            "empty_steps": scores["empty_steps"],
            "stage_score": scores["stage_score"],
            "latency_ms": round(latency_ms, 1),
            "output": output,
        })

        flags = ""
        if scores["duplication_penalty"] > 0: flags += " ⚠ DUP"
        if scores["empty_steps"] > 0: flags += f" ⚠ EMPTY({scores['empty_steps']})"
        print(f"[{i+1}/{len(jobs)}] {job:<35} format={scores['format_score']:.2f}  "
              f"dup={scores['duplication_penalty']:.2f}  "
              f"empty={scores['empty_steps']}  "
              f"stage={scores['stage_score']:.3f}  {latency_ms:.0f}ms{flags}")

    del model

    # Summary
    fmt_avg = sum(r["format_score"] for r in results) / len(results)
    dup_avg = sum(r["duplication_penalty"] for r in results) / len(results)
    dup_rate = sum(1 for r in results if r["duplication_penalty"] > 0) / len(results)
    stg_avg = sum(r["stage_score"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"  Format Score (avg):      {fmt_avg:.3f}")
    print(f"  Dup Penalty (avg):       {dup_avg:.3f}")
    print(f"  Duplication Rate:        {dup_rate*100:.0f}%")
    print(f"  Stage Score (avg):       {stg_avg:.3f}")
    print(f"{'='*60}")

    # Save
    out_path = "results/benchmark/format_dup_benchmark.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"jobs": jobs, "results": results, "summary": {
            "format_score_avg": round(fmt_avg, 3),
            "dup_penalty_avg": round(dup_avg, 3),
            "duplication_rate": round(dup_rate, 3),
            "stage_score_avg": round(stg_avg, 3),
        }}, f, indent=2)
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Stage 1 format & duplication scores on selected jobs."
    )
    parser.add_argument("jobs", nargs="*", help="Job titles to benchmark")
    parser.add_argument("--file", help="File with one job title per line")
    parser.add_argument("--all", action="store_true", help="Run all 30 benchmark jobs")
    parser.add_argument("--top", type=int, default=5,
                        help="Use top N worst-duplication jobs from last benchmark (default: 5)")
    args = parser.parse_args()

    if args.all:
        with open(DATASET_PATH) as f:
            jobs = [d["job"] for d in json.load(f)]
    elif args.file:
        jobs = [l.strip() for l in Path(args.file).read_text().splitlines() if l.strip()]
    elif args.jobs:
        jobs = args.jobs
    else:
        jobs = get_top_dup_jobs(args.top)
        print(f"Using top {args.top} worst-duplication jobs from last benchmark.\n")

    run(jobs)


if __name__ == "__main__":
    main()
