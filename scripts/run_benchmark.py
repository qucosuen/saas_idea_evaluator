"""
Run pipeline inference for benchmark jobs and cache the results.
Scoring is done separately by score_benchmark.py.

Usage:
  python scripts/run_benchmark.py --remote              # all jobs via HF API
  python scripts/run_benchmark.py --remote --jobs 3     # first 3 jobs
  python scripts/run_benchmark.py --remote --job "DevOps Engineer"  # single job
  python scripts/run_benchmark.py                       # local model

Cached results go to results/cache/<job_slug>.json
If a cached result already exists, it is skipped (use --force to re-run).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.runner import PipelineRunner

DATASET_PATH = "src/benchmark/dataset.json"
CACHE_DIR = "results/cache"


def slug(job: str) -> str:
    return job.lower().replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Run pipeline inference and cache results")
    parser.add_argument("--remote", action="store_true", help="Use HF Inference API")
    parser.add_argument("--jobs", type=int, help="Max number of jobs")
    parser.add_argument("--job", help="Single specific job")
    parser.add_argument("--force", action="store_true", help="Re-run even if cached")
    args = parser.parse_args()

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    if args.job:
        dataset = [s for s in dataset if s["job"].lower() == args.job.lower()]
        if not dataset:
            print(f"Job '{args.job}' not found."); return
    if args.jobs:
        dataset = dataset[:args.jobs]

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check what's already cached
    to_run = []
    for sample in dataset:
        cache_path = Path(CACHE_DIR) / f"{slug(sample['job'])}.json"
        if cache_path.exists() and not args.force:
            print(f"  [cached] {sample['job']}")
        else:
            to_run.append(sample)

    if not to_run:
        print(f"\nAll {len(dataset)} jobs already cached. Use --force to re-run.")
        return

    # Load model
    if args.remote:
        from src.pipeline.model import load_remote_model
        print("Using HuggingFace Inference API")
        model = load_remote_model()
    else:
        from src.pipeline.model import find_model, load_model
        path = find_model()
        print(f"Loading {path.name}...")
        model = load_model(path)

    runner = PipelineRunner(model, max_retries=2)
    print(f"\nRunning {len(to_run)} jobs ({len(dataset) - len(to_run)} cached)\n")

    for i, sample in enumerate(to_run):
        job = sample["job"]
        print(f"[{i+1}/{len(to_run)}] {job}", end=" ... ", flush=True)

        start = time.time()
        try:
            result = runner.run(job)
            run_dict = result.to_dict()
            elapsed = time.time() - start
            run_dict["elapsed_s"] = round(elapsed, 1)

            cache_path = Path(CACHE_DIR) / f"{slug(job)}.json"
            with open(cache_path, "w") as f:
                json.dump(run_dict, f, indent=2)

            # Also save to results/pipeline_<slug>.json for the dashboard
            with open(f"results/pipeline_{slug(job)}.json", "w") as f:
                json.dump(run_dict, f, indent=2)

            status = "✓" if result.success else "✗"
            stages_ok = sum(1 for s in run_dict["stages"] if s["valid"])
            print(f"{status} {stages_ok}/4 stages  {elapsed:.0f}s")

        except Exception as e:
            elapsed = time.time() - start
            print(f"✗ ERROR ({elapsed:.0f}s): {e}")
            # Save error result to cache so score_benchmark knows it failed
            cache_path = Path(CACHE_DIR) / f"{slug(job)}.json"
            with open(cache_path, "w") as f:
                json.dump({"job": job, "success": False, "error": str(e),
                           "stages": [], "total_latency_ms": 0, "elapsed_s": round(elapsed, 1)}, f, indent=2)

    print(f"\nDone. Results cached in {CACHE_DIR}/")
    print(f"Run scoring: python scripts/score_benchmark.py")


if __name__ == "__main__":
    main()
