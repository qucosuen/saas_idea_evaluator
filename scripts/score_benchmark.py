"""
Score cached benchmark results. No model inference needed.

Reads pipeline outputs from results/cache/<job>.json and scores them
against the benchmark dataset ground truth.

Usage:
  python scripts/score_benchmark.py                # score all cached results
  python scripts/score_benchmark.py --job "DevOps Engineer"  # score one job
  python scripts/score_benchmark.py --mlflow       # also report to MLflow
  python scripts/score_benchmark.py --explain      # show per-metric breakdown
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.scorer import score_pipeline

DATASET_PATH = "src/benchmark/dataset.json"
CACHE_DIR = "results/cache"
OUTPUT_PATH = "results/benchmark_results.json"


def slug(job: str) -> str:
    return job.lower().replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Score cached benchmark results")
    parser.add_argument("--job", help="Score a single job")
    parser.add_argument("--mlflow", action="store_true", help="Report to MLflow")
    parser.add_argument("--explain", action="store_true", help="Show per-metric breakdown")
    args = parser.parse_args()

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    if args.job:
        dataset = [s for s in dataset if s["job"].lower() == args.job.lower()]

    all_results = []
    missing = []

    for sample in dataset:
        job = sample["job"]
        cache_path = Path(CACHE_DIR) / f"{slug(job)}.json"

        if not cache_path.exists():
            missing.append(job)
            continue

        with open(cache_path) as f:
            run_dict = json.load(f)

        if run_dict.get("error"):
            print(f"  [error] {job}: {run_dict['error']}")
            all_results.append({
                "job": job, "difficulty": sample.get("difficulty", ""),
                "success": False, "final_score": 0, "elapsed_s": run_dict.get("elapsed_s", 0),
                "task_scores": {"score": 0, "max": 5}, "step_scores": {"score": 0, "max": 5},
                "problem_scores": {"score": 0, "max": 5}, "solution_scores": {"score": 0, "max": 5},
                "eval_scores": {"score": 0, "max": 5},
                "system_scores": {"consistency": 0, "redundancy_penalty": 0, "constraint_adherence": 0, "practicality": 0},
            })
            continue

        scores = score_pipeline(run_dict, sample)
        scores["elapsed_s"] = run_dict.get("elapsed_s", 0)
        all_results.append(scores)

        # Per-job output
        s = scores
        print(f"  {job:<35} "
              f"task={s['task_scores']['score']:.3f} "
              f"step={s['step_scores']['score']:.3f} "
              f"prob={s['problem_scores']['score']:.3f} "
              f"sol={s['solution_scores']['score']:.3f} "
              f"eval={s['eval_scores']['score']:.3f} "
              f"→ {s['final_score']:.3f}")

        if args.explain:
            for stage_name in ["task_scores", "step_scores", "problem_scores", "solution_scores", "eval_scores"]:
                stage = s[stage_name]
                details = "  ".join(f"{k}={v}" for k, v in stage.items() if k not in ("score", "max"))
                print(f"    {stage_name}: {details}")
            print(f"    system: {s['system_scores']}")
            print()

    if missing:
        print(f"\n  Missing cache for: {missing}")
        print(f"  Run: python scripts/run_benchmark.py --remote --job \"<job>\"")

    n = len(all_results)
    if n == 0:
        print("No results to score."); return

    # Aggregate
    def _avg(key):
        vals = [r[key] for r in all_results if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 3) if vals else 0

    def _stage_avg(stage_key, sub_key):
        vals = [r[stage_key][sub_key] for r in all_results if stage_key in r and sub_key in r[stage_key]]
        return round(statistics.mean(vals), 3) if vals else 0

    summary = {
        "jobs_evaluated": n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "final_score": {"mean": _avg("final_score"),
                        "std": round(statistics.stdev([r["final_score"] for r in all_results]), 3) if n > 1 else 0},
        "task_score": {"mean": _stage_avg("task_scores", "score"),
                       "coverage": _stage_avg("task_scores", "coverage"),
                       "relevance": _stage_avg("task_scores", "relevance"),
                       "specificity": _stage_avg("task_scores", "specificity"),
                       "non_redundancy": _stage_avg("task_scores", "non_redundancy")},
        "step_score": {"mean": _stage_avg("step_scores", "score"),
                       "atomicity": _stage_avg("step_scores", "atomicity"),
                       "logical_ordering": _stage_avg("step_scores", "logical_ordering"),
                       "actionability": _stage_avg("step_scores", "actionability"),
                       "specificity": _stage_avg("step_scores", "specificity")},
        "problem_score": {"mean": _stage_avg("problem_scores", "score"),
                          "specificity": _stage_avg("problem_scores", "specificity"),
                          "realism": _stage_avg("problem_scores", "realism"),
                          "causality": _stage_avg("problem_scores", "causality")},
        "solution_score": {"mean": _stage_avg("solution_scores", "score"),
                           "correctness": _stage_avg("solution_scores", "correctness"),
                           "specificity": _stage_avg("solution_scores", "specificity"),
                           "diversity": _stage_avg("solution_scores", "diversity"),
                           "feasibility": _stage_avg("solution_scores", "feasibility"),
                           "novelty": _stage_avg("solution_scores", "novelty")},
        "eval_score": {"mean": _stage_avg("eval_scores", "score"),
                       "variance": _stage_avg("eval_scores", "variance"),
                       "justification": _stage_avg("eval_scores", "justification"),
                       "specificity": _stage_avg("eval_scores", "specificity"),
                       "usefulness": _stage_avg("eval_scores", "usefulness")},
        "system": {
            "consistency": _stage_avg("system_scores", "consistency"),
            "redundancy": _stage_avg("system_scores", "redundancy_penalty"),
            "constraint_adherence": _stage_avg("system_scores", "constraint_adherence"),
            "practicality": _stage_avg("system_scores", "practicality"),
        },
        "success_rate": round(sum(1 for r in all_results if r.get("success")) / n, 3),
    }

    for diff in ["easy", "medium", "hard"]:
        ds = [r["final_score"] for r in all_results if r.get("difficulty") == diff]
        if ds:
            summary[f"{diff}_score"] = round(statistics.mean(ds), 3)

    # Print
    ts = summary["task_score"]
    ss = summary["step_score"]
    ps = summary["problem_score"]
    so = summary["solution_score"]
    es = summary["eval_score"]
    sy = summary["system"]

    print(f"\n{'='*70}")
    print(f"  BENCHMARK SCORES — {n} jobs (all scores 0.0-1.0)")
    print(f"{'='*70}")
    print(f"\n  ── TASK: {ts['mean']:.3f} ──")
    print(f"    coverage={ts['coverage']:.3f}  relevance={ts['relevance']:.3f}  specificity={ts['specificity']:.3f}  non_redundancy={ts['non_redundancy']:.3f}")
    print(f"\n  ── STEP: {ss['mean']:.3f} ──")
    print(f"    atomicity={ss['atomicity']:.3f}  ordering={ss['logical_ordering']:.3f}  actionability={ss['actionability']:.3f}  specificity={ss['specificity']:.3f}")
    print(f"\n  ── PROBLEM: {ps['mean']:.3f} ──")
    print(f"    specificity={ps['specificity']:.3f}  realism={ps['realism']:.3f}  causality={ps['causality']:.3f}")
    print(f"\n  ── SOLUTION: {so['mean']:.3f} ──")
    print(f"    correctness={so['correctness']:.3f}  specificity={so['specificity']:.3f}  diversity={so['diversity']:.3f}  feasibility={so['feasibility']:.3f}  novelty={so['novelty']:.3f}")
    print(f"\n  ── EVALUATION: {es['mean']:.3f} ──")
    print(f"    variance={es['variance']:.3f}  justification={es['justification']:.3f}  specificity={es['specificity']:.3f}  usefulness={es['usefulness']:.3f}")
    print(f"\n  ── SYSTEM ──")
    print(f"    consistency={sy['consistency']:.3f}  redundancy={sy['redundancy']:.3f}  constraints={sy['constraint_adherence']:.3f}  practicality={sy['practicality']:.3f}")
    print(f"\n  ── OVERALL ──")
    print(f"  Final: {summary['final_score']['mean']:.3f} ± {summary['final_score']['std']:.3f}  |  Success: {summary['success_rate']*100:.0f}%")
    for diff in ["easy", "medium", "hard"]:
        k = f"{diff}_score"
        if k in summary:
            print(f"  {diff.capitalize():<8}: {summary[k]:.3f}")
    print(f"{'='*70}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    # MLflow
    if args.mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri("http://mlflow.platform.local")
            mlflow.set_experiment("SLM-Project-Evaluator-Candidates")
            with mlflow.start_run(
                run_name=f"benchmark-{n}jobs",
                description=f"Benchmark: {n} jobs, final={summary['final_score']['mean']:.3f}",
            ):
                mlflow.set_tag("run_type", "benchmark")
                mlflow.log_metric("final_score", summary["final_score"]["mean"])
                # Stage scores
                for stage in ["task_score", "step_score", "problem_score", "solution_score", "eval_score"]:
                    for k, v in summary[stage].items():
                        mlflow.log_metric(f"{stage}_{k}", v)
                # System
                for k, v in summary["system"].items():
                    mlflow.log_metric(f"system_{k}", v)
                mlflow.log_metric("success_rate", summary["success_rate"])
                for diff in ["easy", "medium", "hard"]:
                    k = f"{diff}_score"
                    if k in summary:
                        mlflow.log_metric(k, summary[k])
                for i, r in enumerate(all_results):
                    mlflow.log_metric("per_job_final", r["final_score"], step=i)
                print("Reported to MLflow.")
        except Exception as e:
            print(f"MLflow failed: {e}")


if __name__ == "__main__":
    main()
