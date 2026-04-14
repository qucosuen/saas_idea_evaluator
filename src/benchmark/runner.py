"""
Benchmark runner: loops through the dataset, executes the pipeline,
scores each run, and produces a summary report.

Usage:
  .venv/bin/python -m src.benchmark.runner
  .venv/bin/python -m src.benchmark.runner --model models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
  .venv/bin/python -m src.benchmark.runner --jobs 5  # run first 5 jobs only
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_dataset() -> list[dict]:
    ds_path = Path(__file__).parent / "dataset.json"
    with open(ds_path) as f:
        return json.load(f)


def find_default_model() -> Path:
    """Find the best available GGUF model."""
    models_dir = Path("models/gguf")
    for name in [
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ]:
        p = models_dir / name
        if p.exists():
            return p
    raise FileNotFoundError("No GGUF model found in models/gguf/")


def run_benchmark(model_path: Path, dataset: list[dict], max_jobs: int | None = None):
    """Run the full benchmark suite."""
    from llama_cpp import Llama
    from src.benchmark.pipeline import PipelineRunner
    from src.benchmark.scorer import score_pipeline

    n_threads = max(2, (os.cpu_count() or 4) // 2)
    print(f"Loading model: {model_path.name} ({n_threads} threads)")
    model = Llama(model_path=str(model_path), n_ctx=2048, n_threads=n_threads, verbose=False)
    runner = PipelineRunner(model, max_tokens=150)

    jobs = dataset[:max_jobs] if max_jobs else dataset
    print(f"Running benchmark: {len(jobs)} jobs\n")

    all_scores = []
    all_raw = []

    for i, job_data in enumerate(jobs):
        job = job_data["job"]
        patterns = job_data["expected_patterns"]
        difficulty = job_data["difficulty"]

        print(f"[{i+1}/{len(jobs)}] {job} ({difficulty})", end=" ... ", flush=True)

        result = runner.run(job)
        scores = score_pipeline(result, patterns)
        scores["difficulty"] = difficulty

        all_scores.append(scores)
        all_raw.append({
            "job": job,
            "difficulty": difficulty,
            "stages": [
                {"name": s.name, "output": s.output, "latency_ms": round(s.latency_ms, 1), "tokens": s.tokens}
                for s in result.stages
            ],
        })

        fs = scores["final_score"]
        lat = scores["latency"]["total_ms"]
        print(f"score={fs:.3f}  latency={lat:.0f}ms")

    del model
    return all_scores, all_raw


def generate_report(scores: list[dict], model_name: str) -> dict:
    """Generate aggregate report from all benchmark runs."""
    n = len(scores)

    # Per-stage averages
    stage_names = ["stage_1_workflow", "stage_2_problems", "stage_3_solutions", "stage_4_evaluation"]
    stage_avgs = {}
    for sn in stage_names:
        vals = [s[sn]["stage_score"] for s in scores]
        stage_avgs[sn] = {
            "mean": round(statistics.mean(vals), 3),
            "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0,
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }

    # Final scores
    finals = [s["final_score"] for s in scores]

    # Latency
    latencies = [s["latency"]["total_ms"] for s in scores]
    per_stage_lat = {}
    for key in ["stage_1_ms", "stage_2_ms", "stage_3_ms", "stage_4_ms"]:
        vals = [s["latency"][key] for s in scores]
        per_stage_lat[key] = round(statistics.mean(vals), 1)

    # By difficulty
    by_diff = {}
    for diff in ["easy", "medium", "hard"]:
        diff_scores = [s["final_score"] for s in scores if s.get("difficulty") == diff]
        if diff_scores:
            by_diff[diff] = {"mean": round(statistics.mean(diff_scores), 3), "count": len(diff_scores)}

    # Usefulness
    useful_count = sum(1 for s in scores if s["end_to_end"]["is_useful"])

    # Format compliance (Stage 1 format_score == 1.0)
    format_ok = sum(1 for s in scores if s["stage_1_workflow"]["format_score"] >= 0.9)

    return {
        "model": model_name,
        "jobs_evaluated": n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "final_score": {
            "mean": round(statistics.mean(finals), 3),
            "std": round(statistics.stdev(finals), 3) if n > 1 else 0,
            "min": round(min(finals), 3),
            "max": round(max(finals), 3),
        },
        "stage_scores": stage_avgs,
        "latency": {
            "avg_total_ms": round(statistics.mean(latencies), 1),
            "min_total_ms": round(min(latencies), 1),
            "max_total_ms": round(max(latencies), 1),
            "per_stage_avg_ms": per_stage_lat,
        },
        "by_difficulty": by_diff,
        "format_compliance_rate": round(format_ok / n, 3) if n > 0 else 0,
        "usefulness_rate": round(useful_count / n, 3) if n > 0 else 0,
        "success_criteria": {
            "format_gte_90pct": format_ok / n >= 0.9 if n > 0 else False,
            "pipeline_stable": n > 0 and all(len(s["latency"]) > 0 for s in scores),
            "latency_under_500ms": statistics.mean(latencies) < 500 if latencies else False,
        },
    }


def print_report(report: dict):
    """Pretty-print the benchmark report."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK REPORT: {report['model']}")
    print(f"  {report['jobs_evaluated']} jobs evaluated")
    print(f"{'='*70}")

    fs = report["final_score"]
    print(f"\n  Final Score: {fs['mean']:.3f} ± {fs['std']:.3f} (range: {fs['min']:.3f} - {fs['max']:.3f})")

    print(f"\n  Stage Scores:")
    for sn, sv in report["stage_scores"].items():
        label = sn.replace("_", " ").title()
        print(f"    {label:<30} {sv['mean']:.3f} ± {sv['std']:.3f}")

    lat = report["latency"]
    print(f"\n  Latency:")
    print(f"    Average total: {lat['avg_total_ms']:.0f}ms")
    for k, v in lat["per_stage_avg_ms"].items():
        print(f"    {k}: {v:.0f}ms")

    print(f"\n  By Difficulty:")
    for diff, dv in report["by_difficulty"].items():
        print(f"    {diff:<8} {dv['mean']:.3f} (n={dv['count']})")

    print(f"\n  Format compliance: {report['format_compliance_rate']*100:.0f}%")
    print(f"  Usefulness rate:   {report['usefulness_rate']*100:.0f}%")

    print(f"\n  Success Criteria:")
    for k, v in report["success_criteria"].items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Run the 4-stage LLM pipeline benchmark")
    parser.add_argument("--model", help="Path to GGUF model file")
    parser.add_argument("--jobs", type=int, help="Max number of jobs to evaluate")
    parser.add_argument("--sample", type=int, help="Sample N jobs per difficulty level")
    parser.add_argument("--output-dir", default="results/benchmark", help="Output directory")
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_default_model()
    model_name = model_path.stem
    dataset = load_dataset()

    if args.sample:
        import random
        random.seed(42)
        sampled = []
        for diff in ["easy", "medium", "hard"]:
            pool = [j for j in dataset if j["difficulty"] == diff]
            sampled.extend(random.sample(pool, min(args.sample, len(pool))))
        dataset = sampled

    scores, raw_outputs = run_benchmark(model_path, dataset, max_jobs=args.jobs)

    report = generate_report(scores, model_name)
    print_report(report)

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{model_name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out_dir / f"{model_name}_scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    with open(out_dir / f"{model_name}_raw_outputs.json", "w") as f:
        json.dump(raw_outputs, f, indent=2)

    print(f"\nResults saved to {out_dir}/")

    # Log to MLflow
    if args.mlflow:
        log_to_mlflow(report, scores)


def log_to_mlflow(report: dict, scores: list[dict]):
    """Log benchmark results to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://mlflow.platform.local")
        mlflow.set_experiment("SLM-Project-Evaluator-Candidates")

        with mlflow.start_run(
            run_name=f"benchmark-{report['model']}",
            description=f"4-stage pipeline benchmark for {report['model']}. "
                        f"{report['jobs_evaluated']} jobs evaluated.",
        ):
            mlflow.set_tag("run_type", "pipeline_benchmark")
            mlflow.set_tag("model", report["model"])

            # Final score
            mlflow.log_metric("benchmark_final_score", report["final_score"]["mean"])
            mlflow.log_metric("benchmark_final_score_std", report["final_score"]["std"])

            # Stage scores
            for sn, sv in report["stage_scores"].items():
                mlflow.log_metric(f"benchmark_{sn}_mean", sv["mean"])

            # Latency
            mlflow.log_metric("benchmark_avg_latency_ms", report["latency"]["avg_total_ms"])
            for k, v in report["latency"]["per_stage_avg_ms"].items():
                mlflow.log_metric(f"benchmark_{k}", v)

            # Rates
            mlflow.log_metric("benchmark_format_compliance", report["format_compliance_rate"])
            mlflow.log_metric("benchmark_usefulness_rate", report["usefulness_rate"])

            # By difficulty
            for diff, dv in report["by_difficulty"].items():
                mlflow.log_metric(f"benchmark_{diff}_score", dv["mean"])

            # Success criteria as tags
            for k, v in report["success_criteria"].items():
                mlflow.set_tag(f"criteria_{k}", str(v))

            print("Results logged to MLflow.")
    except Exception as e:
        print(f"MLflow logging failed: {e}")


if __name__ == "__main__":
    main()
