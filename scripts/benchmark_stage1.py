"""
Benchmark Stage 1 (Workflow Generator) for the champion model.
Runs all 30 jobs, scores format/concreteness/relevance, reports to MLflow.
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama
from src.benchmark.prompts import STAGE1_SYSTEM, STAGE1_USER
from src.benchmark.metrics import score_stage1_workflow


MODEL_PATH = "models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"
DATASET_PATH = "src/benchmark/dataset.json"
OUTPUT_PATH = "results/benchmark/stage1_benchmark.json"


def main():
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    n_threads = max(2, (os.cpu_count() or 4) // 2)
    print(f"Loading champion model: {MODEL_PATH} ({n_threads} threads)")
    model = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=n_threads, verbose=False)

    # Warmup
    model.create_chat_completion(
        messages=[{"role": "user", "content": "Hi"}], max_tokens=5, temperature=0.0
    )

    all_scores = []
    print(f"Running Stage 1 benchmark on {len(dataset)} jobs...\n")

    for i, job_data in enumerate(dataset):
        job = job_data["job"]
        patterns = job_data["expected_patterns"]
        difficulty = job_data["difficulty"]

        start = time.perf_counter()
        resp = model.create_chat_completion(
            messages=[
                {"role": "system", "content": STAGE1_SYSTEM},
                {"role": "user", "content": STAGE1_USER.format(job=job)},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        output = resp["choices"][0]["message"]["content"]
        tokens = resp["usage"]["completion_tokens"]

        scores = score_stage1_workflow(output, patterns)
        scores["job"] = job
        scores["difficulty"] = difficulty
        scores["latency_ms"] = round(latency_ms, 1)
        scores["tokens"] = tokens
        scores["output"] = output
        all_scores.append(scores)

        print(f"[{i+1:2d}/{len(dataset)}] {job:<35} format={scores['format_score']:.2f}  "
              f"concrete={scores['concreteness']:.2f}  relevance={scores['relevance']:.2f}  "
              f"stage={scores['stage_score']:.3f}  {latency_ms:.0f}ms")

    del model

    # Aggregate
    fmt_scores = [s["format_score"] for s in all_scores]
    conc_scores = [s["concreteness"] for s in all_scores]
    rel_scores = [s["relevance"] for s in all_scores]
    stage_scores = [s["stage_score"] for s in all_scores]
    latencies = [s["latency_ms"] for s in all_scores]

    agg = {
        "model": "Qwen2.5-1.5B-Instruct Q4_K_M",
        "stage": "Stage 1 — Workflow Generator",
        "jobs_evaluated": len(all_scores),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "format_score":   {"mean": round(statistics.mean(fmt_scores), 3),   "std": round(statistics.stdev(fmt_scores), 3)},
        "concreteness":   {"mean": round(statistics.mean(conc_scores), 3),  "std": round(statistics.stdev(conc_scores), 3)},
        "relevance":      {"mean": round(statistics.mean(rel_scores), 3),   "std": round(statistics.stdev(rel_scores), 3)},
        "stage_score":    {"mean": round(statistics.mean(stage_scores), 3), "std": round(statistics.stdev(stage_scores), 3)},
        "latency_ms":     {"mean": round(statistics.mean(latencies), 1),    "std": round(statistics.stdev(latencies), 1)},
    }

    # By difficulty
    for diff in ["easy", "medium", "hard"]:
        ds = [s["stage_score"] for s in all_scores if s["difficulty"] == diff]
        if ds:
            agg[f"{diff}_stage_score"] = {"mean": round(statistics.mean(ds), 3), "count": len(ds)}

    # Print summary
    print(f"\n{'='*70}")
    print(f"  STAGE 1 BENCHMARK — {agg['model']}")
    print(f"  {agg['jobs_evaluated']} jobs evaluated")
    print(f"{'='*70}")
    print(f"  Format Score:   {agg['format_score']['mean']:.3f} ± {agg['format_score']['std']:.3f}")
    print(f"  Concreteness:   {agg['concreteness']['mean']:.3f} ± {agg['concreteness']['std']:.3f}")
    print(f"  Relevance:      {agg['relevance']['mean']:.3f} ± {agg['relevance']['std']:.3f}")
    print(f"  Stage Score:    {agg['stage_score']['mean']:.3f} ± {agg['stage_score']['std']:.3f}")
    print(f"  Avg Latency:    {agg['latency_ms']['mean']:.0f}ms ± {agg['latency_ms']['std']:.0f}ms")
    for diff in ["easy", "medium", "hard"]:
        k = f"{diff}_stage_score"
        if k in agg:
            print(f"  {diff.capitalize():<8} score:  {agg[k]['mean']:.3f} (n={agg[k]['count']})")
    print(f"{'='*70}")

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result = {"summary": agg, "detailed_scores": all_scores}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Report to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://mlflow.platform.local")
        mlflow.set_experiment("SLM-Project-Evaluator-Candidates")

        with mlflow.start_run(
            run_name="stage1-benchmark-champion",
            description=f"Stage 1 (Workflow Generator) benchmark for champion model "
                        f"Qwen2.5-1.5B-Instruct Q4_K_M. {len(all_scores)} jobs evaluated.",
        ):
            mlflow.set_tag("run_type", "stage1_benchmark")
            mlflow.set_tag("model", "Qwen2.5-1.5B-Instruct Q4_K_M")
            mlflow.set_tag("stage", "Stage 1 — Workflow Generator")

            mlflow.log_metric("s1_format_score_mean", agg["format_score"]["mean"])
            mlflow.log_metric("s1_format_score_std", agg["format_score"]["std"])
            mlflow.log_metric("s1_concreteness_mean", agg["concreteness"]["mean"])
            mlflow.log_metric("s1_concreteness_std", agg["concreteness"]["std"])
            mlflow.log_metric("s1_relevance_mean", agg["relevance"]["mean"])
            mlflow.log_metric("s1_relevance_std", agg["relevance"]["std"])
            mlflow.log_metric("s1_stage_score_mean", agg["stage_score"]["mean"])
            mlflow.log_metric("s1_stage_score_std", agg["stage_score"]["std"])
            mlflow.log_metric("s1_latency_ms_mean", agg["latency_ms"]["mean"])
            mlflow.log_metric("s1_latency_ms_std", agg["latency_ms"]["std"])

            for diff in ["easy", "medium", "hard"]:
                k = f"{diff}_stage_score"
                if k in agg:
                    mlflow.log_metric(f"s1_{diff}_score", agg[k]["mean"])

            # Log per-job scores as a step metric
            for i, s in enumerate(all_scores):
                mlflow.log_metric("s1_per_job_stage_score", s["stage_score"], step=i)
                mlflow.log_metric("s1_per_job_format", s["format_score"], step=i)
                mlflow.log_metric("s1_per_job_concreteness", s["concreteness"], step=i)
                mlflow.log_metric("s1_per_job_relevance", s["relevance"], step=i)

            mlflow.log_artifact(OUTPUT_PATH)
            print("Results reported to MLflow.")
    except Exception as e:
        print(f"MLflow reporting failed: {e}")


if __name__ == "__main__":
    main()
