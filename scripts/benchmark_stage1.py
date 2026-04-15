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
from src.pipeline.prompts import STAGE1_SYSTEM, STAGE1_USER
from src.benchmark.metrics import score_stage1_workflow
from src.config import get_model_config, get_generation_params, get_cpu_percent


DATASET_PATH = "src/benchmark/dataset.json"
OUTPUT_PATH = "results/benchmark/stage1_benchmark.json"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 benchmark")
    parser.add_argument("--remote", action="store_true", help="Use HuggingFace Inference API")
    args = parser.parse_args()

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    gen_params = get_generation_params("stage1_workflow")

    if args.remote:
        from src.pipeline.model import load_remote_model
        print("Using HuggingFace Inference API")
        model = load_remote_model()
    else:
        model_cfg = get_model_config()
        cpu_pct = get_cpu_percent()
        total_cpus = os.cpu_count() or 4
        n_threads = max(1, round(total_cpus * cpu_pct / 100))
        print(f"Loading champion model: {model_cfg['path']} ({n_threads} threads)")
        model = Llama(
            model_path=model_cfg["path"],
            n_ctx=model_cfg["n_ctx"],
            n_threads=n_threads,
            n_gpu_layers=model_cfg["n_gpu_layers"],
            verbose=False,
        )

    print(f"Gen params: {gen_params}")

    # Warmup
    model.create_chat_completion(
        messages=[{"role": "user", "content": "Hi"}], max_tokens=5, temperature=0.01
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
            **gen_params,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        output = resp["choices"][0]["message"]["content"]
        tokens = resp["usage"]["completion_tokens"]

        scores = score_stage1_workflow(output, patterns,
                                     min_tasks=job_data.get("min_tasks", 3),
                                     critical_tasks=job_data.get("critical_tasks"))
        scores["job"] = job
        scores["difficulty"] = difficulty
        scores["latency_ms"] = round(latency_ms, 1)
        scores["tokens"] = tokens
        scores["output"] = output
        all_scores.append(scores)

        # Per-job: show key metrics in pipeline order
        tc = scores.get('task_count', 0)
        ts = scores.get('total_steps', 0)
        print(f"[{i+1:2d}/{len(dataset)}] {job:<30} "
              f"tasks={tc} steps={ts}  "
              f"fmt={scores['format_score']:.2f} "
              f"cnt={scores.get('task_count_score',0):.2f} "
              f"crit={scores.get('criticality_score',0):.2f} "
              f"conc={scores['concreteness']:.2f} "
              f"rel={scores['relevance']:.2f} "
              f"coh={scores.get('coherence',0):.2f}  "
              f"→ {scores['stage_score']:.3f}  {latency_ms:.0f}ms")

    del model

    # Aggregate — grouped by pipeline order
    fmt_scores = [s["format_score"] for s in all_scores]
    tcs_scores = [s.get("task_count_score", 0) for s in all_scores]
    emp_penalties = [s["emptiness_penalty"] for s in all_scores]
    crit_scores = [s.get("criticality_score", 0) for s in all_scores]
    conc_scores = [s["concreteness"] for s in all_scores]
    rel_scores = [s["relevance"] for s in all_scores]
    coh_scores = [s.get("coherence", 0) for s in all_scores]
    dup_penalties = [s["duplication_penalty"] for s in all_scores]
    stage_scores = [s["stage_score"] for s in all_scores]
    latencies = [s["latency_ms"] for s in all_scores]
    task_counts = [s.get("task_count", 0) for s in all_scores]
    step_counts = [s.get("total_steps", 0) for s in all_scores]
    dup_count = sum(1 for s in all_scores if s["duplication_penalty"] > 0)
    empty_count = sum(1 for s in all_scores if s.get("empty_tasks", 0) + s.get("empty_steps", 0) > 0)

    n = len(all_scores)
    def _stat(vals):
        return {"mean": round(statistics.mean(vals), 3), "std": round(statistics.stdev(vals), 3) if n > 1 else 0}

    agg = {
        "model": "Qwen2.5-1.5B-Instruct",
        "stage": "Stage 1 — Workflow Generator",
        "jobs_evaluated": n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Structure metrics
        "format_score": _stat(fmt_scores),
        "task_count_score": _stat(tcs_scores),
        "avg_task_count": round(statistics.mean(task_counts), 1),
        "avg_step_count": round(statistics.mean(step_counts), 1),
        "emptiness_penalty": _stat(emp_penalties),
        "emptiness_rate": round(empty_count / n, 3),
        # Content metrics
        "criticality_score": _stat(crit_scores),
        "concreteness": _stat(conc_scores),
        "relevance": _stat(rel_scores),
        # Flow metrics
        "coherence": _stat(coh_scores),
        "duplication_penalty": _stat(dup_penalties),
        "duplication_rate": round(dup_count / n, 3),
        # Overall
        "stage_score": _stat(stage_scores),
        "latency_ms": {"mean": round(statistics.mean(latencies), 1), "std": round(statistics.stdev(latencies), 1) if n > 1 else 0},
    }

    # By difficulty
    for diff in ["easy", "medium", "hard"]:
        ds = [s["stage_score"] for s in all_scores if s["difficulty"] == diff]
        if ds:
            agg[f"{diff}_stage_score"] = {"mean": round(statistics.mean(ds), 3), "count": len(ds)}

    # Print summary — grouped by pipeline order
    print(f"\n{'='*70}")
    print(f"  STAGE 1 BENCHMARK — {agg['model']}")
    print(f"  {agg['jobs_evaluated']} jobs | avg {agg['avg_task_count']} tasks | avg {agg['avg_step_count']} steps")
    print(f"{'='*70}")
    print(f"\n  ── STRUCTURE (does the output have the right shape?) ──")
    print(f"  Format Score:      {agg['format_score']['mean']:.3f} ± {agg['format_score']['std']:.3f}   (tasks present, each with 5+ steps)")
    print(f"  Task Count Score:  {agg['task_count_score']['mean']:.3f} ± {agg['task_count_score']['std']:.3f}   (enough tasks for the job?)")
    print(f"  Emptiness Penalty: {agg['emptiness_penalty']['mean']:.3f} ± {agg['emptiness_penalty']['std']:.3f}   ({agg['emptiness_rate']*100:.0f}% of jobs have short tasks)")
    print(f"  Duplication:       {agg['duplication_penalty']['mean']:.3f} ± {agg['duplication_penalty']['std']:.3f}   ({agg['duplication_rate']*100:.0f}% of jobs affected)")
    print(f"\n  ── CONTENT (is the content good?) ──")
    print(f"  Criticality:       {agg['criticality_score']['mean']:.3f} ± {agg['criticality_score']['std']:.3f}   (tasks match expected critical tasks)")
    print(f"  Concreteness:      {agg['concreteness']['mean']:.3f} ± {agg['concreteness']['std']:.3f}   (action verbs, no vague terms)")
    print(f"  Relevance:         {agg['relevance']['mean']:.3f} ± {agg['relevance']['std']:.3f}   (matches expected job keywords)")
    print(f"\n  ── FLOW (do steps make sense together?) ──")
    print(f"  Coherence:         {agg['coherence']['mean']:.3f} ± {agg['coherence']['std']:.3f}   (steps flow logically within tasks)")
    print(f"\n  ── OVERALL ──")
    print(f"  Stage Score:       {agg['stage_score']['mean']:.3f} ± {agg['stage_score']['std']:.3f}")
    print(f"  Avg Latency:       {agg['latency_ms']['mean']:.0f}ms ± {agg['latency_ms']['std']:.0f}ms")
    for diff in ["easy", "medium", "hard"]:
        k = f"{diff}_stage_score"
        if k in agg:
            print(f"  {diff.capitalize():<8} score:   {agg[k]['mean']:.3f} (n={agg[k]['count']})")
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
            mlflow.set_tag("model", "Qwen2.5-1.5B-Instruct")
            mlflow.set_tag("stage", "Stage 1 — Workflow Generator")

            # Structure
            mlflow.log_metric("s1_format_score", agg["format_score"]["mean"])
            mlflow.log_metric("s1_task_count_score", agg["task_count_score"]["mean"])
            mlflow.log_metric("s1_avg_task_count", agg["avg_task_count"])
            mlflow.log_metric("s1_avg_step_count", agg["avg_step_count"])
            mlflow.log_metric("s1_emptiness_penalty", agg["emptiness_penalty"]["mean"])
            mlflow.log_metric("s1_emptiness_rate", agg["emptiness_rate"])
            mlflow.log_metric("s1_duplication_penalty", agg["duplication_penalty"]["mean"])
            mlflow.log_metric("s1_duplication_rate", agg["duplication_rate"])
            # Content
            mlflow.log_metric("s1_criticality_score", agg["criticality_score"]["mean"])
            mlflow.log_metric("s1_concreteness", agg["concreteness"]["mean"])
            mlflow.log_metric("s1_relevance", agg["relevance"]["mean"])
            # Flow
            mlflow.log_metric("s1_coherence", agg["coherence"]["mean"])
            # Overall
            mlflow.log_metric("s1_stage_score", agg["stage_score"]["mean"])
            mlflow.log_metric("s1_latency_ms", agg["latency_ms"]["mean"])

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
                mlflow.log_metric("s1_per_job_dup_penalty", s["duplication_penalty"], step=i)

            mlflow.log_artifact(OUTPUT_PATH)
            print("Results reported to MLflow.")
    except Exception as e:
        print(f"MLflow reporting failed: {e}")


if __name__ == "__main__":
    main()
