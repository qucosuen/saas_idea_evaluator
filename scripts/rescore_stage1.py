"""
Re-score Stage 1 benchmark from saved outputs (no model inference needed).

Usage:
  # Re-score all jobs
  python scripts/rescore_stage1.py

  # Re-score a specific job with detailed explanation
  python scripts/rescore_stage1.py --job "Hospital Administrator"

  # Re-score multiple specific jobs
  python scripts/rescore_stage1.py --job "Sales Representative" --job "Janitor"

  # Re-score from a specific file
  python scripts/rescore_stage1.py --input results/benchmark/stage1_benchmark.json

  # Save to a different output file
  python scripts/rescore_stage1.py --output results/benchmark/stage1_rescored.json

  # Also report to MLflow
  python scripts/rescore_stage1.py --mlflow
"""

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.metrics import score_stage1_workflow, ACTION_VERBS

DEFAULT_INPUT = "results/benchmark/stage1_benchmark.json"
DATASET_PATH = "src/benchmark/dataset.json"


def load_expected_patterns() -> dict[str, dict]:
    with open(DATASET_PATH) as f:
        return {d["job"]: d for d in json.load(f)}


def explain_score(scores: dict, output: str, expected_patterns: list[str]):
    """Print a detailed explanation of why each sub-score is not 1.0."""
    print(f"\n  {'─'*60}")
    print(f"  MODEL OUTPUT:")
    print(f"  {'─'*60}")
    for line in output.strip().split("\n"):
        print(f"    {line.rstrip()}")
    print(f"  {'─'*60}")

    print(f"\n  SCORE BREAKDOWN:")
    print(f"    raw = 0.15×format + 0.15×task_count + 0.15×criticality + 0.15×concreteness + 0.15×relevance + 0.25×coherence - penalties\n")

    # Format
    fmt = scores["format_score"]
    print(f"    format_score = {fmt:.3f}", end="")
    if fmt < 1.0:
        print(f"  ← NOT PERFECT")
        print(f"      Expected: at least 1 task, each with 5+ steps")
        print(f"      Found: {scores.get('task_count', 0)} task(s), {scores.get('tasks_with_enough_steps', 0)} with 5+ steps")
        print(f"      Total steps: {scores.get('total_steps', 0)}")
    else:
        print(f"  ✓")

    # Task count
    tcs = scores.get("task_count_score", 0)
    print(f"    task_count_score = {tcs:.3f}", end="")
    if tcs < 1.0:
        print(f"  ← NOT PERFECT")
        print(f"      Tasks generated: {scores.get('task_count', 0)} (1 is poor, more is better)")
    else:
        print(f"  ✓")

    # Criticality
    crit = scores.get("criticality_score", 0)
    print(f"    criticality_score = {crit:.3f}", end="")
    if crit < 1.0:
        print(f"  ← NOT PERFECT")
        print(f"      Checks if generated tasks match expected critical tasks for this job")
    else:
        print(f"  ✓")

    # Concreteness
    conc = scores["concreteness"]
    print(f"    concreteness = {conc:.3f}", end="")
    if conc < 1.0:
        print(f"  ← NOT PERFECT")
        text_lower = output.lower()
        found_verbs = sorted([v for v in ACTION_VERBS if v in text_lower])
        print(f"      Action verbs found: {scores['verb_hits']} (need ≥5 for full score)")
        if scores["verb_hits"] < 5:
            print(f"      Verbs matched: {found_verbs[:10]}")
        if scores["vague_hits"] > 0:
            print(f"      Vague terms found: {scores['vague_hits']}")
    else:
        print(f"  ✓")

    # Relevance
    rel = scores["relevance"]
    print(f"    relevance = {rel:.3f}", end="")
    if rel < 1.0:
        print(f"  ← NOT PERFECT")
        text_lower = output.lower()
        matched = [p for p in expected_patterns if p.lower() in text_lower]
        missed = [p for p in expected_patterns if p.lower() not in text_lower]
        print(f"      Matched: {matched} ({scores['pattern_hits']}/{len(expected_patterns)})")
        print(f"      Missed:  {missed}")
    else:
        print(f"  ✓")

    # Coherence
    coh = scores.get("coherence", 0)
    print(f"    coherence = {coh:.3f}", end="")
    if coh < 1.0:
        print(f"  ← NOT PERFECT")
        for d in scores.get("coherence_details", []):
            print(f"      {d.get('task','')}: {d.get('coherence',0):.3f}")
    else:
        print(f"  ✓")

    # Penalties
    dup = scores["duplication_penalty"]
    emp = scores["emptiness_penalty"]
    if dup > 0:
        print(f"    duplication_penalty = -{dup:.3f}  ← PENALTY ({scores['total_duplicates']} duplicates)")
    else:
        print(f"    duplication_penalty = 0.000  ✓")
    if emp > 0:
        print(f"    emptiness_penalty = -{emp:.3f}  ← PENALTY")
    else:
        print(f"    emptiness_penalty = 0.000  ✓")

    # Final
    raw = 0.15*fmt + 0.15*tcs + 0.15*crit + 0.15*conc + 0.15*rel + 0.25*coh
    final = scores["stage_score"]
    print(f"\n    raw_score   = {raw:.3f}")
    if dup > 0 or emp > 0:
        print(f"    penalties   = -{dup + emp:.3f}")
    print(f"    stage_score = {final:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Re-score Stage 1 from saved outputs.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to previous benchmark JSON")
    parser.add_argument("--output", help="Output path (default: overwrites input when no --job)")
    parser.add_argument("--job", action="append", help="Specific job(s) to re-score with explanation (repeatable)")
    parser.add_argument("--mlflow", action="store_true", help="Report to MLflow")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    prev_scores = data["detailed_scores"]
    patterns = load_expected_patterns()

    # Filter to specific jobs if requested
    if args.job:
        job_set = set(j.lower() for j in args.job)
        filtered = [e for e in prev_scores if e["job"].lower() in job_set]
        not_found = job_set - set(e["job"].lower() for e in filtered)
        if not_found:
            print(f"Warning: jobs not found in benchmark: {not_found}")
            available = [e["job"] for e in prev_scores]
            print(f"Available: {available}\n")
        if not filtered:
            return
        prev_scores = filtered

    print(f"Re-scoring {len(prev_scores)} job(s) from {args.input}\n")

    rescored = []
    for i, entry in enumerate(prev_scores):
        job = entry["job"]
        output = entry["output"]
        job_data = patterns.get(job, {})
        job_patterns = job_data.get("expected_patterns", []) if isinstance(job_data, dict) else job_data

        new_scores = score_stage1_workflow(output, job_patterns,
                                           min_tasks=job_data.get("min_tasks", 3) if isinstance(job_data, dict) else 3,
                                           critical_tasks=job_data.get("critical_tasks") if isinstance(job_data, dict) else None)
        new_scores["job"] = job
        new_scores["difficulty"] = entry.get("difficulty", "")
        new_scores["latency_ms"] = entry.get("latency_ms", 0)
        new_scores["tokens"] = entry.get("tokens", 0)
        new_scores["output"] = output
        rescored.append(new_scores)

        old_stage = entry.get("stage_score", 0)
        new_stage = new_scores["stage_score"]
        delta = new_stage - old_stage
        marker = f" ({'+' if delta >= 0 else ''}{delta:.3f})" if abs(delta) > 0.001 else ""

        print(f"[{i+1:2d}/{len(prev_scores)}] {job:<35} "
              f"fmt={new_scores['format_score']:.2f}  "
              f"cnt={new_scores['task_count_score']:.2f}  "
              f"crit={new_scores['criticality_score']:.2f}  "
              f"conc={new_scores['concreteness']:.2f}  "
              f"rel={new_scores['relevance']:.2f}  "
              f"coh={new_scores['coherence']:.2f}  "
              f"stage={new_stage:.3f}{marker}")

        if args.job and new_stage < 1.0:
            explain_score(new_scores, output, job_patterns)

    # If --job mode, skip aggregate/save/mlflow
    if args.job:
        return

    # Aggregate
    fmt = [s["format_score"] for s in rescored]
    tcs = [s["task_count_score"] for s in rescored]
    crit = [s["criticality_score"] for s in rescored]
    conc = [s["concreteness"] for s in rescored]
    rel = [s["relevance"] for s in rescored]
    coh = [s["coherence"] for s in rescored]
    dup = [s["duplication_penalty"] for s in rescored]
    emp = [s["emptiness_penalty"] for s in rescored]
    stg = [s["stage_score"] for s in rescored]
    lat = [s["latency_ms"] for s in rescored]

    n = len(rescored)
    dup_count = sum(1 for s in rescored if s["duplication_penalty"] > 0)
    emp_count = sum(1 for s in rescored if s["empty_steps"] > 0)

    agg = {
        "model": data.get("summary", {}).get("model", "unknown"),
        "stage": "Stage 1 — Workflow Generator (rescored)",
        "jobs_evaluated": n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rescored_from": args.input,
        "format_score":        {"mean": round(statistics.mean(fmt), 3),  "std": round(statistics.stdev(fmt), 3) if n > 1 else 0},
        "task_count_score":    {"mean": round(statistics.mean(tcs), 3),  "std": round(statistics.stdev(tcs), 3) if n > 1 else 0},
        "criticality_score":   {"mean": round(statistics.mean(crit), 3), "std": round(statistics.stdev(crit), 3) if n > 1 else 0},
        "concreteness":        {"mean": round(statistics.mean(conc), 3), "std": round(statistics.stdev(conc), 3) if n > 1 else 0},
        "relevance":           {"mean": round(statistics.mean(rel), 3),  "std": round(statistics.stdev(rel), 3) if n > 1 else 0},
        "coherence":           {"mean": round(statistics.mean(coh), 3),  "std": round(statistics.stdev(coh), 3) if n > 1 else 0},
        "duplication_penalty": {"mean": round(statistics.mean(dup), 3),  "std": round(statistics.stdev(dup), 3) if n > 1 else 0},
        "duplication_rate":    round(dup_count / n, 3),
        "emptiness_penalty":   {"mean": round(statistics.mean(emp), 3),  "std": round(statistics.stdev(emp), 3) if n > 1 else 0},
        "emptiness_rate":      round(emp_count / n, 3),
        "stage_score":         {"mean": round(statistics.mean(stg), 3),  "std": round(statistics.stdev(stg), 3) if n > 1 else 0},
        "latency_ms":          {"mean": round(statistics.mean(lat), 1),  "std": round(statistics.stdev(lat), 1) if n > 1 else 0},
    }

    for diff in ["easy", "medium", "hard"]:
        ds = [s["stage_score"] for s in rescored if s.get("difficulty") == diff]
        if ds:
            agg[f"{diff}_stage_score"] = {"mean": round(statistics.mean(ds), 3), "count": len(ds)}

    old_agg = data.get("summary", {})
    old_mean = old_agg.get("stage_score", {}).get("mean")
    new_mean = agg["stage_score"]["mean"]

    print(f"\n{'='*70}")
    print(f"  RESCORED STAGE 1 BENCHMARK")
    print(f"  {n} jobs")
    print(f"{'='*70}")
    print(f"\n  ── STRUCTURE ──")
    print(f"  Format Score:      {agg['format_score']['mean']:.3f} ± {agg['format_score']['std']:.3f}")
    print(f"  Task Count:        {agg['task_count_score']['mean']:.3f} ± {agg['task_count_score']['std']:.3f}")
    print(f"  Emptiness Penalty: {agg['emptiness_penalty']['mean']:.3f} ± {agg['emptiness_penalty']['std']:.3f}  ({agg['emptiness_rate']*100:.0f}%)")
    print(f"  Duplication:       {agg['duplication_penalty']['mean']:.3f} ± {agg['duplication_penalty']['std']:.3f}  ({agg['duplication_rate']*100:.0f}%)")
    print(f"\n  ── CONTENT ──")
    print(f"  Criticality:       {agg['criticality_score']['mean']:.3f} ± {agg['criticality_score']['std']:.3f}")
    print(f"  Concreteness:      {agg['concreteness']['mean']:.3f} ± {agg['concreteness']['std']:.3f}")
    print(f"  Relevance:         {agg['relevance']['mean']:.3f} ± {agg['relevance']['std']:.3f}")
    print(f"\n  ── FLOW ──")
    print(f"  Coherence:         {agg['coherence']['mean']:.3f} ± {agg['coherence']['std']:.3f}")
    print(f"\n  ── OVERALL ──")
    print(f"  Stage Score:       {agg['stage_score']['mean']:.3f} ± {agg['stage_score']['std']:.3f}")
    if old_mean is not None:
        delta = new_mean - old_mean
        print(f"  Previous Score: {old_mean:.3f}  →  Delta: {'+' if delta >= 0 else ''}{delta:.3f}")
    for diff in ["easy", "medium", "hard"]:
        k = f"{diff}_stage_score"
        if k in agg:
            print(f"  {diff.capitalize():<8} score:  {agg[k]['mean']:.3f} (n={agg[k]['count']})")
    print(f"{'='*70}")

    out_path = args.output or args.input
    result = {"summary": agg, "detailed_scores": rescored}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    if args.mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri("http://mlflow.platform.local")
            mlflow.set_experiment("SLM-Project-Evaluator-Candidates")
            with mlflow.start_run(
                run_name="stage1-rescored",
                description=f"Re-scored Stage 1 benchmark ({n} jobs) with updated criteria.",
            ):
                mlflow.set_tag("run_type", "stage1_rescore")
                mlflow.set_tag("model", agg["model"])
                for key in ["format_score", "task_count_score", "criticality_score", "concreteness", "relevance", "coherence", "duplication_penalty", "emptiness_penalty", "stage_score", "latency_ms"]:
                    mlflow.log_metric(f"s1_{key}_mean", agg[key]["mean"])
                    mlflow.log_metric(f"s1_{key}_std", agg[key]["std"])
                mlflow.log_metric("s1_duplication_rate", agg["duplication_rate"])
                mlflow.log_metric("s1_emptiness_rate", agg["emptiness_rate"])
                for diff in ["easy", "medium", "hard"]:
                    k = f"{diff}_stage_score"
                    if k in agg:
                        mlflow.log_metric(f"s1_{diff}_score", agg[k]["mean"])
                print("Reported to MLflow.")
        except Exception as e:
            print(f"MLflow failed: {e}")


if __name__ == "__main__":
    main()
