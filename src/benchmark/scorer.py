"""
Scoring system: computes per-stage and weighted final scores for a pipeline run.
"""

from src.benchmark.metrics import (
    score_stage1_workflow,
    score_stage2_problems,
    score_stage3_solutions,
    score_stage4_evaluation,
    score_usefulness,
)
from src.benchmark.pipeline import PipelineResult


def score_pipeline(result: PipelineResult, expected_patterns: list[str],
                   min_tasks: int = 3, critical_tasks: list[str] | None = None,
                   expected_solutions: list[str] | None = None) -> dict:
    """Score all stages of a pipeline run and compute weighted final score."""

    # Stage 1
    s1 = score_stage1_workflow(result.workflow, expected_patterns,
                               min_tasks=min_tasks, critical_tasks=critical_tasks)

    # Stage 2 — extract task/step text for alignment check
    workflow_lines = [l.strip() for l in result.workflow.split("\n") if l.strip()]
    s2 = score_stage2_problems(result.problems, workflow_lines,
                               expected_solutions=expected_solutions)

    # Stage 3
    s3 = score_stage3_solutions(result.solutions, s2.get("problems_found", 5))

    # Stage 4
    s4 = score_stage4_evaluation(result.evaluation)

    # End-to-end usefulness
    e2e = score_usefulness(result.evaluation)

    # Weighted final score (from Phase 5 of roadmap)
    final = (
        0.25 * s1["stage_score"]
        + 0.20 * s2["stage_score"]
        + 0.20 * s3["stage_score"]
        + 0.20 * s4["stage_score"]
        + 0.15 * e2e["usefulness_score"]
    )

    # Latency
    latency = {
        "stage_1_ms": round(result.stages[0].latency_ms, 1) if len(result.stages) > 0 else 0,
        "stage_2_ms": round(result.stages[1].latency_ms, 1) if len(result.stages) > 1 else 0,
        "stage_3_ms": round(result.stages[2].latency_ms, 1) if len(result.stages) > 2 else 0,
        "stage_4_ms": round(result.stages[3].latency_ms, 1) if len(result.stages) > 3 else 0,
        "total_ms": round(result.total_latency_ms, 1),
        "total_tokens": sum(s.tokens for s in result.stages),
    }

    return {
        "job": result.job,
        "stage_1_workflow": s1,
        "stage_2_step_analysis": s2,
        "stage_3_solutions": s3,
        "stage_4_evaluation": s4,
        "end_to_end": e2e,
        "latency": latency,
        "final_score": round(final, 3),
    }
