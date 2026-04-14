"""
Stage-level metrics for the 4-stage LLM pipeline benchmark.

Each stage has specific format, content, and quality metrics.
All scoring functions return a dict with individual scores and a combined stage_score (0-1).
"""

import re
import statistics

# ---------------------------------------------------------------------------
# Stage 1: Workflow Generator
# ---------------------------------------------------------------------------

ACTION_VERBS = {
    "collect", "gather", "review", "verify", "enter", "process", "submit",
    "check", "validate", "update", "create", "generate", "send", "prepare",
    "analyze", "compile", "organize", "schedule", "assign", "approve",
    "monitor", "track", "report", "file", "archive", "distribute", "sort",
    "inspect", "calculate", "reconcile", "coordinate", "manage", "evaluate",
    "document", "record", "input", "extract", "transfer", "notify", "escalate",
}

VAGUE_TERMS = {
    "do stuff", "handle things", "work on it", "take care of", "deal with",
    "do the needful", "various tasks", "miscellaneous", "etc", "and so on",
}


def score_stage1_workflow(output: str, expected_patterns: list[str]) -> dict:
    """Score the Workflow Generator output."""
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]

    # 1. Format check: exactly 5 steps with "Step X:" prefix
    step_pattern = re.compile(r"^Step\s+\d+\s*:", re.IGNORECASE)
    steps = [l for l in lines if step_pattern.match(l)]
    has_5_steps = len(steps) == 5
    correct_prefix = all(step_pattern.match(s) for s in steps) if steps else False
    format_score = 1.0 if (has_5_steps and correct_prefix) else (len(steps) / 5.0) * 0.7

    # 2. Concreteness: action verbs present, vague terms absent
    text_lower = output.lower()
    verb_hits = sum(1 for v in ACTION_VERBS if v in text_lower)
    vague_hits = sum(1 for v in VAGUE_TERMS if v in text_lower)
    concreteness = min(1.0, verb_hits / 5.0) * (1.0 - min(1.0, vague_hits / 2.0))

    # 3. Relevance: match against expected keywords
    pattern_hits = sum(1 for p in expected_patterns if p.lower() in text_lower)
    relevance = pattern_hits / max(len(expected_patterns), 1)

    stage_score = 0.40 * format_score + 0.30 * concreteness + 0.30 * relevance
    return {
        "format_score": round(format_score, 3),
        "step_count": len(steps),
        "concreteness": round(concreteness, 3),
        "verb_hits": verb_hits,
        "vague_hits": vague_hits,
        "relevance": round(relevance, 3),
        "pattern_hits": pattern_hits,
        "stage_score": round(stage_score, 3),
    }


# ---------------------------------------------------------------------------
# Stage 2: Problem Extractor
# ---------------------------------------------------------------------------

PROBLEM_KEYWORDS = {"time-consuming", "repetitive", "error-prone", "manual", "tedious", "slow", "inefficient"}


def score_stage2_problems(output: str, workflow_steps: list[str]) -> dict:
    """Score the Problem Extractor output."""
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    text_lower = output.lower()

    # 1. Constraint adherence: must include problem-type keywords
    keyword_hits = sum(1 for k in PROBLEM_KEYWORDS if k in text_lower)
    constraint = min(1.0, keyword_hits / 2.0)

    # 2. Coverage: should identify ~5 problems (one per step)
    # Count numbered items or bullet points
    problem_pattern = re.compile(r"^(\d+[\.\):]|\-|\*)\s+", re.MULTILINE)
    problems = problem_pattern.findall(output)
    problem_count = len(problems) if problems else len(lines)
    coverage = 1.0 if problem_count == 5 else max(0, 1.0 - abs(problem_count - 5) / 5.0)

    # 3. Alignment: problems reference workflow step content
    alignment_hits = 0
    for step in workflow_steps:
        step_words = set(step.lower().split())
        if any(w in text_lower for w in step_words if len(w) > 4):
            alignment_hits += 1
    alignment = alignment_hits / max(len(workflow_steps), 1)

    stage_score = 0.30 * constraint + 0.35 * coverage + 0.35 * alignment
    return {
        "constraint_adherence": round(constraint, 3),
        "keyword_hits": keyword_hits,
        "coverage": round(coverage, 3),
        "problem_count": problem_count,
        "alignment": round(alignment, 3),
        "stage_score": round(stage_score, 3),
    }


# ---------------------------------------------------------------------------
# Stage 3: Solution Mapper
# ---------------------------------------------------------------------------

VALID_SOLUTIONS = {
    "automation script", "ocr extraction", "api integration",
    "dashboard", "notification system",
}


def score_stage3_solutions(output: str, problem_count: int) -> dict:
    """Score the Solution Mapper output."""
    text_lower = output.lower()

    # 1. Valid choices: solutions must be from the allowed set
    solution_hits = sum(1 for s in VALID_SOLUTIONS if s in text_lower)
    valid_choices = min(1.0, solution_hits / max(problem_count, 1))

    # 2. Coverage: 1 solution per problem
    # Count solution mentions
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    numbered = re.findall(r"^(\d+[\.\):]|\-|\*)\s+", output, re.MULTILINE)
    solution_count = len(numbered) if numbered else len(lines)
    coverage = 1.0 if solution_count >= problem_count else solution_count / max(problem_count, 1)

    # 3. Fit score: check if solution types match problem types
    # Simple heuristic: "repetitive" → "automation script", "manual" → "automation script", etc.
    fit_pairs = {
        "repetitive": "automation script",
        "manual": "automation script",
        "error-prone": "automation script",
        "data entry": "ocr extraction",
        "document": "ocr extraction",
        "reporting": "dashboard",
        "monitoring": "dashboard",
        "notification": "notification system",
        "alert": "notification system",
        "integration": "api integration",
    }
    fit_hits = sum(1 for k, v in fit_pairs.items() if k in text_lower and v in text_lower)
    fit_score = min(1.0, fit_hits / 3.0)

    stage_score = 0.35 * valid_choices + 0.30 * coverage + 0.35 * fit_score
    return {
        "valid_choices": round(valid_choices, 3),
        "solution_hits": solution_hits,
        "coverage": round(coverage, 3),
        "solution_count": solution_count,
        "fit_score": round(fit_score, 3),
        "stage_score": round(stage_score, 3),
    }


# ---------------------------------------------------------------------------
# Stage 4: Evaluator
# ---------------------------------------------------------------------------

def score_stage4_evaluation(output: str) -> dict:
    """Score the Evaluator output."""
    text_lower = output.lower()

    # 1. Format check: contains scoring structure
    has_scores = bool(re.search(r"\d\s*/\s*5|\bscore\b.*\d|:\s*[1-5]\b", text_lower))
    format_score = 1.0 if has_scores else 0.3

    # 2. Value range: scores between 1-5
    scores_found = [int(x) for x in re.findall(r"\b([1-5])\b(?:\s*/\s*5)?", output)]
    in_range = all(1 <= s <= 5 for s in scores_found) if scores_found else False
    range_score = 1.0 if (in_range and len(scores_found) >= 3) else len(scores_found) / 10.0

    # 3. Consistency: check for feasibility and impact mentions
    has_feasibility = "feasibility" in text_lower or "feasible" in text_lower
    has_impact = "impact" in text_lower
    has_complexity = "complexity" in text_lower or "complex" in text_lower
    consistency = sum([has_feasibility, has_impact, has_complexity]) / 3.0

    stage_score = 0.35 * format_score + 0.35 * range_score + 0.30 * consistency
    return {
        "format_score": round(format_score, 3),
        "has_scores": has_scores,
        "range_score": round(range_score, 3),
        "scores_found": scores_found[:10],
        "consistency": round(consistency, 3),
        "stage_score": round(stage_score, 3),
    }


# ---------------------------------------------------------------------------
# End-to-End: Usefulness Proxy
# ---------------------------------------------------------------------------

def score_usefulness(stage4_output: str) -> dict:
    """Check if ≥2 steps have Feasibility ≥ 4 and Impact ≥ 4."""
    # Extract feasibility and impact scores
    feas = re.findall(r"feasibility[:\s]*([1-5])", stage4_output.lower())
    impact = re.findall(r"impact[:\s]*([1-5])", stage4_output.lower())

    high_feas = sum(1 for f in feas if int(f) >= 4)
    high_impact = sum(1 for i in impact if int(i) >= 4)
    useful = high_feas >= 2 and high_impact >= 2

    return {
        "feasibility_scores": [int(f) for f in feas],
        "impact_scores": [int(i) for i in impact],
        "high_feasibility_count": high_feas,
        "high_impact_count": high_impact,
        "is_useful": useful,
        "usefulness_score": 1.0 if useful else 0.0,
    }
