"""
Benchmark scoring system.
All scores are normalized to 0.0 - 1.0 range.

Stage scores:
  - Task Score: coverage, relevance, specificity, non-redundancy
  - Step Score: atomicity, logical ordering, actionability, specificity
  - Problem Score: specificity, realism, causality
  - Solution Score: correctness, specificity, diversity, feasibility, novelty
  - Evaluation Score: variance, justification, specificity, usefulness

System-level:
  - Consistency, redundancy penalty, constraint adherence, practicality

Final score: weighted average of all stage scores + system bonus, 0.0-1.0
"""

import json
import re


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}


def _keyword_hits(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for k in keywords if k.lower() in text_lower)


def _bad_pattern_hits(text: str, bad_patterns: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for p in bad_patterns if p.lower() in text_lower)


_VAGUE_PHRASES = [
    "various",
    "general",
    "miscellaneous",
    "etc",
    "and so on",
    "handle things",
    "do stuff",
    "take care of",
    "deal with",
    "as needed",
    "if necessary",
    "when required",
    "as appropriate",
    "ensure quality",
    "improve processes",
    "manage tasks",
    "perform duties",
    "assigned duties",
    "daily activities",
]


def _specificity_score(texts: list[str], min_good_len: int = 40) -> float:
    """Score specificity of text strings (0.0-1.0)."""
    if not texts:
        return 0.0
    scores = []
    for t in texts:
        if not t or len(t.strip()) < 5:
            scores.append(0.0)
            continue
        s = 0.0
        s += min(0.4, len(t) / (min_good_len * 2.5))
        proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", t)
        acronyms = re.findall(r"\b[A-Z]{2,}\b", t)
        numbers = re.findall(r"\b\d+\b", t)
        s += min(0.3, (len(proper_nouns) + len(acronyms) * 2 + len(numbers)) * 0.1)
        vague_count = sum(1 for v in _VAGUE_PHRASES if v in t.lower())
        s -= vague_count * 0.15
        scores.append(max(0.0, min(1.0, s)))
    return round(sum(scores) / len(scores), 3)


# ---------------------------------------------------------------------------
# Stage 1: Task Score (0.0-1.0)
# ---------------------------------------------------------------------------


def score_tasks(parsed_tasks: list[dict], ground_truth: dict) -> dict:
    gt_tasks = ground_truth.get("tasks", [])
    if not parsed_tasks:
        return {
            "coverage": 0,
            "relevance": 0,
            "specificity": 0,
            "non_redundancy": 0,
            "score": 0,
        }

    all_text = json.dumps(parsed_tasks).lower()

    hits = _keyword_hits(all_text, gt_tasks)
    coverage = min(1.0, hits / max(len(gt_tasks), 1))

    generic = ["general", "miscellaneous", "other", "various", "daily tasks"]
    generic_count = sum(
        1 for t in parsed_tasks if any(g in t.get("title", "").lower() for g in generic)
    )
    relevance = max(0, 1.0 - generic_count * 0.25)

    titles = [t.get("title", "") for t in parsed_tasks]
    all_step_descs = [
        s.get("description", "") if isinstance(s, dict) else str(s)
        for t in parsed_tasks
        for s in t.get("steps", [])
    ]
    specificity = _specificity_score(titles + all_step_descs, min_good_len=30)

    title_strs = [t.get("title", "").lower().strip() for t in parsed_tasks]
    non_redundancy = len(set(title_strs)) / max(len(title_strs), 1)

    score = round(
        0.30 * coverage + 0.25 * relevance + 0.25 * specificity + 0.20 * non_redundancy,
        3,
    )
    return {
        "coverage": round(coverage, 3),
        "relevance": round(relevance, 3),
        "specificity": round(specificity, 3),
        "non_redundancy": round(non_redundancy, 3),
        "score": score,
    }


# ---------------------------------------------------------------------------
# Stage 1: Step Score (0.0-1.0)
# ---------------------------------------------------------------------------


def score_steps(parsed_tasks: list[dict], ground_truth: dict) -> dict:
    all_steps = []
    for t in parsed_tasks:
        for s in t.get("steps", []):
            desc = s.get("description", "") if isinstance(s, dict) else str(s)
            all_steps.append(desc)

    if not all_steps:
        return {
            "atomicity": 0,
            "logical_ordering": 0,
            "actionability": 0,
            "specificity": 0,
            "score": 0,
        }

    gt_keywords = ground_truth.get("steps_keywords", [])
    all_text = " ".join(all_steps).lower()

    compound_markers = [" and then ", " followed by ", " while also ", "; "]
    compound_count = sum(
        1 for s in all_steps for m in compound_markers if m in s.lower()
    )
    atomicity = max(0, 1.0 - compound_count * 0.15)

    overlaps = []
    for t in parsed_tasks:
        steps = [
            s.get("description", "") if isinstance(s, dict) else str(s)
            for s in t.get("steps", [])
        ]
        for i in range(len(steps) - 1):
            w1, w2 = _tokenize(steps[i]), _tokenize(steps[i + 1])
            if w1 and w2:
                overlaps.append(len(w1 & w2) / min(len(w1), len(w2)))
    logical_ordering = min(1.0, (sum(overlaps) / len(overlaps) * 5) if overlaps else 0)

    kw_hits = _keyword_hits(all_text, gt_keywords)
    actionability = min(1.0, kw_hits / max(len(gt_keywords) * 0.4, 1))

    specificity = _specificity_score(all_steps, min_good_len=40)

    score = round(
        0.25 * atomicity
        + 0.25 * logical_ordering
        + 0.25 * actionability
        + 0.25 * specificity,
        3,
    )
    return {
        "atomicity": round(atomicity, 3),
        "logical_ordering": round(logical_ordering, 3),
        "actionability": round(actionability, 3),
        "specificity": round(specificity, 3),
        "score": score,
    }


# ---------------------------------------------------------------------------
# Stage 2: Problem Score (0.0-1.0)
# ---------------------------------------------------------------------------


def score_problems(parsed_analyses: list[dict], ground_truth: dict) -> dict:
    if not parsed_analyses:
        return {"specificity": 0, "realism": 0, "causality": 0, "score": 0}

    problems = [a.get("problem", "") for a in parsed_analyses if a.get("problem")]
    if not problems:
        return {"specificity": 0, "realism": 0, "causality": 0, "score": 0}

    gt_keywords = ground_truth.get("problems_keywords", [])
    all_text = " ".join(problems).lower()

    specificity = _specificity_score(problems, min_good_len=50)

    hits = _keyword_hits(all_text, gt_keywords)
    realism = min(1.0, hits / max(len(gt_keywords) * 0.3, 1))

    causal_words = [
        "because",
        "due to",
        "causes",
        "leads to",
        "results in",
        "prone to",
        "risk of",
        "which means",
    ]
    causal_hits = sum(1 for p in problems for c in causal_words if c in p.lower())
    causality = min(1.0, causal_hits / max(len(problems) * 0.3, 1))

    score = round(0.40 * specificity + 0.35 * realism + 0.25 * causality, 3)
    return {
        "specificity": round(specificity, 3),
        "realism": round(realism, 3),
        "causality": round(causality, 3),
        "score": score,
    }


# ---------------------------------------------------------------------------
# Stage 3: Solution Score (0.0-1.0)
# ---------------------------------------------------------------------------


def score_solutions(
    parsed_solutions: list[dict], ground_truth: dict, bad_patterns: list[str]
) -> dict:
    if not parsed_solutions:
        return {
            "correctness": 0,
            "specificity": 0,
            "diversity": 0,
            "feasibility": 0,
            "novelty": 0,
            "score": 0,
        }

    acceptable = ground_truth.get("acceptable_solutions", [])
    all_text = json.dumps(parsed_solutions).lower()

    hits = _keyword_hits(all_text, acceptable)
    correctness = min(1.0, hits / max(len(acceptable) * 0.2, 1))

    rationales = [s.get("rationale", "") for s in parsed_solutions]
    specificity = _specificity_score(rationales, min_good_len=50)

    types = [s.get("solution_type", "").lower().strip() for s in parsed_solutions]
    unique_types = len(set(t for t in types if t))
    diversity = min(1.0, unique_types / max(min(len(parsed_solutions), 4), 1))

    with_rationale = sum(
        1 for s in parsed_solutions if len(s.get("rationale", "")) > 10
    )
    feasibility = round(with_rationale / max(len(parsed_solutions), 1), 3)

    bad_hits = _bad_pattern_hits(all_text, bad_patterns)
    novelty = max(0, 1.0 - bad_hits * 0.25)

    score = round(
        0.25 * correctness
        + 0.25 * specificity
        + 0.20 * diversity
        + 0.15 * feasibility
        + 0.15 * novelty,
        3,
    )
    return {
        "correctness": round(correctness, 3),
        "specificity": round(specificity, 3),
        "diversity": round(diversity, 3),
        "feasibility": round(feasibility, 3),
        "novelty": round(novelty, 3),
        "score": score,
    }


# ---------------------------------------------------------------------------
# Stage 4: Evaluation Score (0.0-1.0)
# ---------------------------------------------------------------------------


def score_evaluations(parsed_evals: list[dict]) -> dict:
    if not parsed_evals:
        return {
            "variance": 0,
            "justification": 0,
            "specificity": 0,
            "usefulness": 0,
            "score": 0,
        }

    feas = [e.get("feasibility", 3) for e in parsed_evals]
    impacts = [e.get("impact", 3) for e in parsed_evals]
    complexities = [e.get("complexity", 3) for e in parsed_evals]
    all_scores = feas + impacts + complexities

    if len(all_scores) > 1:
        import statistics

        std = statistics.stdev(all_scores)
        variance = min(1.0, std / 1.5)
    else:
        variance = 0

    complete = sum(
        1
        for e in parsed_evals
        if e.get("feasibility") and e.get("impact") and e.get("complexity")
    )
    justification = complete / max(len(parsed_evals), 1)

    unique_combos = set()
    for e in parsed_evals:
        unique_combos.add(
            (e.get("feasibility", 0), e.get("impact", 0), e.get("complexity", 0))
        )
    specificity = len(unique_combos) / max(len(parsed_evals), 1)

    high_quality = sum(
        1
        for e in parsed_evals
        if e.get("feasibility", 0) >= 4 and e.get("impact", 0) >= 4
    )
    usefulness = min(1.0, high_quality / 2)

    score = round(
        0.25 * variance + 0.30 * justification + 0.20 * specificity + 0.25 * usefulness,
        3,
    )
    return {
        "variance": round(variance, 3),
        "justification": round(justification, 3),
        "specificity": round(specificity, 3),
        "usefulness": round(usefulness, 3),
        "score": score,
    }


# ---------------------------------------------------------------------------
# System-Level Scores (each 0.0-1.0)
# ---------------------------------------------------------------------------


def score_system(
    parsed_tasks, parsed_analyses, parsed_solutions, parsed_evals, bad_patterns
) -> dict:
    all_text = json.dumps(
        (parsed_tasks or [])
        + (parsed_analyses or [])
        + (parsed_solutions or [])
        + (parsed_evals or [])
    ).lower()

    consistency = 1.0
    if parsed_solutions and parsed_analyses:
        sol_problems = {s.get("problem", "").lower()[:30] for s in parsed_solutions}
        ana_problems = {
            a.get("problem", "").lower()[:30]
            for a in parsed_analyses
            if a.get("problem")
        }
        if ana_problems:
            overlap = len(sol_problems & ana_problems) / max(len(sol_problems), 1)
            consistency = min(1.0, overlap + 0.3)

    if parsed_solutions:
        rationales = [s.get("rationale", "").lower().strip() for s in parsed_solutions]
        redundancy = len(set(rationales)) / max(len(rationales), 1)
    else:
        redundancy = 0

    bad_hits = _bad_pattern_hits(all_text, bad_patterns)
    constraint_adherence = max(0, 1.0 - bad_hits * 0.2)

    stages_present = sum(
        [
            bool(parsed_tasks),
            bool(parsed_analyses),
            bool(parsed_solutions),
            bool(parsed_evals),
        ]
    )
    practicality = stages_present / 5

    return {
        "consistency": round(consistency, 3),
        "redundancy": round(redundancy, 3),
        "constraint_adherence": round(constraint_adherence, 3),
        "practicality": round(practicality, 3),
    }


# ---------------------------------------------------------------------------
# Final Aggregation (0.0-1.0)
# ---------------------------------------------------------------------------


def score_pipeline(run: dict, sample: dict) -> dict:
    stages = run.get("stages", [])
    gt = sample.get("ground_truth", {})
    bad = sample.get("bad_patterns", [])

    s1 = (
        stages[0].get("parsed", [])
        if len(stages) > 0 and stages[0].get("valid")
        else []
    )
    s2a = (
        stages[1].get("parsed", [])
        if len(stages) > 1 and stages[1].get("valid")
        else []
    )
    s2b = (
        stages[2].get("parsed", [])
        if len(stages) > 2 and stages[2].get("valid")
        else []
    )
    s3 = (
        stages[3].get("parsed", [])
        if len(stages) > 3 and stages[3].get("valid")
        else []
    )
    s4 = (
        stages[4].get("parsed", [])
        if len(stages) > 4 and stages[4].get("valid")
        else []
    )

    # Merge stage 2A (problems) and 2B (current solutions) for analyses
    merged_analyses = []
    if s2a and s2b:
        problem_map = {
            (p.get("task_number", 0), p.get("step_number", 0)): p for p in s2a
        }
        for sol in s2b:
            key = (sol.get("task_number", 0), sol.get("step_number", 0))
            prob = problem_map.get(key, {})
            merged_analyses.append(
                {
                    "task_number": sol.get("task_number", 0),
                    "step_number": sol.get("step_number", 0),
                    "step_description": sol.get("step_description", ""),
                    "current_solution": sol.get("current_solution", ""),
                    "problem": prob.get("problem", ""),
                }
            )
    elif s2b:
        merged_analyses = s2b
    elif s2a:
        merged_analyses = s2a

    task_scores = score_tasks(s1, gt)
    step_scores = score_steps(s1, gt)
    problem_scores = score_problems(s2a, gt)
    solution_scores = score_solutions(s3, gt, bad)
    eval_scores = score_evaluations(s4)
    system_scores = score_system(s1, merged_analyses, s3, s4, bad)

    # Weighted stage score
    stage_score = (
        0.20 * task_scores["score"]
        + 0.20 * step_scores["score"]
        + 0.20 * problem_scores["score"]
        + 0.25 * solution_scores["score"]
        + 0.15 * eval_scores["score"]
    )

    # System bonus (up to +0.1)
    sys_avg = (
        system_scores["consistency"]
        + system_scores["redundancy"]
        + system_scores["constraint_adherence"]
        + system_scores["practicality"]
    ) / 4
    sys_bonus = sys_avg * 0.1

    final = min(1.0, stage_score + sys_bonus)

    return {
        "job": run.get("job", ""),
        "difficulty": sample.get("difficulty", ""),
        "success": run.get("success", False),
        "task_scores": task_scores,
        "step_scores": step_scores,
        "problem_scores": problem_scores,
        "solution_scores": solution_scores,
        "eval_scores": eval_scores,
        "system_scores": system_scores,
        "final_score": round(final, 3),
        "latency_ms": run.get("total_latency_ms", 0),
    }
