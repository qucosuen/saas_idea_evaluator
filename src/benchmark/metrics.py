"""
Stage-level metrics for the 4-stage LLM pipeline benchmark.

Each stage has specific format, content, and quality metrics.
All scoring functions return a dict with individual scores and a combined stage_score (0-1).

Stage 1 output format (tasks with steps):
  Task 1: <title>
    Step 1: <description>
    Step 2: <description>
    ...
  Task 2: <title>
    ...
"""

import re
import statistics

# ---------------------------------------------------------------------------
# Stage 1: Workflow Generator (Tasks with Steps)
# ---------------------------------------------------------------------------

ACTION_VERBS = {
    "analyze", "assess", "audit", "calculate", "compare", "evaluate",
    "examine", "forecast", "inspect", "measure", "model", "review",
    "test", "validate", "verify",
    "collect", "gather", "extract", "retrieve", "input", "enter",
    "update", "clean", "transform", "process", "store", "archive",
    "backup", "migrate",
    "create", "generate", "build", "develop", "design", "draft",
    "prepare", "produce", "write", "document", "record", "compile",
    "report", "present", "communicate", "notify", "inform",
    "escalate", "coordinate", "collaborate", "brief",
    "execute", "perform", "conduct", "implement", "run", "operate",
    "deploy", "configure", "install", "maintain", "troubleshoot",
    "debug", "optimize", "monitor", "track",
    "organize", "schedule", "plan", "prioritize", "assign",
    "allocate", "manage", "supervise",
    "approve", "authorize", "reconcile", "submit",
    "distribute", "transfer", "file", "close",
    "respond", "resolve", "assist", "support", "handle",
    "address", "follow up",
    "inspect", "enforce", "ensure", "comply", "standardize",
}

VAGUE_TERMS = {
    "do stuff", "handle things", "work on it", "take care of",
    "deal with", "do the needful", "various tasks",
    "miscellaneous", "etc", "and so on",
    "assist with", "help with", "support tasks",
    "be responsible for", "be involved in",
    "participate in", "contribute to",
    "manage tasks", "handle requests", "process things",
    "work on tasks", "perform duties",
    "ensure things are done", "make sure everything works",
    "keep things running", "improve processes",
    "leverage synergies", "drive value", "add value",
    "align with goals", "optimize solutions",
    "general tasks", "daily activities", "assigned duties",
}

# Stop words for coherence overlap checks
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "that", "this", "these",
    "those", "it", "its", "as", "if", "not", "no", "so", "up", "out",
    "all", "any", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "step", "task", "1", "2", "3", "4", "5",
}

_TRANSITION_CUES = {
    "then", "next", "after", "afterwards", "subsequently", "following",
    "based on", "using", "from the", "with the", "once", "upon",
    "resulting", "output", "results", "previous", "earlier", "prior",
    "collected", "gathered", "prepared", "generated", "completed",
    "identified", "reviewed", "processed", "validated", "verified",
}

_EARLY_VERBS = {"collect", "gather", "receive", "review", "check", "input", "enter", "retrieve", "identify", "assess"}
_LATE_VERBS = {"report", "submit", "present", "distribute", "archive", "file", "close", "finalize", "deliver", "send"}


def _tokenize(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return {w for w in words if len(w) > 2 and w not in _STOP_WORDS}


def _parse_tasks(output: str) -> list[dict]:
    """
    Parse task-with-steps output into structured data.
    Handles both JSON format and text format.
    Returns list of {"title": str, "steps": [str, ...]}
    """
    # Try JSON first
    try:
        import json
        text = output.strip()
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list) and len(data) > 0 and "title" in data[0]:
                tasks = []
                for item in data:
                    steps = [s.get("description", "") for s in item.get("steps", [])]
                    tasks.append({"title": item.get("title", ""), "steps": steps})
                return tasks
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fall back to text format
    tasks = []
    current_task = None
    task_re = re.compile(r"^Task\s+\d+\s*:\s*(.*)", re.IGNORECASE)
    step_re = re.compile(r"^\s+Step\s+\d+\s*:\s*(.*)", re.IGNORECASE)

    for line in output.strip().split("\n"):
        line_stripped = line.rstrip()
        tm = task_re.match(line_stripped)
        if tm:
            if current_task:
                tasks.append(current_task)
            current_task = {"title": tm.group(1).strip(), "steps": []}
            continue
        sm = step_re.match(line_stripped)
        if sm and current_task is not None:
            desc = sm.group(1).strip()
            if desc:
                current_task["steps"].append(desc)

    if current_task:
        tasks.append(current_task)
    return tasks


def _score_coherence_within_task(steps: list[str]) -> tuple[float, list[float]]:
    """Score coherence of steps within a single task (0-1)."""
    if len(steps) < 2:
        return 0.0, []

    pair_overlaps = []
    for i in range(len(steps) - 1):
        words_a = _tokenize(steps[i])
        words_b = _tokenize(steps[i + 1])
        if not words_a or not words_b:
            pair_overlaps.append(0.0)
            continue
        shared = words_a & words_b
        overlap = len(shared) / min(len(words_a), len(words_b))
        pair_overlaps.append(min(1.0, overlap))

    overlap_score = sum(pair_overlaps) / len(pair_overlaps) if pair_overlaps else 0.0

    # Transition cues within the task
    full_text = " ".join(steps).lower()
    cue_hits = sum(1 for cue in _TRANSITION_CUES if cue in full_text)
    transition_score = min(1.0, cue_hits / 2.0)

    coherence = 0.60 * overlap_score + 0.40 * transition_score
    return coherence, pair_overlaps


def score_stage1_workflow(output: str, expected_patterns: list[str],
                         min_tasks: int = 3, critical_tasks: list[str] | None = None) -> dict:
    """Score the Workflow Generator output (tasks with steps format)."""
    tasks = _parse_tasks(output)
    text_lower = output.lower()

    # 1. Format: at least 1 task, each with 5+ steps
    task_count = len(tasks)
    tasks_with_enough_steps = sum(1 for t in tasks if len(t["steps"]) >= 5)
    if task_count >= 1 and tasks_with_enough_steps == task_count:
        format_score = 1.0
    elif task_count >= 1:
        format_score = 0.5 + 0.5 * (tasks_with_enough_steps / max(task_count, 1))
    else:
        format_score = 0.0

    # 2. Task count score: penalize too few tasks. 1 task is poor.
    #    Score ramps up: 1→0.2, 2→0.5, min_tasks→0.8, min_tasks+2→1.0
    if task_count == 0:
        task_count_score = 0.0
    elif task_count == 1:
        task_count_score = 0.2
    elif task_count < min_tasks:
        task_count_score = 0.3 + 0.5 * ((task_count - 1) / max(min_tasks - 1, 1))
    else:
        task_count_score = min(1.0, 0.8 + 0.2 * ((task_count - min_tasks) / 2))

    # 3. Task criticality: do the generated task titles match expected critical tasks?
    critical_kw = [c.lower() for c in (critical_tasks or [])]
    if critical_kw:
        all_titles = " ".join(t["title"].lower() for t in tasks)
        all_content = text_lower
        crit_hits = sum(1 for c in critical_kw if c in all_titles or c in all_content)
        criticality_score = min(1.0, crit_hits / max(len(critical_kw) * 0.5, 1))
    else:
        criticality_score = 0.5  # neutral if no expected critical tasks

    # 4. Concreteness: action verbs in steps
    all_steps_text = " ".join(s for t in tasks for s in t["steps"]).lower()
    combined_text = text_lower if not all_steps_text else all_steps_text
    verb_hits = sum(1 for v in ACTION_VERBS if v in combined_text)
    vague_hits = sum(1 for v in VAGUE_TERMS if v in combined_text)
    concreteness = min(1.0, verb_hits / 5.0) * (1.0 - min(1.0, vague_hits / 2.0))

    # 5. Relevance: match against expected keywords
    pattern_hits = sum(1 for p in expected_patterns if p.lower() in text_lower)
    relevance = pattern_hits / max(len(expected_patterns), 1)

    # 6. Duplication penalty
    titles = [t["title"].lower().strip() for t in tasks if t["title"]]
    title_dups = len(titles) - len(set(titles))
    all_steps = [s.lower().strip() for t in tasks for s in t["steps"]]
    step_dups = len(all_steps) - len(set(all_steps))
    total_duplicates = title_dups + step_dups
    echo_dups = len(re.findall(r"(Task\s+\d+\s*:\s*)(Task\s+\d+\s*:)", output, re.IGNORECASE))
    total_duplicates += echo_dups

    if total_duplicates == 0:
        duplication_penalty = 0.0
    elif total_duplicates <= 3:
        duplication_penalty = 0.15
    elif total_duplicates <= 10:
        duplication_penalty = 0.4
    else:
        duplication_penalty = 0.8

    # 7. Emptiness penalty
    empty_tasks = sum(1 for t in tasks if len(t["steps"]) < 5)
    empty_steps = sum(1 for t in tasks for s in t["steps"] if len(s) < 5)
    total_items = max(task_count + len(all_steps), 1)
    emptiness_penalty = round((empty_tasks + empty_steps) / total_items * 0.5, 3)

    # 8. Coherence: do steps within each task form a logical sequence?
    task_coherences = []
    coherence_details = []
    for t in tasks:
        if len(t["steps"]) >= 2:
            coh, overlaps = _score_coherence_within_task(t["steps"])
            task_coherences.append(coh)
            coherence_details.append({
                "task": t["title"],
                "coherence": round(coh, 3),
                "pair_overlaps": [round(o, 3) for o in overlaps],
            })
        else:
            task_coherences.append(0.0)
            coherence_details.append({"task": t.get("title", ""), "coherence": 0.0, "pair_overlaps": []})

    coherence_score = sum(task_coherences) / len(task_coherences) if task_coherences else 0.0

    # Weighted score
    raw_score = (0.15 * format_score + 0.15 * task_count_score + 0.15 * criticality_score
                 + 0.15 * concreteness + 0.15 * relevance + 0.25 * coherence_score)
    stage_score = max(0.0, raw_score - duplication_penalty - emptiness_penalty)

    total_steps = sum(len(t["steps"]) for t in tasks)

    return {
        "format_score": round(format_score, 3),
        "task_count": task_count,
        "task_count_score": round(task_count_score, 3),
        "criticality_score": round(criticality_score, 3),
        "total_steps": total_steps,
        "tasks_with_enough_steps": tasks_with_enough_steps,
        "concreteness": round(concreteness, 3),
        "verb_hits": verb_hits,
        "vague_hits": vague_hits,
        "relevance": round(relevance, 3),
        "pattern_hits": pattern_hits,
        "coherence": round(coherence_score, 3),
        "coherence_details": coherence_details,
        "duplication_penalty": round(duplication_penalty, 3),
        "total_duplicates": total_duplicates,
        "emptiness_penalty": round(emptiness_penalty, 3),
        "empty_tasks": empty_tasks,
        "empty_steps": empty_steps,
        "stage_score": round(stage_score, 3),
    }


# ---------------------------------------------------------------------------
# Stage 2: Step Analyzer (Current Solutions & Problems)
# ---------------------------------------------------------------------------

SOLUTION_SPECIFICITY_KEYWORDS = {
    "software", "tool", "system", "platform", "app", "application",
    "excel", "spreadsheet", "database", "crm", "erp", "email",
    "scanner", "barcode", "manual", "paper", "phone", "calendar",
    "slack", "teams", "zoom", "google", "microsoft", "sap",
    "jira", "trello", "asana", "notion", "confluence",
}


def score_stage2_problems(output: str, workflow_steps: list[str],
                         expected_solutions: list[str] | None = None) -> dict:
    """Score the Step Analyzer output (current solutions & problems)."""
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    text_lower = output.lower()

    # Also try to parse JSON format
    analyses = []
    try:
        import json
        text = output.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list) and len(data) > 0 and "current_solution" in data[0]:
                analyses = data
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # 1. Coverage: how many steps got analyzed
    if analyses:
        analyzed_count = len(analyses)
    else:
        analysis_pattern = re.compile(r"Task\s+\d+.*Step\s+\d+", re.IGNORECASE)
        analyzed_count = sum(1 for l in lines if analysis_pattern.search(l))
    expected_steps = max(len(workflow_steps), 5)
    coverage = min(1.0, analyzed_count / expected_steps) if expected_steps > 0 else 0.0

    # 2. Solution specificity: do the current solutions reference real tools/methods?
    specificity_hits = sum(1 for k in SOLUTION_SPECIFICITY_KEYWORDS if k in text_lower)
    solution_specificity = min(1.0, specificity_hits / 3.0)

    # 3. Solution modernness: do solutions match expected modern tools for this job?
    expected_sol_kw = [s.lower() for s in (expected_solutions or [])]
    if expected_sol_kw:
        modern_hits = sum(1 for s in expected_sol_kw if s in text_lower)
        solution_modernness = min(1.0, modern_hits / max(len(expected_sol_kw) * 0.3, 1))
    else:
        solution_modernness = solution_specificity  # fallback to specificity

    # 4. Solution detail: average length of current_solution descriptions
    if analyses:
        sol_lengths = [len(a.get("current_solution", "")) for a in analyses]
        avg_sol_len = sum(sol_lengths) / len(sol_lengths) if sol_lengths else 0
    else:
        # Extract solution parts from pipe-delimited text
        sol_parts = []
        for l in lines:
            if "|" in l:
                parts = l.split("|")
                if len(parts) >= 2:
                    sol_parts.append(parts[1].strip() if len(parts) >= 3 else parts[0].strip())
        avg_sol_len = sum(len(s) for s in sol_parts) / len(sol_parts) if sol_parts else 0
    # Score: <10 chars is vague, 20+ is good, 40+ is detailed
    if avg_sol_len >= 40:
        solution_detail = 1.0
    elif avg_sol_len >= 20:
        solution_detail = 0.6 + 0.4 * ((avg_sol_len - 20) / 20)
    elif avg_sol_len >= 10:
        solution_detail = 0.3 + 0.3 * ((avg_sol_len - 10) / 10)
    else:
        solution_detail = avg_sol_len / 10 * 0.3

    # 5. Problem identification
    if analyses:
        problems_found = sum(1 for a in analyses if a.get("problem", "").strip())
        no_problem = len(analyses) - problems_found
    else:
        problem_lines = [l for l in lines if "|" in l]
        problems_found = 0
        no_problem = 0
        for pl in problem_lines:
            parts = pl.split("|")
            if len(parts) >= 2:
                prob_part = parts[-1].strip().lower()
                if prob_part and prob_part != "none" and len(prob_part) > 5:
                    problems_found += 1
                else:
                    no_problem += 1
    total_analyzed = problems_found + no_problem
    if total_analyzed > 0:
        problem_rate = problems_found / total_analyzed
        problem_quality = 1.0 - abs(problem_rate - 0.65) * 2
        problem_quality = max(0.0, min(1.0, problem_quality))
    else:
        problem_quality = 0.0

    # 6. Alignment: do analyses reference content from the workflow?
    alignment_hits = 0
    for step in workflow_steps:
        step_words = set(step.lower().split())
        if any(w in text_lower for w in step_words if len(w) > 4):
            alignment_hits += 1
    alignment = alignment_hits / max(len(workflow_steps), 1)

    stage_score = (0.20 * coverage + 0.15 * solution_specificity + 0.15 * solution_modernness
                   + 0.15 * solution_detail + 0.15 * problem_quality + 0.20 * alignment)
    return {
        "coverage": round(coverage, 3),
        "analyzed_count": analyzed_count,
        "solution_specificity": round(solution_specificity, 3),
        "specificity_hits": specificity_hits,
        "solution_modernness": round(solution_modernness, 3),
        "solution_detail": round(solution_detail, 3),
        "avg_solution_length": round(avg_sol_len, 1),
        "problem_quality": round(problem_quality, 3),
        "problems_found": problems_found,
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

    solution_hits = sum(1 for s in VALID_SOLUTIONS if s in text_lower)
    valid_choices = min(1.0, solution_hits / max(problem_count, 1))

    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    numbered = re.findall(r"^(\d+[\.\):]|\-|\*)\s+", output, re.MULTILINE)
    solution_count = len(numbered) if numbered else len(lines)
    coverage = 1.0 if solution_count >= problem_count else solution_count / max(problem_count, 1)

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

    has_scores = bool(re.search(r"\d\s*/\s*5|\bscore\b.*\d|:\s*[1-5]\b", text_lower))
    format_score = 1.0 if has_scores else 0.3

    scores_found = [int(x) for x in re.findall(r"\b([1-5])\b(?:\s*/\s*5)?", output)]
    in_range = all(1 <= s <= 5 for s in scores_found) if scores_found else False
    range_score = 1.0 if (in_range and len(scores_found) >= 3) else len(scores_found) / 10.0

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
    """Check if ≥2 tasks have Feasibility ≥ 4 and Impact ≥ 4."""
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
