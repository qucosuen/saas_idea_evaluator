"""
Validators for each pipeline stage output.
Parse JSON, enforce schema, return structured data or raise ValueError.
"""

import json
import re
from src.pipeline.schemas import (
    WorkflowStep, Problem, Solution, Evaluation,
    VALID_PROBLEM_CATEGORIES, VALID_SOLUTION_TYPES,
)


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from model output, tolerating markdown fences."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # Try to find array in text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON array from output: {text[:200]}")


def validate_stage1(output: str) -> list[WorkflowStep]:
    """Validate and parse Stage 1 (Workflow Generator) output."""
    items = _extract_json_array(output)
    if len(items) != 5:
        raise ValueError(f"Expected 5 steps, got {len(items)}")
    steps = []
    for i, item in enumerate(items):
        steps.append(WorkflowStep(
            step_number=item.get("step_number", i + 1),
            title=str(item.get("title", "")),
            description=str(item.get("description", "")),
        ))
    return steps


def validate_stage2(output: str) -> list[Problem]:
    """Validate and parse Stage 2 (Problem Extractor) output."""
    items = _extract_json_array(output)
    if len(items) != 5:
        raise ValueError(f"Expected 5 problems, got {len(items)}")
    problems = []
    for item in items:
        cat = str(item.get("category", "")).lower().strip()
        if cat not in VALID_PROBLEM_CATEGORIES:
            cat = "manual"  # fallback
        problems.append(Problem(
            step_number=item.get("step_number", 0),
            problem=str(item.get("problem", "")),
            category=cat,
        ))
    return problems


def validate_stage3(output: str) -> list[Solution]:
    """Validate and parse Stage 3 (Solution Mapper) output."""
    items = _extract_json_array(output)
    if len(items) != 5:
        raise ValueError(f"Expected 5 solutions, got {len(items)}")
    solutions = []
    for item in items:
        stype = str(item.get("solution_type", "")).lower().strip()
        if stype not in VALID_SOLUTION_TYPES:
            stype = "automation script"  # fallback
        solutions.append(Solution(
            step_number=item.get("step_number", 0),
            problem=str(item.get("problem", "")),
            solution_type=stype,
            rationale=str(item.get("rationale", "")),
        ))
    return solutions


def validate_stage4(output: str) -> list[Evaluation]:
    """Validate and parse Stage 4 (Evaluator) output."""
    items = _extract_json_array(output)
    if len(items) != 5:
        raise ValueError(f"Expected 5 evaluations, got {len(items)}")
    evals = []
    for item in items:
        evals.append(Evaluation(
            step_number=item.get("step_number", 0),
            solution_type=str(item.get("solution_type", "")),
            feasibility=max(1, min(5, int(item.get("feasibility", 3)))),
            impact=max(1, min(5, int(item.get("impact", 3)))),
            complexity=max(1, min(5, int(item.get("complexity", 3)))),
        ))
    return evals
