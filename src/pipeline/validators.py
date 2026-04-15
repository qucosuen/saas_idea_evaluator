"""
Validators for each pipeline stage output.
Parse JSON, enforce schema, return structured data or raise ValueError.
"""

import json
import re
from src.pipeline.schemas import (
    TaskStep, WorkflowTask, StepAnalysis, Solution, Evaluation,
    VALID_SOLUTION_TYPES,
)


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from model output, tolerating markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON array from output: {text[:200]}")


def validate_stage1(output: str) -> list[WorkflowTask]:
    """Validate and parse Stage 1 (Workflow Generator) output — tasks with steps."""
    items = _extract_json_array(output)
    if len(items) < 1:
        raise ValueError(f"Expected at least 1 task, got {len(items)}")
    tasks = []
    for i, item in enumerate(items):
        raw_steps = item.get("steps", [])
        if not isinstance(raw_steps, list) or len(raw_steps) < 2:
            raise ValueError(f"Task {i+1} must have at least 2 steps, got {len(raw_steps) if isinstance(raw_steps, list) else 0}")
        steps = [
            TaskStep(step_number=s.get("step_number", j + 1), description=str(s.get("description", "")))
            for j, s in enumerate(raw_steps)
        ]
        tasks.append(WorkflowTask(
            task_number=item.get("task_number", i + 1),
            title=str(item.get("title", "")),
            steps=steps,
        ))
    return tasks


def validate_stage2(output: str) -> list[StepAnalysis]:
    """Validate and parse Stage 2 (Step Analyzer) output."""
    items = _extract_json_array(output)
    if len(items) < 1:
        raise ValueError("Expected at least 1 step analysis")
    analyses = []
    for item in items:
        analyses.append(StepAnalysis(
            task_number=item.get("task_number", 0),
            step_number=item.get("step_number", 0),
            step_description=str(item.get("step_description", "")),
            current_solution=str(item.get("current_solution", "")),
            problem=str(item.get("problem", "")),
        ))
    return analyses


def validate_stage3(output: str) -> list[Solution]:
    """Validate and parse Stage 3 (Solution Mapper) output."""
    items = _extract_json_array(output)
    if len(items) < 1:
        raise ValueError("Expected at least 1 solution")
    solutions = []
    for item in items:
        stype = str(item.get("solution_type", "")).lower().strip()
        if stype not in VALID_SOLUTION_TYPES:
            stype = "automation script"
        solutions.append(Solution(
            task_number=item.get("task_number", 0),
            step_number=item.get("step_number", 0),
            problem=str(item.get("problem", "")),
            solution_type=stype,
            rationale=str(item.get("rationale", "")),
        ))
    return solutions


def validate_stage4(output: str) -> list[Evaluation]:
    """Validate and parse Stage 4 (Evaluator) output."""
    items = _extract_json_array(output)
    if len(items) < 1:
        raise ValueError("Expected at least 1 evaluation")
    evals = []
    for item in items:
        evals.append(Evaluation(
            task_number=item.get("task_number", 0),
            step_number=item.get("step_number", 0),
            solution_type=str(item.get("solution_type", "")),
            feasibility=max(1, min(5, int(item.get("feasibility", 3)))),
            impact=max(1, min(5, int(item.get("impact", 3)))),
            complexity=max(1, min(5, int(item.get("complexity", 3)))),
        ))
    return evals
