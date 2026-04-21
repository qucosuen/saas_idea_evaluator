"""
Strict output schemas for each pipeline stage.
All stages produce JSON. Validators enforce structure before passing to next stage.
"""

from dataclasses import dataclass, field


@dataclass
class TaskStep:
    step_number: int
    description: str
    tool_stack: list[str] = field(default_factory=list)


@dataclass
class WorkflowTask:
    task_number: int
    title: str
    steps: list[TaskStep] = field(default_factory=list)


@dataclass
class StepProblem:
    task_number: int
    step_number: int
    step_description: str
    problem: str


@dataclass
class StepSolution:
    task_number: int
    step_number: int
    step_description: str
    current_solution: str


@dataclass
class StepAnalysis:
    task_number: int
    step_number: int
    step_description: str
    current_solution: str  # what tool/method is currently used for this step
    problem: str  # problem with the current solution (empty string if none)


@dataclass
class Solution:
    task_number: int
    step_number: int
    problem: str
    solution_type: str  # Automation Script | OCR Extraction | API Integration | Dashboard | Notification System
    rationale: str


@dataclass
class Evaluation:
    task_number: int
    step_number: int
    solution_type: str
    feasibility: int  # 1-5
    impact: int  # 1-5
    complexity: int  # 1-5


VALID_PROBLEM_CATEGORIES = {
    "time-consuming",
    "repetitive",
    "error-prone",
    "manual",
    "tedious",
}

VALID_SOLUTION_TYPES = {
    "automation script",
    "ocr extraction",
    "api integration",
    "dashboard",
    "notification system",
}
