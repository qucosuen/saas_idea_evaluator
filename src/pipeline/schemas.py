"""
Strict output schemas for each pipeline stage.
All stages produce JSON. Validators enforce structure before passing to next stage.
"""

from dataclasses import dataclass, field


@dataclass
class WorkflowStep:
    step_number: int
    title: str
    description: str


@dataclass
class Problem:
    step_number: int
    problem: str
    category: str  # time-consuming | repetitive | error-prone | manual | tedious


@dataclass
class Solution:
    step_number: int
    problem: str
    solution_type: str  # Automation Script | OCR Extraction | API Integration | Dashboard | Notification System
    rationale: str


@dataclass
class Evaluation:
    step_number: int
    solution_type: str
    feasibility: int  # 1-5
    impact: int       # 1-5
    complexity: int   # 1-5


VALID_PROBLEM_CATEGORIES = {
    "time-consuming", "repetitive", "error-prone", "manual", "tedious",
}

VALID_SOLUTION_TYPES = {
    "automation script", "ocr extraction", "api integration",
    "dashboard", "notification system",
}
