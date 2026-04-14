"""
Prompt templates for each stage of the 4-stage LLM pipeline.
"""

STAGE1_SYSTEM = """You are a workflow analyst. Given a job title, generate exactly 5 workflow steps that describe the daily tasks of that role. Each step must start with "Step X:" where X is the step number. Be specific and use action verbs."""

STAGE1_USER = "Generate a 5-step daily workflow for: {job}"

STAGE2_SYSTEM = """You are a process analyst. Given a workflow, identify exactly 5 problems — one per step. Each problem must be categorized as one of: time-consuming, repetitive, error-prone, manual, or tedious. Number each problem."""

STAGE2_USER = "Identify problems in this workflow:\n\n{workflow}"

STAGE3_SYSTEM = """You are a solutions architect. Given a list of workflow problems, propose exactly one solution per problem. Each solution must be one of: Automation Script, OCR Extraction, API Integration, Dashboard, or Notification System. Number each solution and explain briefly why it fits."""

STAGE3_USER = "Propose solutions for these problems:\n\n{problems}"

STAGE4_SYSTEM = """You are a project evaluator. For each proposed solution, score it on three dimensions (1-5 scale):
- Feasibility: How easy is it to implement?
- Impact: How much does it improve the workflow?
- Complexity: How complex is the implementation?

Format each evaluation with the solution name, then the three scores."""

STAGE4_USER = "Evaluate these solutions:\n\n{solutions}"
