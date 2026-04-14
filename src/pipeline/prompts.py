"""
Prompt templates for each pipeline stage.
Phase 5: Includes few-shot examples for improved output quality.
All prompts request JSON output for structured parsing.
"""

STAGE1_SYSTEM = """You generate daily workflows for jobs. Output ONLY a JSON array of exactly 5 steps.
Each step: {"step_number": N, "title": "short title", "description": "one sentence"}

Example for "Librarian":
[{"step_number":1,"title":"Process Returns","description":"Check in returned books and inspect for damage."},{"step_number":2,"title":"Shelve Materials","description":"Sort and shelve returned items in correct locations."},{"step_number":3,"title":"Assist Patrons","description":"Help visitors find books and answer reference questions."},{"step_number":4,"title":"Catalog New Items","description":"Enter new acquisitions into the library database system."},{"step_number":5,"title":"Close Registers","description":"Reconcile checkout records and prepare end-of-day reports."}]"""

STAGE1_USER = "Generate a 5-step daily workflow for: {job}"

STAGE2_SYSTEM = """You identify problems in workflows. Output ONLY a JSON array of exactly 5 problems.
Each problem: {"step_number": N, "problem": "description", "category": "one of: time-consuming|repetitive|error-prone|manual|tedious"}

Example:
[{"step_number":1,"problem":"Manually checking each returned book for damage is slow","category":"time-consuming"},{"step_number":2,"problem":"Shelving books by hand in correct order is repetitive","category":"repetitive"},{"step_number":3,"problem":"Looking up references without a search tool is tedious","category":"tedious"},{"step_number":4,"problem":"Typing catalog entries by hand introduces typos","category":"error-prone"},{"step_number":5,"problem":"Manually tallying checkout records is error-prone","category":"error-prone"}]"""

STAGE2_USER = "Identify one problem per step in this workflow:\n{workflow}"

STAGE3_SYSTEM = """You propose automation solutions. Output ONLY a JSON array of exactly 5 solutions.
Each: {"step_number": N, "problem": "the problem", "solution_type": "one of: Automation Script|OCR Extraction|API Integration|Dashboard|Notification System", "rationale": "why this fits"}

Example:
[{"step_number":1,"problem":"Manually checking books is slow","solution_type":"Automation Script","rationale":"Barcode scanning automates check-in and flags damage automatically"},{"step_number":2,"problem":"Shelving is repetitive","solution_type":"Dashboard","rationale":"A dashboard showing optimal shelving routes reduces wasted time"},{"step_number":3,"problem":"Reference lookup is tedious","solution_type":"API Integration","rationale":"Integrating a search API lets patrons self-serve"},{"step_number":4,"problem":"Catalog entry has typos","solution_type":"OCR Extraction","rationale":"OCR can scan book details and auto-populate catalog fields"},{"step_number":5,"problem":"Manual tallying is error-prone","solution_type":"Automation Script","rationale":"A script can auto-reconcile checkout records at end of day"}]"""

STAGE3_USER = "Propose one solution per problem:\n{problems}"

STAGE4_SYSTEM = """You evaluate automation solutions. Output ONLY a JSON array of exactly 5 evaluations.
Each: {"step_number": N, "solution_type": "the solution", "feasibility": 1-5, "impact": 1-5, "complexity": 1-5}

Where: feasibility=how easy to implement, impact=how much it improves the workflow, complexity=implementation difficulty.

Example:
[{"step_number":1,"solution_type":"Automation Script","feasibility":4,"impact":4,"complexity":2},{"step_number":2,"solution_type":"Dashboard","feasibility":3,"impact":3,"complexity":3},{"step_number":3,"solution_type":"API Integration","feasibility":4,"impact":5,"complexity":3},{"step_number":4,"solution_type":"OCR Extraction","feasibility":3,"impact":4,"complexity":4},{"step_number":5,"solution_type":"Automation Script","feasibility":5,"impact":4,"complexity":1}]"""

STAGE4_USER = "Evaluate these solutions:\n{solutions}"
