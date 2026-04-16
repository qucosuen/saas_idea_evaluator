"""
Prompt templates for each pipeline stage.
Stage 1: Job → Tasks with Steps
Stage 2: Steps → Current Solutions & Problems
Stage 3: Problems → Automation Solutions
Stage 4: Solutions → Evaluation Scores
All prompts request JSON output for structured parsing.
"""


# STAGE1_SYSTEM = """You list the most critical daily tasks for a job. Each task has detailed sequential steps.
# Output ONLY a JSON array of tasks. Include as many tasks as are critical to the role — do not pad with unimportant tasks.
# Each task: {"task_number": N, "title": "short title", "steps": [{"step_number": 1, "description": "..."}, ...]}
# Each task must have at least 5 detailed steps. Steps must be sequential — each step follows logically from the previous one.
# 
# Example for "Librarian":
# [{"task_number":1,"title":"Process Returns","steps":[{"step_number":1,"description":"Collect returned books from the drop-off bin."},{"step_number":2,"description":"Scan each book barcode to check it in."},{"step_number":3,"description":"Inspect books for damage and flag issues."},{"step_number":4,"description":"Update the book status in the library system."},{"step_number":5,"description":"Place checked-in books on the reshelving cart."}]},{"task_number":2,"title":"Catalog New Items","steps":[{"step_number":1,"description":"Unbox new acquisitions from the delivery."},{"step_number":2,"description":"Verify items against the purchase order."},{"step_number":3,"description":"Enter book details into the library database."},{"step_number":4,"description":"Assign call numbers based on classification system."},{"step_number":5,"description":"Attach barcode labels and spine stickers."}]}]"""

STAGE1_SYSTEM = """You are a workflow analyst describing REALISTIC daily work.

Output ONLY a JSON array of tasks.

STRICT RULES:
- Include ONLY critical tasks (no filler)
- Each task must have at least 5 steps
- Each step must be:
  - a concrete, observable action
  - sequential and logically connected
  - specific to real-world execution
- Each step MUST include at least one of:
  - a tool (e.g., Excel, Jenkins, Docker)
  - a system (e.g., CRM, database, cloud service)
  - a file/artifact (e.g., invoice, report, ticket)
- Avoid vague verbs: "manage", "handle", "ensure", "support", "coordinate"
- Prefer specific verbs: "collect", "validate", "deploy", "configure", "analyze"

FAIL CONDITIONS (must avoid):
- Generic steps with no tools/systems
- Repetitive or redundant steps
- Non-sequential steps

OUTPUT FORMAT:
[
  {
    "task_number": 1,
    "title": "Deploy Application via CI/CD Pipeline",
    "steps": [
      {
        "step_number": 1,
        "description": "Pull the latest code from the GitHub repository using Git."
      },
      {
        "step_number": 2,
        "description": "Trigger the CI pipeline in Jenkins to run automated tests and build artifacts."
      },
      {
        "step_number": 3,
        "description": "Build a Docker image for the application using Docker CLI."
      },
      {
        "step_number": 4,
        "description": "Push the Docker image to a container registry such as AWS ECR."
      },
      {
        "step_number": 5,
        "description": "Deploy the updated container to a Kubernetes cluster using kubectl."
      }
    ]
  },
  {
    "task_number": 2,
    "title": "Monitor System Performance and Alerts",
    "steps": [
      {
        "step_number": 1,
        "description": "Collect system and application metrics using Prometheus exporters."
      },
      {
        "step_number": 2,
        "description": "Visualize metrics on Grafana dashboards for CPU, memory, and latency."
      },
      {
        "step_number": 3,
        "description": "Configure alert rules in Prometheus Alertmanager for threshold breaches."
      },
      {
        "step_number": 4,
        "description": "Receive and review alerts through Slack or PagerDuty notifications."
      },
      {
        "step_number": 5,
        "description": "Investigate logs using Elasticsearch and Kibana to identify root causes."
      }
    ]
  }
]
"""

STAGE1_USER = "List the critical daily tasks with detailed steps for: {job}"

STAGE2_SYSTEM = """You analyze workflow steps and identify current solutions and their limitations.

Output ONLY a JSON array. One entry per step.

STRICT RULES:
- Each step must include:
  - a REALISTIC current solution (tool, software, or method)
  - a SPECIFIC problem (not generic)
- Each problem MUST be one of:
  - time-consuming
  - repetitive
  - error-prone
- The problem MUST explain WHY the current solution fails
- If no clear problem exists, set problem to ""

AVOID:
- Generic problems like "inefficient process"
- Repeating the step description
- Vague reasoning

OUTPUT FORMAT:
[
  {
    "task_number": 1,
    "step_number": 1,
    "step_description": "Pull the latest code from the GitHub repository using Git.",
    "current_solution": "Developers manually run git pull from GitHub repository via CLI or IDE",
    "problem": "Manual pulling across multiple environments is time-consuming and can lead to inconsistent code versions",
    "problem_type": "time-consuming"
  },
  {
    "task_number": 1,
    "step_number": 2,
    "step_description": "Trigger the CI pipeline in Jenkins to run automated tests and build artifacts.",
    "current_solution": "CI pipeline triggered manually or via webhook in Jenkins",
    "problem": "Pipeline failures require repeated manual re-triggering, making the process repetitive",
    "problem_type": "repetitive"
  },
  {
    "task_number": 1,
    "step_number": 3,
    "step_description": "Build a Docker image for the application using Docker CLI.",
    "current_solution": "Docker image built using docker build command in CI environment",
    "problem": "Build failures due to inconsistent dependencies are error-prone and hard to debug",
    "problem_type": "error-prone"
  },
  {
    "task_number": 1,
    "step_number": 4,
    "step_description": "Push the Docker image to a container registry such as AWS ECR.",
    "current_solution": "Docker images pushed manually or via CI pipeline to AWS ECR",
    "problem": "Authentication token expiration can interrupt pushes, requiring manual re-authentication",
    "problem_type": "error-prone"
  },
  {
    "task_number": 1,
    "step_number": 5,
    "step_description": "Deploy the updated container to a Kubernetes cluster using kubectl.",
    "current_solution": "Deployment executed using kubectl apply commands or Helm charts",
    "problem": "Manual deployment steps across environments are time-consuming and prone to configuration drift",
    "problem_type": "time-consuming"
  },
  {
    "task_number": 2,
    "step_number": 1,
    "step_description": "Collect system and application metrics using Prometheus exporters.",
    "current_solution": "Metrics collected via Prometheus exporters configured per service",
    "problem": "",
    "problem_type": ""
  },
  {
    "task_number": 2,
    "step_number": 2,
    "step_description": "Visualize metrics on Grafana dashboards for CPU, memory, and latency.",
    "current_solution": "Grafana dashboards manually configured and updated by engineers",
    "problem": "Updating dashboards for new services is repetitive and requires manual configuration",
    "problem_type": "repetitive"
  },
  {
    "task_number": 2,
    "step_number": 3,
    "step_description": "Configure alert rules in Prometheus Alertmanager for threshold breaches.",
    "current_solution": "Alert rules defined in YAML configuration files for Alertmanager",
    "problem": "Incorrect threshold configuration can lead to missed alerts or false positives, making it error-prone",
    "problem_type": "error-prone"
  },
  {
    "task_number": 2,
    "step_number": 4,
    "step_description": "Receive and review alerts through Slack or PagerDuty notifications.",
    "current_solution": "Alerts sent via Slack or PagerDuty integrations",
    "problem": "",
    "problem_type": ""
  },
  {
    "task_number": 2,
    "step_number": 5,
    "step_description": "Investigate logs using Elasticsearch and Kibana to identify root causes.",
    "current_solution": "Logs queried manually in Kibana dashboards",
    "problem": "Manual log searching across multiple services is time-consuming and inefficient",
    "problem_type": "time-consuming"
  }
]
"""

STAGE2_USER = "Analyze each step in this workflow — identify the current solution and any problems:\n{workflow}"

STAGE3_SYSTEM = """You propose automation solutions for steps that have problems. Output ONLY a JSON array.
Only include steps where a problem was identified (skip steps with no problem).
Each: {"task_number": N, "step_number": M, "problem": "the problem", "solution_type": "one of: Automation Script|OCR Extraction|API Integration|Dashboard|Notification System", "rationale": "why this fits"}

Example:
[{"task_number":1,"step_number":1,"problem":"Time-consuming during peak hours","solution_type":"Automation Script","rationale":"Automated conveyor system can sort returns without staff intervention"},{"task_number":1,"step_number":2,"problem":"Scanner misreads damaged barcodes","solution_type":"OCR Extraction","rationale":"OCR can read book covers when barcodes fail"}]"""

STAGE3_USER = "Propose automation solutions for steps with problems:\n{problems}"

STAGE4_SYSTEM = """You evaluate automation solutions. Output ONLY a JSON array.
Each: {"task_number": N, "step_number": M, "solution_type": "the solution", "feasibility": 1-5, "impact": 1-5, "complexity": 1-5}

Where: feasibility=how easy to implement, impact=how much it improves the step, complexity=implementation difficulty.

Example:
[{"task_number":1,"step_number":1,"solution_type":"Automation Script","feasibility":4,"impact":4,"complexity":2},{"task_number":1,"step_number":2,"solution_type":"OCR Extraction","feasibility":3,"impact":3,"complexity":4}]"""

STAGE4_USER = "Evaluate these solutions:\n{solutions}"
