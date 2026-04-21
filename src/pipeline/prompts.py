"""
Prompt templates for each pipeline stage.
Stage 1: Job → Tasks with Steps
Stage 2A: Steps → Problems
Stage 2B: Problems → Current Solutions
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

STAGE1_SYSTEM = """
You are a workflow analyst describing REALISTIC daily work for a {job}.

Output ONLY a JSON array of tasks.

LIMITS:
- Each task must have at least 10 steps.

ROLE FIDELITY:
- Each task must reflect real responsibilities of a {job}
- Avoid tasks typically performed by other roles

STRICT RULES:

- Each step must be:
  - a concrete, observable action
  - sequential and logically connected
  - realistic in real-world execution

- Each step description must:
  - describe WHAT is done (action + object)
  - NOT depend on any specific tool or product name
  - remain valid across different companies and tech stacks

- Each step must:
  - act on something specific (object, data, request, system state)
  - produce a visible result or state change

- Each step must include a separate "tool_stack" field:
  - a list of relevant tools, systems, or methods that could be used
  - include multiple possible options where appropriate
  - tools should be commonly used in the industry, not obscure

- Avoid vague verbs:
  "manage", "handle", "ensure", "assist", "coordinate"

- Prefer specific, observable actions:
  "record", "verify", "update", "inspect", "transfer", "deploy", "analyze"

DIVERSITY RULE:
- Tasks must represent different types of responsibilities
- Steps must not repeat the same structure or phrasing

FAIL CONDITIONS:
- Generic or vague steps
- Steps that depend on a specific tool to make sense
- Missing "tool_stack" field
- Non-sequential steps
- Tasks not aligned with the role

EXAMPLES:

BAD STEP:
"description": "Trigger CI pipeline in Jenkins"

GOOD STEP:
"description": "Initiate an automated build and test process after code changes are submitted",
"tool_stack": ["Jenkins", "GitHub Actions", "GitLab CI"]

BAD:
"Handle customer request"

GOOD:
"description": "Record customer request details into a tracking system",
"tool_stack": ["CRM system", "ticketing system"]

SELF-CHECK:
- Step descriptions are tool-agnostic
- Each step includes a non-empty tool_stack
- Each step produces a visible outcome
- Tasks reflect real job responsibilities

OUTPUT FORMAT:
[
  {
    "task_number": 1,
    "title": "Deploy Application via CI/CD Pipeline",
    "steps": [
      {
        "step_number": 1,
        "description": "Retrieve the latest version of the application source code from the version control repository.",
        "tool_stack": ["Git", "GitHub", "GitLab", "Bitbucket"]
      },
      {
        "step_number": 2,
        "description": "Initiate an automated build and testing process based on the updated source code.",
        "tool_stack": ["Jenkins", "GitHub Actions", "GitLab CI", "CircleCI"]
      },
      {
        "step_number": 3,
        "description": "Package the application into a deployable artifact suitable for the target environment.",
        "tool_stack": ["Docker", "Buildpacks", "Maven", "Gradle"]
      },
      {
        "step_number": 4,
        "description": "Store the built artifact in a centralized repository for versioned access and distribution.",
        "tool_stack": ["AWS ECR", "Docker Hub", "Artifact Registry", "Nexus"]
      },
      {
        "step_number": 5,
        "description": "Release the new application version to the target runtime environment and update the running service.",
        "tool_stack": ["Kubernetes", "AWS ECS", "Azure App Service"]
      }
    ]
  },
  {
    "task_number": 2,
    "title": "Monitor System Performance and Alerts",
    "steps": [
      {
        "step_number": 1,
        "description": "Collect performance and operational metrics from application components and infrastructure resources.",
        "tool_stack": ["Prometheus", "CloudWatch", "Datadog", "New Relic"]
      },
      {
        "step_number": 2,
        "description": "Aggregate and display collected metrics through dashboards for real-time visibility.",
        "tool_stack": ["Grafana", "Kibana", "Datadog Dashboards"]
      },
      {
        "step_number": 3,
        "description": "Define threshold-based conditions to detect abnormal system behavior.",
        "tool_stack": ["Prometheus Alertmanager", "Datadog Alerts", "CloudWatch Alarms"]
      },
      {
        "step_number": 4,
        "description": "Notify responsible teams when alert conditions are triggered.",
        "tool_stack": ["Slack", "PagerDuty", "Opsgenie", "Email Systems"]
      },
      {
        "step_number": 5,
        "description": "Examine system logs and metrics to identify the root cause of detected issues.",
        "tool_stack": ["Elasticsearch", "Kibana", "Splunk", "Cloud Logging"]
      }
    ]
  }
]
"""

STAGE1_USER = "List the critical daily tasks with detailed steps for: {job}"

STAGE2A_SYSTEM = """
You analyze workflow steps and identify REALISTIC problems in current execution.

Output ONLY a JSON array. One entry per step.

CORE PRINCIPLE:
A good problem explains a SPECIFIC failure, limitation, or friction in how the step is currently performed.

STRICT RULES:

- Each problem MUST describe a concrete failure mechanism, such as:
  - what breaks
  - when it breaks
  - why it breaks
  - under what conditions it degrades

- Each problem MUST fall into ONE of these categories:
  - failure_mode: something produces incorrect or unreliable results
  - bottleneck: something slows down execution due to a specific constraint
  - scalability: something fails or degrades as volume/complexity increases
  - coordination: multiple people/systems must align and cause friction
  - observability: lack of visibility makes it hard to detect or debug issues

- The problem MUST:
  - reference a specific part of the step
  - describe a cause-and-effect relationship
  - be realistic in real-world workflows (any industry)

- If no clear problem exists, set:
  "problem": "",
  "problem_type": ""

FORBIDDEN:
- Generic phrases like:
  "time-consuming"
  "repetitive"
  "error-prone"
  UNLESS paired with a clear mechanism explaining WHY

- Restating the step
- Vague issues like "inefficient process"
- Overly abstract statements without concrete behavior

QUALITY BAR:
Bad:
"Manual processing is time-consuming"

Good:
"Data must be re-entered across multiple systems without synchronization, causing delays and inconsistent records when updates occur in parallel"

OUTPUT FORMAT:
[
  {
    "task_number": 1,
    "step_number": 1,
    "step_description": "Retrieve the latest version of the application source code from the version control repository.",
    "problem": "When multiple repositories are involved, there is no single synchronized snapshot, leading to mismatched dependency versions during integration",
    "problem_type": "failure_mode"
  },
  {
    "task_number": 1,
    "step_number": 2,
    "step_description": "Initiate an automated build and testing process based on the updated source code.",
    "problem": "Build pipelines depend on cached dependencies that may differ across environments, causing inconsistent test results between runs",
    "problem_type": "failure_mode"
  },
  {
    "task_number": 1,
    "step_number": 3,
    "step_description": "Package the application into a deployable artifact suitable for the target environment.",
    "problem": "Packaging configurations are environment-specific, requiring manual overrides that can diverge over time and introduce inconsistencies",
    "problem_type": "coordination"
  },
  {
    "task_number": 1,
    "step_number": 4,
    "step_description": "Store the built artifact in a centralized repository for versioned access and distribution.",
    "problem": "Artifact repositories do not enforce strict version immutability, allowing overwritten or duplicated versions that create ambiguity during deployment",
    "problem_type": "failure_mode"
  },
  {
    "task_number": 1,
    "step_number": 5,
    "step_description": "Release the new application version to the target runtime environment and update the running service.",
    "problem": "Deployment configurations differ across environments, requiring manual adjustments that increase the risk of configuration drift over time",
    "problem_type": "coordination"
  },
  {
    "task_number": 2,
    "step_number": 1,
    "step_description": "Collect performance and operational metrics from system components.",
    "problem": "Metrics collection depends on agents configured per component, and missing instrumentation leads to blind spots in system visibility",
    "problem_type": "observability"
  },
  {
    "task_number": 2,
    "step_number": 2,
    "step_description": "Aggregate and display collected metrics through dashboards.",
    "problem": "Dashboards require manual updates when new components are added, causing gaps where critical metrics are not visualized",
    "problem_type": "coordination"
  },
  {
    "task_number": 2,
    "step_number": 3,
    "step_description": "Define threshold-based conditions to detect abnormal behavior.",
    "problem": "Static thresholds do not adapt to changing workloads, leading to frequent false positives during peak periods and missed anomalies during low activity",
    "problem_type": "failure_mode"
  },
  {
    "task_number": 2,
    "step_number": 4,
    "step_description": "Notify responsible teams when alert conditions are triggered.",
    "problem": "",
    "problem_type": ""
  },
  {
    "task_number": 2,
    "step_number": 5,
    "step_description": "Examine logs and metrics to identify the root cause of detected issues.",
    "problem": "Logs from different systems are not correlated by a unified identifier, making it difficult to trace requests across services during debugging",
    "problem_type": "observability"
  }
]
"""

STAGE2A_USER = "Identify problems for each step in this workflow:\n{workflow}"

STAGE2B_SYSTEM = """
You analyze workflow steps and describe how each step is ACTUALLY executed today in real-world environments.

Output ONLY a JSON array. One entry per step.

GOAL:
- Describe the CURRENT REAL-WORLD EXECUTION of each step
- Focus on how people and systems interact to complete the step

STRICT RULES:

1. DEFINE CURRENT SOLUTION AS:
- A description of how the step is performed in practice
- Include:
  - human actions (what people do)
  - system behavior (what systems/tools do)
  - interaction flow (how actions trigger systems)

2. DO NOT JUST NAME TOOLS
- Tool names are optional, not required
- If tools are mentioned, they must be embedded in a workflow description
- NEVER output tool-only descriptions

BAD:
"Prometheus used to collect metrics"

GOOD:
"Engineers configure monitoring agents on each service, which continuously send metrics to a centralized monitoring system for aggregation and querying"

3. INCLUDE EXECUTION DYNAMICS
Each current_solution should reflect at least one:
- how the step is triggered (manual, automated, scheduled, event-based)
- how humans interact with the system
- where configuration or setup is required

4. AVOID:
- repeating the step description
- generic phrases like "is used to"
- overly specific vendor lock-in (e.g., always saying Jenkins, Prometheus)
- assuming everything is manual

5. KEEP IT REALISTIC:
- Reflect modern, commonly used practices
- Solutions should generalize across organizations

6. LENGTH:
- Each current_solution should be 1–2 sentences
- Must contain enough detail to expose how the workflow operates

OUTPUT FORMAT:
[
  {
    "task_number": 1,
    "step_number": 1,
    "step_description": "Pull the latest code from the GitHub repository using Git.",
    "current_solution": "Developers synchronize their local environment with the remote repository by fetching and merging recent changes, typically triggered before starting new work or deployments."
  },
  {
    "task_number": 1,
    "step_number": 2,
    "step_description": "Trigger the CI pipeline in Jenkins to run automated tests and build artifacts.",
    "current_solution": "Code changes pushed to the repository automatically trigger a build pipeline through repository events, while engineers monitor execution and manually re-trigger jobs if failures occur."
  },
  {
    "task_number": 1,
    "step_number": 3,
    "step_description": "Build a Docker image for the application using Docker CLI.",
    "current_solution": "The build system packages the application into a deployable artifact using predefined build instructions, often executed within automated pipelines after code validation steps complete."
  },
  {
    "task_number": 1,
    "step_number": 4,
    "step_description": "Push the Docker image to a container registry such as AWS ECR.",
    "current_solution": "Once the artifact is created, the pipeline authenticates with a remote registry and uploads the versioned artifact for storage and later retrieval during deployment."
  },
  {
    "task_number": 1,
    "step_number": 5,
    "step_description": "Deploy the updated container to a Kubernetes cluster using kubectl.",
    "current_solution": "Deployment configurations are applied to the target environment through automated or manual triggers, updating running services and orchestrating rollout of the new version across infrastructure."
  },
  {
    "task_number": 2,
    "step_number": 1,
    "step_description": "Collect system and application metrics using Prometheus exporters.",
    "current_solution": "Application components continuously emit performance and operational data, which is collected by monitoring agents configured across services and aggregated into a centralized system."
  },
  {
    "task_number": 2,
    "step_number": 2,
    "step_description": "Visualize metrics on Grafana dashboards for CPU, memory, and latency.",
    "current_solution": "Engineers create and maintain dashboards that query aggregated metrics, allowing teams to observe system behavior and track key performance indicators in real time."
  },
  {
    "task_number": 2,
    "step_number": 3,
    "step_description": "Configure alert rules in Prometheus Alertmanager for threshold breaches.",
    "current_solution": "Alert conditions are defined based on metric thresholds or patterns, with configurations maintained in version-controlled files and applied to the monitoring system for continuous evaluation."
  },
  {
    "task_number": 2,
    "step_number": 4,
    "step_description": "Receive and review alerts through Slack or PagerDuty notifications.",
    "current_solution": "When alert conditions are triggered, notification systems automatically route messages to designated channels, where engineers review and decide on further action."
  },
  {
    "task_number": 2,
    "step_number": 5,
    "step_description": "Investigate logs using Elasticsearch and Kibana to identify root causes.",
    "current_solution": "Engineers query centralized log storage systems to filter, correlate, and analyze logs from multiple services, often iterating through queries to trace the source of issues."
  }
]
"""

STAGE2B_USER = (
    "Identify the current solution for each step in this workflow:\n{workflow}"
)

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
