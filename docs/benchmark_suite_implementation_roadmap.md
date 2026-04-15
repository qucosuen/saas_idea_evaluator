# Benchmark Suite Implementation Roadmap

A practical, step-by-step roadmap to design, build, and iterate on a benchmark suite for a 4-stage LLM pipeline:

1. Workflow Generator
2. Problem Extractor
3. Solution Mapper
4. Evaluator

---

# Phase 0: Define Goals

## Objectives

* Ensure outputs are structurally correct
* Measure task correctness per stage
* Track latency and efficiency
* Detect error propagation across stages

## Success Criteria

* ≥90% format correctness across stages
* Stable end-to-end pipeline execution
* Acceptable latency (<500ms total for local inference)

---

# Phase 1: Dataset Creation

## Step 1.1: Define Job List

Create a dataset of 20–30 jobs:

* Easy:

  * Data Entry Clerk
  * Customer Support Agent

* Medium:

  * Accountant
  * HR Manager

* Hard:

  * Supply Chain Manager
  * Construction Project Manager

## Step 1.2: Add Weak Ground Truth

Store dataset as JSON:

```json
{
  "job": "Accountant",
  "expected_patterns": ["data entry", "verification", "reporting"]
}
```

## Step 1.3: Store Dataset

* File: `dataset.json`
* Keep versioned for reproducibility

---

# Phase 2: Stage-Level Benchmarks

## Stage 1: Workflow Generator

### Output Structure

Given a job title, the model outputs a list of daily tasks. Each task contains a sequence of steps describing how to complete it.

### Metrics

#### 1. Format Check

* Correct task/step structure
* Each task has a title and numbered steps

#### 2. Concreteness Score

* Detect action verbs in steps
* Penalize vague terms

#### 3. Relevance Score

* Match against expected keywords for the job

#### 4. Coherence Score

* Steps within each task form a logical sequence
* Word overlap between consecutive steps
* Transition cues (e.g. "then", "after", "using")
* Temporal ordering (early steps gather, late steps output)

#### 5. Duplication Penalty

* Detect repeated task/step content

#### 6. Emptiness Penalty

* Detect tasks or steps with no meaningful content

---

## Stage 2: Problem Extractor

### Output Structure

For each step within each task from Stage 1, the model identifies:
- The current modern solution used for that step (e.g. software, tool, process)
- Any problems with that current solution (if problems exist)

### Metrics

#### 1. Coverage

* Every step from Stage 1 has a corresponding analysis

#### 2. Solution Specificity

* Current solutions reference real tools, software, or methods (not vague)

#### 3. Problem Quality

* Problems are specific and actionable, not generic complaints

#### 4. Alignment

* Solutions and problems are relevant to the step they reference

---

## Stage 3: Solution Mapper

### Metrics

#### 1. Valid Choices

Allowed:

* Automation Script
* OCR Extraction
* API Integration
* Dashboard
* Notification System

#### 2. Coverage

* 1 solution per step

#### 3. Fit Score

* Match problem type → appropriate solution

---

## Stage 4: Evaluator

### Metrics

#### 1. Format Check

* Must follow scoring format

#### 2. Value Range

* Scores between 1–5

#### 3. Consistency

* Logical scoring (e.g., simple solutions ≠ high complexity)

---

# Phase 3: End-to-End Benchmark

## Step 3.1: Pipeline Execution

Chain stages:

```python
workflow = stage1(job)
problems = stage2(workflow)
solutions = stage3(problems)
evaluation = stage4(solutions)
```

## Step 3.2: Track Degradation

Measure score drop across stages:

```
Stage 1 → score
Stage 2 → score
Stage 3 → score
Stage 4 → score
```

## Step 3.3: Usefulness Proxy

Check if:

* ≥2 steps have:

  * Feasibility ≥ 4
  * Impact ≥ 4

---

# Phase 4: Latency & Efficiency Tracking

## Metrics

* Per-stage latency (ms)
* Total latency
* Token usage
* Memory usage

## Example Output

```json
{
  "stage_1_ms": 120,
  "stage_2_ms": 80,
  "stage_3_ms": 70,
  "stage_4_ms": 90,
  "total_ms": 360
}
```

---

# Phase 5: Scoring System

## Weighted Score

```python
final_score = (
    0.25 * stage1_score +
    0.20 * stage2_score +
    0.20 * stage3_score +
    0.20 * stage4_score +
    0.15 * end_to_end_score
)
```

---

# Phase 6: Implementation Plan

## Step 6.1: Build Evaluation Functions

* Regex-based format checks
* Keyword matching
* Rule-based scoring

## Step 6.2: Build Runner Script

* Loop through dataset
* Execute all stages
* Store outputs

## Step 6.3: Logging

* Save raw outputs
* Save scores
* Save latency metrics

## Step 6.4: Reporting

* Aggregate metrics
* Output summary table

---

# Phase 7: Iteration & Improvement

## Step 7.1: Identify Weak Stages

* Lowest scoring stage
* Highest variance

## Step 7.2: Improve Prompts

* Add constraints
* Reduce ambiguity

## Step 7.3: Compare Models

* Run benchmark across models
* Track improvements

---

# Phase 8: Advanced Extensions (Optional)

## 1. Golden Dataset

* Human-written workflows
* Compare model outputs

## 2. Self-Consistency

* Run same input multiple times
* Measure variance

## 3. Adversarial Testing

* Vague jobs
* Complex roles

## 4. Leaderboard

* Rank models/prompts

---

# Final Notes

This benchmark suite enables:

* Objective evaluation of each pipeline stage
* Identification of bottlenecks
* Continuous improvement of prompts and models
* Data-driven decision making for system design

Start simple, measure everything, and iterate fast.
