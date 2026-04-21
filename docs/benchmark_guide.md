# Benchmark Dataset & Scoring System Guide

## 1. Overview

This document describes how to build a **comprehensive benchmark dataset and evaluation system** for a multi-stage reasoning pipeline:

Job → Tasks → Steps → Problems → Solutions → Evaluation

The goal is to evaluate not just correctness, but **reasoning quality, realism, and usefulness**.

---

## 2. Benchmark Objectives

### 2.1 Decomposition Quality
- Are tasks complete and relevant?
- Are steps logically ordered and non-overlapping?

### 2.2 Diagnostic Quality
- Are problems realistic and specific?
- Do they reflect real-world failure modes?

### 2.3 Solution Quality
- Are solutions correct and actionable?
- Are they diverse (not repetitive)?

### 2.4 Decision Quality
- Does the evaluator differentiate meaningfully?
- Are tradeoffs considered?

---

## 3. Dataset Design

### 3.1 Core Schema

Each benchmark sample should follow this structure:

{
  "job": "DevOps Engineer",
  "context": {
    "company_size": "startup | mid | enterprise",
    "infra_scale": "e.g. 50 services, 1000 nodes",
    "budget": "low | medium | high",
    "reliability_target": "e.g. 99.9%"
  },
  "ground_truth": {
    "tasks": [],
    "steps": [],
    "problems": [],
    "solutions": []
  }
}

---

### 3.2 Acceptable Answer Space (AAS)

Avoid single “correct answers”. Define acceptable alternatives:

{
  "step": "Monitor logs",
  "acceptable_solutions": [
    "ELK stack",
    "Datadog",
    "CloudWatch",
    "Loki + Grafana"
  ]
}

---

### 3.3 Negative Patterns

Define explicitly what bad outputs look like:

{
  "bad_patterns": [
    "manual-only solution",
    "no monitoring tools",
    "repeated 'automation script'",
    "no alerting system"
  ]
}

---

### 3.4 Job Coverage

Include multiple domains:
- DevOps Engineer
- Data Engineer
- Security Engineer
- Product Manager
- Customer Support Lead

---

### 3.5 Difficulty Levels

Each sample should include:

"difficulty": "easy | medium | hard"

---

## 4. Dataset Generation Process

### Step 1: Seed Generation
Use a strong model to generate tasks, steps, problems, solutions.

### Step 2: Review
Validate realism, completeness, and remove generic outputs.

### Step 3: Normalize
Ensure consistent structure and terminology.

### Step 4: Add Constraints
Inject realistic constraints like budget, infra, team size.

### Step 5: Add Adversarial Cases
Include edge cases (e.g. zero budget, legacy systems).

---

## 5. Scoring System

### 5.1 Stage-Level Scores (0–5)

#### Task Score
- coverage
- relevance
- non-redundancy

#### Step Score
- atomicity
- logical ordering
- actionability

#### Problem Score
- specificity (0–2)
- realism (0–2)
- causality (0–1)

#### Solution Score
- correctness (0–2)
- diversity (0–1)
- feasibility (0–1)
- novelty (0–1)

#### Evaluation Score
- variance (0–2)
- justification (0–2)
- usefulness (0–1)

---

### 5.2 System-Level Scores
- Consistency
- Redundancy
- Constraint adherence
- Practicality

---

## 6. Automatic Evaluation

### LLM-as-Judge
Use structured prompts to score outputs.

### Pairwise Comparison
Compare outputs for more reliable evaluation.

### Rule-Based Checks
- diversity detection
- repetition penalties

### Similarity Matching
Match outputs against acceptable solutions.

---

## 7. Final Score Aggregation

final_score =
  0.2 * task +
  0.2 * step +
  0.2 * problem +
  0.25 * solution +
  0.15 * evaluation

Adjust with bonuses and penalties.

---

## 8. Benchmark Pipeline

1. Load dataset  
2. Run pipeline  
3. Score outputs  
4. Store results  
5. Track trends  

---

## 9. Advanced Enhancements

- Gold vs model comparison  
- Latency vs quality tracking  
- Failure mode logging  

---

## 10. Success Criteria

System improves when:
- diversity increases
- realism improves
- evaluator shows variance
- constraint handling improves
- redundancy decreases

---

## 11. Key Principles

Reward:
- specificity
- realism
- diversity
- tradeoffs

Penalize:
- generic outputs
- repetition
- lack of constraints
- shallow reasoning

---

## 12. Recommended Scope

- 20–50 samples
- 5 job domains
- mixed difficulty

---

## 13. Next Steps

- Build dataset generator  
- Implement scoring pipeline  
- Run baseline evaluation  
- Iterate based on failures  
