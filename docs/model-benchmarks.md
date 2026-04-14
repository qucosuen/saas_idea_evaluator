# Model Candidate Benchmarks & Evaluation Metrics

This document compiles public benchmark scores for our candidate SLMs and our own empirical results from Phase 1 testing. It explains what each metric measures and why it matters for our project evaluation task.

---

## 1. Public Benchmark Scores (Instruct Models)

Sources: [Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5-llm/), [Phi-3.5-mini HuggingFace card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), [SmolLM2 comparisons](https://markaicode.com/vs/smollm2-17b-vs-phi-2/). Content was rephrased for compliance with licensing restrictions.

### 1.1 General Reasoning & Knowledge

| Benchmark | SmolLM2-1.7B | Qwen2.5-3B | Phi-3.5-mini (3.8B) | Llama-3.2-3B | Gemma-2-2.6B |
|---|---|---|---|---|---|
| MMLU (5-shot) | 56.7 | 65.6 | 69.0 | 60.9 | 52.2 |
| MMLU-Pro | ~28* | 34.6 | 47.4 | 28.5 | 23.0 |
| BBH (few-shot) | ~36* | 56.3 | 69.0 | 45.1 | 41.9 |
| HellaSwag | ~67* | 74.6 | 69.4 | 67.9 | 74.6 |
| ARC-Challenge | ~44* | 56.5 | 84.6 | 54.7 | 55.7 |
| TruthfulQA | ~46* | 48.9 | 64.0 | 46.6 | 36.2 |

*SmolLM2-1.7B values marked with ~ are interpolated from available comparisons and may not be exact.

### 1.2 Math & Code

| Benchmark | SmolLM2-1.7B | Qwen2.5-3B | Phi-3.5-mini (3.8B) | Llama-3.2-3B | Gemma-2-2.6B |
|---|---|---|---|---|---|
| GSM8K (CoT) | 46.8 | 86.7 | 86.2 | 68.5 | 30.3 |
| MATH | ~21* | 65.9 | 48.5 | 35.0 | 18.3 |
| HumanEval | 31.4 | 74.4 | 62.8 | 37.2 | 19.5 |
| MBPP | ~47* | 72.7 | 69.6 | 60.2 | 42.1 |

### 1.3 Instruction Following

| Benchmark | SmolLM2-1.7B | Qwen2.5-3B | Phi-3.5-mini (3.8B) | Llama-3.2-3B |
|---|---|---|---|---|
| IFEval (strict-prompt) | ~28* | 58.2 | ~52* | 51.0 |

---

## 2. What Each Metric Means (and Why It Matters for Our Task)

### MMLU — Massive Multitask Language Understanding
- **What:** 57 subjects (STEM, humanities, social sciences, etc.), multiple-choice questions.
- **Why it matters for us:** Measures breadth of world knowledge. Our evaluator needs to reason about diverse industries (fintech, healthtech, hardware, etc.). Higher MMLU = better at understanding what a project description is actually about.
- **Takeaway:** Qwen2.5-3B (65.6) and Phi-3.5-mini (69.0) are significantly ahead of SmolLM2-1.7B (56.7). This gap directly impacts evaluation quality for domain-specific projects.

### MMLU-Pro — Harder MMLU with 10 answer choices
- **What:** A more challenging version of MMLU with 10 options instead of 4, reducing lucky guesses.
- **Why it matters for us:** Tests deeper reasoning under ambiguity — exactly what happens when evaluating whether a project idea is viable vs. not.
- **Takeaway:** Phi-3.5-mini (47.4) dominates here. Qwen2.5-3B (34.6) is decent. SmolLM2-1.7B (~28) struggles.

### BBH — BIG-Bench Hard
- **What:** 23 challenging tasks from BIG-Bench that prior LLMs failed at, testing multi-step reasoning.
- **Why it matters for us:** Project evaluation requires chaining multiple reasoning steps (assess pain → estimate market → judge defensibility → synthesize). BBH measures exactly this capability.
- **Takeaway:** Phi-3.5-mini (69.0) and Qwen2.5-3B (56.3) are strong. SmolLM2-1.7B (~36) is weak here — explains why its evaluations tend to be shallow.

### HellaSwag — Commonsense Reasoning
- **What:** Sentence completion requiring commonsense understanding of everyday situations.
- **Why it matters for us:** Evaluating "time to value" or "pain frequency" requires commonsense about how people actually use products. A model that doesn't understand everyday scenarios will produce nonsensical justifications.
- **Takeaway:** Qwen2.5-3B and Gemma-2 tie at 74.6, slightly ahead of Phi-3.5-mini (69.4).

### GSM8K — Grade School Math
- **What:** 8,500 grade-school math word problems requiring multi-step arithmetic.
- **Why it matters for us:** Our model needs to compute weighted averages for the overall score and reason about numbers (market sizes, cost savings percentages). Models that can't do basic math will produce inconsistent overall scores.
- **Takeaway:** Qwen2.5-3B (86.7) and Phi-3.5-mini (86.2) are nearly identical and far ahead of SmolLM2-1.7B (46.8).

### MATH — Competition-Level Mathematics
- **What:** 12,500 problems from math competitions (algebra, geometry, number theory, etc.).
- **Why it matters for us:** Less directly relevant than GSM8K, but indicates general quantitative reasoning ability.
- **Takeaway:** Qwen2.5-3B (65.9) leads significantly. Phi-3.5-mini (48.5) is decent. Others are weak.

### HumanEval / MBPP — Code Generation
- **What:** Generate correct Python functions from docstrings/descriptions.
- **Why it matters for us:** Not directly relevant to project evaluation, but indicates the model's ability to produce structured, syntactically correct output (like JSON). Models good at code tend to be good at structured output.
- **Takeaway:** Qwen2.5-3B (74.4 HumanEval) leads. Phi-3.5-mini (62.8) is solid. SmolLM2-1.7B (31.4) is weak — correlates with its struggles producing valid JSON without few-shot examples.

### IFEval — Instruction Following Evaluation
- **What:** Tests whether the model follows specific formatting instructions (e.g., "respond in JSON", "use bullet points", "write exactly 3 sentences").
- **Why it matters for us:** This is arguably the MOST important benchmark for our task. We need the model to output valid JSON with exactly 10 dimensions, each with a score and reason. IFEval directly measures this capability.
- **Takeaway:** Qwen2.5-3B (58.2) leads among the small models. This is the single strongest argument for choosing Qwen2.5-3B as our base model.

### TruthfulQA — Factual Accuracy
- **What:** Tests whether models generate truthful answers vs. common misconceptions.
- **Why it matters for us:** We want the model to give honest assessments, not optimistic hallucinations. A model that scores low here may inflate project scores.
- **Takeaway:** Phi-3.5-mini (64.0) leads significantly. This may explain why SmolLM2-1.7B tends to be overly optimistic in its evaluations.

### ARC-Challenge — Science Reasoning
- **What:** Grade-school science questions that require reasoning, not just recall.
- **Why it matters for us:** Moderate relevance. Tests the model's ability to reason about cause-and-effect, which is useful for evaluating scalability and defensibility.
- **Takeaway:** Phi-3.5-mini (84.6) dominates. Others are clustered around 44–57.

---

## 3. Our Empirical Results (Phase 1 Baseline)

### 3.1 SmolLM2-360M-Instruct — FAILED
- Could not produce structured JSON output at all
- Generated free-text summaries instead of the requested format
- **Verdict:** Too small for structured evaluation tasks. Not viable.

### 3.2 SmolLM2-1.7B-Instruct — PARTIAL SUCCESS

Tested on 2 out of 5 test cases (CPU timeout after ~13 minutes):

| Metric | Result |
|---|---|
| JSON validity rate | 100% (2/2) |
| Schema compliance | 100% (2/2) |
| Avg generation time (CPU, FP32) | 388.7 seconds |
| Overall score range | 5.9 – 8.1 |

**Per-test-case results:**

| Test Case | Expected | Overall Score | Verdict |
|---|---|---|---|
| Freelancer invoice tracker (SaaS) | Good idea | 8.1 | Slightly inflated but directionally correct |
| Vintage typewriter social network (Niche) | Niche/weak idea | 5.9 | Too generous — pain_severity=7 for a hobby social network is unrealistic |

**Observed issues:**
- Score inflation: The model is too optimistic. A niche hobby social network scoring 5.9 overall and 7/10 on pain severity suggests the model lacks calibration for weak ideas.
- Shallow justifications: Reasons are generic ("Growing community of passionate collectors worldwide" for market_size=7 is vague and arguably wrong).
- Slow inference: ~6.5 minutes per evaluation on CPU makes it impractical for batch operations.

---

## 4. Composite Ranking for Our Task

Weighting the benchmarks by relevance to our project evaluation task:

| Weight | Benchmark | Rationale |
|---|---|---|
| High | IFEval | Must output structured JSON reliably |
| High | MMLU | Must understand diverse industries |
| High | BBH | Must chain multi-step reasoning |
| Medium | GSM8K | Must compute scores and averages |
| Medium | TruthfulQA | Must give honest, calibrated assessments |
| Medium | HumanEval | Proxy for structured output capability |
| Low | MATH | Overkill for our needs |
| Low | ARC-Challenge | Tangentially relevant |

### Weighted Composite Score (our estimate)

| Model | Params | IFEval | MMLU | BBH | GSM8K | TruthfulQA | HumanEval | Composite* | RAM (Q4) |
|---|---|---|---|---|---|---|---|---|---|
| Phi-3.5-mini | 3.8B | ~52 | 69.0 | 69.0 | 86.2 | 64.0 | 62.8 | ~67 | ~2.5 GB |
| Qwen2.5-3B | 3.1B | 58.2 | 65.6 | 56.3 | 86.7 | 48.9 | 74.4 | ~65 | ~2.0 GB |
| Llama-3.2-3B | 3.2B | 51.0 | 60.9 | 45.1 | 68.5 | 46.6 | 37.2 | ~52 | ~2.0 GB |
| SmolLM2-1.7B | 1.7B | ~28 | 56.7 | ~36 | 46.8 | ~46 | 31.4 | ~41 | ~1.2 GB |
| Gemma-2-2.6B | 2.6B | — | 52.2 | 41.9 | 30.3 | 36.2 | 19.5 | ~36 | ~1.8 GB |

*Composite = weighted average: IFEval×0.25 + MMLU×0.20 + BBH×0.20 + GSM8K×0.10 + TruthfulQA×0.15 + HumanEval×0.10

---

## 5. Recommendations

1. **Primary choice: Qwen2.5-3B-Instruct** — Best IFEval score (structured output), strong math, excellent code generation (proxy for JSON output). Slightly smaller than Phi-3.5-mini. Best overall balance for our specific task.

2. **Alternative: Phi-3.5-mini-instruct** — Highest MMLU and BBH (deepest reasoning), best TruthfulQA (most calibrated/honest). Slightly larger at 3.8B. Better if evaluation quality matters more than size.

3. **Budget option: SmolLM2-1.7B-Instruct** — Works for proof-of-concept. Produces valid JSON with few-shot prompting. But score calibration is poor and reasoning is shallow. Fine-tuning should help significantly given the low baseline.

4. **Not recommended: Gemma-2-2.6B, Llama-3.2-3B** — Gemma-2 has weak math and code scores. Llama-3.2-3B is mediocre across the board for this size class.

---

## 6. Key Insight: The Fine-Tuning Opportunity

The gap between SmolLM2-1.7B (composite ~41) and Qwen2.5-3B (composite ~65) is large on general benchmarks. But for our narrow, well-defined task (structured JSON evaluation of project ideas), fine-tuning can close much of this gap. The model doesn't need to know everything — it just needs to:

1. Output valid JSON (learnable with ~50 examples)
2. Score 10 dimensions consistently (learnable with ~200 examples)
3. Write calibrated justifications (learnable with ~500+ examples)

This is why the approach doc recommends starting with the smallest viable model and investing in data quality over model size.
