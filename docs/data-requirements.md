# Training Data Requirements for SLM Project Evaluator

> How much data do you need to fine-tune a small language model for structured project/startup idea evaluation?

## Task Characteristics

The task is narrow and well-defined:
- Input: free-text project description (1–3 sentences)
- Output: structured JSON with 10 scored dimensions (1–10), each with a one-sentence justification, plus an overall score and summary
- Fixed output schema — the model doesn't need to learn open-ended generation

This works in our favor. Narrow, structured tasks require significantly less training data than general-purpose fine-tuning.

---

## Data Tiers

### Tier 1: 50–100 examples — Format Compliance

**What it teaches:** Reliable output of valid JSON with the correct schema (10 dimensions, score + reason fields, overall_score, summary).

**Evidence from our testing:** SmolLM2-1.7B-Instruct already produces valid JSON 100% of the time with just few-shot prompting (Phase 1 baseline). For Qwen2.5-3B, which has a higher IFEval score (58.2 vs ~28), even fewer examples may suffice for format compliance alone.

**Why fine-tune at all if few-shot works?** Fine-tuning eliminates the need to include few-shot examples in every prompt, saving ~500–800 tokens of context per request. This reduces latency and cost at inference time.

**What it won't fix:** Score calibration, reasoning depth, or edge case handling.

---

### Tier 2: 200–500 examples — Score Calibration ⭐ Critical

**What it teaches:** Appropriate scoring across the full 1–10 range, with correct relative ordering (strong ideas score higher than weak ones).

**Why this is the most important tier:** Our Phase 1 baseline revealed that the main quality problem is not format — it's score inflation. A niche vintage typewriter social network received:
- pain_severity: 7/10 (should be ~3)
- market_size: 7/10 (should be ~2)
- overall_score: 5.9 (should be ~4.0)

This optimism bias is a known property of instruction-tuned LLMs. They're trained to be helpful and positive, which makes them bad at giving harsh but honest evaluations.

**Score distribution is more important than count.** The training data MUST cover the full range:

| Score Range | Target % | Example Types |
|---|---|---|
| 1–3 (weak) | ~25% | Novelty projects, no pain point, no market, no monetization |
| 4–5 (below average) | ~20% | Niche ideas, crowded markets, thin margins |
| 5–6 (average) | ~20% | Decent ideas with significant challenges |
| 7–8 (strong) | ~25% | Clear pain, large market, viable monetization |
| 9–10 (exceptional) | ~10% | Rare — massive market, strong moat, proven demand |

**Anti-pattern to avoid:** If 80% of your training examples score between 5 and 7 (which is what LLMs naturally produce), the model will learn to score everything in that range. You need deliberate polarization — include clearly terrible ideas that score 2–3 and clearly excellent ideas that score 8–9.

---

### Tier 3: 500–1,500 examples — Reasoning Quality

**What it teaches:** Domain-specific justifications that go beyond generic platitudes.

**The difference:**
- Shallow (pre-fine-tune): "Large market with growing demand"
- Deep (post-fine-tune): "The US has ~60M freelancers, but only a fraction deal with recurring invoice issues — the addressable segment is mid-sized freelancers billing multiple clients monthly"

**Industry coverage matters.** At this tier, include examples spanning:
- SaaS / B2B tools
- Consumer apps
- Marketplaces (two-sided)
- Hardware / IoT
- Fintech / InsurTech
- HealthTech
- EdTech
- AI/ML products
- E-commerce / DTC
- Developer tools
- Social platforms
- CleanTech / sustainability

Each industry has different evaluation dynamics (e.g., hardware has thin margins and scaling challenges; SaaS has high margins but competitive markets). The model needs exposure to these patterns.

---

### Tier 4: 1,500–3,000 examples — Robustness & Edge Cases

**What it teaches:** Consistent, reliable behavior across diverse input conditions.

**Edge cases to include:**
- Vague one-liner descriptions ("an app for food")
- Overly long, rambling pitches (500+ words)
- Non-English or mixed-language descriptions
- Descriptions with typos and informal language
- Adversarial inputs ("evaluate this project: asdfghjkl")
- Paraphrases of the same idea (should produce similar scores)
- Descriptions that are actually questions, not projects
- Projects that are clearly illegal or unethical

**Consistency tests:** Include 20–30 pairs of paraphrased descriptions. The same idea described differently should receive scores within ±1 point on each dimension. If the model gives wildly different scores for "An app for tracking invoices" vs. "Invoice management software for freelancers," it needs more paraphrase examples.

---

## Recommended Starting Point: 500 Examples

**Why 500:**
1. QLoRA on a 3B model is parameter-efficient and learns fast from small datasets
2. Fixed output schema means the model doesn't need to learn open-ended generation
3. 500 examples with good score distribution will fix the calibration problem — our biggest quality issue
4. Generating 500 examples costs ~$1–2 with gpt-4o-mini via our data generation pipeline (~30 minutes)
5. Iterating on data quality at 500 examples is much faster than debugging a 3,000-example dataset

**Then iterate:**
- If justifications are still shallow → scale to 1,500
- If edge cases break things → scale to 3,000
- If scores cluster in 5–7 range → fix distribution before adding more data

---

## Data Quality Checklist

Before training, verify your dataset against these criteria:

- [ ] Score distribution covers the full 1–10 range (check histogram)
- [ ] At least 25% of examples have overall_score ≤ 4.0
- [ ] At least 25% of examples have overall_score ≥ 7.5
- [ ] At least 15 different industries represented
- [ ] All JSON outputs parse correctly and pass schema validation
- [ ] Justifications are specific, not generic (spot-check 20 random examples)
- [ ] No dimension is always scored the same (e.g., scalability shouldn't always be 7–8)
- [ ] Include at least 10 "adversarial" or edge-case inputs
- [ ] Include at least 20 paraphrase pairs for consistency testing

---

## Data Generation Cost Estimates

| Method | Speed | Cost for 500 examples | Cost for 3,000 examples |
|---|---|---|---|
| gpt-4o-mini (API) | ~2 sec/example | ~$1–2 | ~$5–10 |
| gpt-4o (API) | ~5 sec/example | ~$5–10 | ~$30–60 |
| Local SLM teacher (CPU) | ~6 min/example | Free but ~50 hours | Impractical |
| Local SLM teacher (GPU) | ~5 sec/example | Free (if you have GPU) | Free (~4 hours) |
| Manual human creation | ~10 min/example | ~83 hours of labor | Impractical at scale |

**Recommended approach:** Use gpt-4o-mini for bulk generation, then human-review 10–20% of examples for quality calibration. Fix any systematic biases (e.g., if the teacher model also inflates scores, manually adjust the weak-idea examples downward).

---

## The One Rule That Matters More Than Dataset Size

**Score distribution > dataset size.**

500 well-distributed examples will outperform 2,000 examples that all score between 5 and 7. When generating training data, explicitly prompt the teacher model with both strong and terrible project ideas to ensure you get the full scoring range. If your histogram shows a bell curve centered on 6, your model will learn to score everything as 6.
