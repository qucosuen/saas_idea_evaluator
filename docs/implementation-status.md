# SLM Project Evaluator — Implementation Status

## Champion Model

Qwen2.5-1.5B-Instruct (Q4_K_M GGUF) — QLE = 49.7

Selected using QLE = Quality_Score × 1/(1 + latency/30s). The 1.5B model has nearly identical quality to the 3B (69.1 vs 69.9) but is 2.8x faster (11.7s vs 32.9s). Full selection data in `results/champion_selection.json`.

## What's Built

| Component | Status | How to Run |
|---|---|---|
| Project evaluator | ✅ Working | `python evaluate.py "description"` |
| Fast evaluator | ✅ Working | `python evaluate_fast.py "description"` |
| 4-stage pipeline CLI | ✅ Working | `python -m src.pipeline.cli "Accountant"` |
| Pipeline REST API | ✅ Working | `python -m src.pipeline.api` |
| Benchmark suite | ✅ Working | `python -m src.benchmark.runner --sample 2` |
| Champion selection | ✅ Complete | `python scripts/champion_selection.py` |
| Training data (501 examples) | ✅ Complete | `data/training_data_full.json` |
| Indie Hackers scraper | ✅ Complete | `python scripts/scrape_indiehackers.py` |
| PostgreSQL storage | ✅ Active | 4 tables + 1 view in feast_offline |
| Feast feature store | ✅ Active | 3 feature views, materialized |
| MLflow tracking | ✅ Active | http://mlflow.platform.local |
| Web dashboard | ✅ Working | `python ui/server.py` → localhost:3000 |
| Remote deployment | ✅ Working | `./scripts/deploy_remote.sh user@host` |
| HF Inference API | ✅ Working | `python -m src.pipeline.cli --remote "job"` |
| QLoRA fine-tuning | ⚠️ Pipeline ready | Needs GPU to execute |
| GPU inference (remote) | ❌ Blocked | NVIDIA driver 462.62 too old |

## Measured Latencies

| Machine | CPU% | Threads | Single Eval | Full Pipeline |
|---|---|---|---|---|
| Linux i7-1185G7 | 50% | 4/8 | 11.7s | ~160s |
| Windows i7 (remote) | 100% | 16/16 | 2.8s | 13.7s |
| Windows i7 (remote) | 50% | 8/16 | — | 12.9s |
| Windows i7 (remote) | 20% | 3/16 | — | 15.0s |
| HF Inference API | — | — | ~1-3s | ~5-15s (est.) |

## Infrastructure

- PostgreSQL: localhost:5432/feast_offline — training data (501 rows), features (501 rows), Indie Hackers ideas (113 rows)
- Feast: 3 feature views (project_evaluation_scores, project_evaluation_training_data, indiehackers_ideas)
- MLflow: experiment `SLM-Project-Evaluator-Candidates` with model benchmarks, pipeline benchmarks, latency measurements
- GGUF models cached: 0.5B (0.5GB), 1.5B (1.1GB), 3B (2.1GB)

## What's Not Done

- QLoRA fine-tuning (needs GPU)
- DPO alignment (needs fine-tuned model first)
- GPU inference on remote machine (needs driver update)
- GGUF model quantization from fine-tuned weights

## Stage 1 Spec Change

Stage 1 output has been redesigned. Previously, the model output a flat list of 5 workflow steps. Now, given a job title, the model outputs a list of daily tasks, where each task contains a sequence of steps describing how to complete it.

Old format:
```
Step 1: Collect data
Step 2: Validate data
...
```

New format:
```
Task 1: Data Entry
  Step 1: Open the data entry form
  Step 2: Input data from source documents
  Step 3: Verify entered data against source
  Step 4: Submit the completed form
Task 2: Data Validation
  Step 1: Retrieve entries from database
  Step 2: Run validation rules
  ...
```

This change requires updates to: prompts, validators, schemas, benchmark metrics, and the dashboard UI.

## Stage 1 Benchmark Results (Champion Model)

Benchmark run across all 30 jobs. Results in `results/benchmark/stage1_benchmark.json`, logged to MLflow.

| Metric | Mean | Std |
|---|---|---|
| Format Score | 0.931 | 0.403 |
| Concreteness | 0.647 | 0.399 |
| Relevance | 0.193 | 0.220 |
| Duplication Penalty | 0.213 | 0.275 |
| Stage Score (with dup penalty) | 0.510 | 0.246 |

Duplication rate: 70% of jobs affected. Severity breakdown:
- None (0.00 penalty): 9/30 jobs (30%)
- Mild — "Step 1: Step 1: content" (0.15 penalty): 17/30 jobs (57%)
- Severe — infinite "Step 1: Step 1: Step 1:..." loop (0.80 penalty): 4/30 jobs (13%)

Top 5 worst duplication jobs (all severe, 49 duplicates each): Library Assistant, Janitor, Sales Representative, Real Estate Agent, University Registrar.
