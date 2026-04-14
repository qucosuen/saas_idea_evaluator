# SLM Project Evaluator

A local-first system using small language models (≤3B params) to evaluate startup ideas and discover automation opportunities from real-world jobs.

## Champion Model

Qwen2.5-1.5B-Instruct (Q4_K_M GGUF) — selected via QLE metric (Quality × Latency Penalty = 49.7).

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Evaluate a startup idea
python evaluate.py "An app that helps freelancers track unpaid invoices"

# Run the 4-stage automation pipeline
python -m src.pipeline.cli "Accountant"

# Start the dashboard
python ui/server.py  # → http://localhost:3000

# Start the REST API
python -m src.pipeline.api  # → http://localhost:8000
```

---

## Components

### 1. Project Evaluator (`evaluate.py`, `evaluate_fast.py`)

Evaluates startup/project ideas across 10 business-viability dimensions (pain severity, frequency, market size, etc.), each scored 1-10 with justification.

```bash
# Standard evaluation (Qwen2.5-3B, ~50-80s on CPU)
python evaluate.py "Your project description"

# Fast evaluation (compact schema, thread-limited, prefers 1.5B model)
python evaluate_fast.py "Your project description"

# Interactive mode
python evaluate.py
python evaluate_fast.py
```

### 2. Automation Pipeline (`src/pipeline/`)

4-stage pipeline: Job → Workflow → Problems → Solutions → Evaluation.

```bash
# CLI (single job)
python -m src.pipeline.cli "Accountant"

# CLI with JSON output and caching
python -m src.pipeline.cli --json --cache "Data Entry Clerk"

# REST API
python -m src.pipeline.api
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"job": "Accountant"}'
```

Configuration via `.env`:
```
CPU_PERCENT=50   # Use 50% of available CPU threads (default)
```

### 3. Benchmark Suite (`src/benchmark/`)

Evaluates the pipeline across 30 jobs with per-stage metrics.

```bash
# Run full benchmark (all 30 jobs)
python -m src.benchmark.runner

# Sample 2 jobs per difficulty level
python -m src.benchmark.runner --sample 2

# With MLflow logging
python -m src.benchmark.runner --sample 2 --mlflow

# Specific model
python -m src.benchmark.runner --model models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf
```

### 4. Champion Selection (`scripts/champion_selection.py`)

Selects the best model using the QLE (Quality-Latency Efficiency) metric.

```bash
python scripts/champion_selection.py
# Tests 0.5B, 1.5B, 3B models on 3 test cases
# Outputs: results/champion_selection.json
```

### 5. Training Data (`data/`, `src/data/`)

501 training examples for fine-tuning, plus 113 scraped Indie Hackers ideas.

```bash
# Generate synthetic training data (requires OpenAI API key)
OPENAI_API_KEY=sk-... python -m src.data.generate_training_data --teacher openai

# Upload to PostgreSQL + Feast feature store
python -m src.data.upload_to_postgres

# Scrape Indie Hackers ideas
python scripts/scrape_indiehackers.py
```

### 6. Fine-Tuning (`src/training/`)

QLoRA fine-tuning pipeline (requires GPU).

```bash
python -m src.training.finetune_qlora \
  --data data/training_data_full.json \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 4 --epochs 3 --lora-rank 32
```

### 7. MLflow Tracking (`src/tracking/`)

Log model benchmarks and experiment results.

```bash
# Log all model candidates to MLflow
python -m src.tracking.report_to_mlflow
# View at http://mlflow.platform.local
```

### 8. Dashboard (`ui/`)

Web UI visualizing all project data — model comparison, pipeline benchmarks, training data distribution, Indie Hackers ideas.

```bash
python ui/server.py
# Open http://localhost:3000
```

### 9. Remote Deployment (`scripts/deploy_remote.sh`)

Deploy and run on a remote Windows machine via SSH.

```bash
# One-time: set up SSH key auth
ssh-copy-id user@192.168.1.2

# Deploy everything
./scripts/deploy_remote.sh user@192.168.1.2

# Or run individual scripts remotely
ssh user@192.168.1.2 "cd C:\workspace\slm_evaluator && .venv\Scripts\python.exe scripts\remote_benchmark.py"
```

---

## Project Structure

```
├── src/
│   ├── evaluator/          # Core inference engine
│   │   ├── inference.py    # HuggingFace + GGUF model loading, JSON parsing
│   │   ├── inference_fast.py  # Multi-backend engine (GGUF > ONNX > Transformers)
│   │   └── baseline.py     # Few-shot baseline evaluation
│   ├── pipeline/           # 4-stage automation pipeline
│   │   ├── cli.py          # Command-line interface
│   │   ├── api.py          # REST API server
│   │   ├── runner.py       # Pipeline execution with retry + caching
│   │   ├── validators.py   # JSON schema validation per stage
│   │   ├── prompts.py      # Stage prompts with few-shot examples
│   │   ├── schemas.py      # Dataclass schemas
│   │   ├── model.py        # Model loader utility
│   │   └── cpu_config.py   # CPU thread config from .env
│   ├── benchmark/          # Pipeline benchmark suite
│   │   ├── runner.py       # Benchmark runner with reporting
│   │   ├── scorer.py       # Weighted scoring system
│   │   ├── metrics.py      # Per-stage metric functions
│   │   ├── pipeline.py     # Benchmark pipeline runner
│   │   ├── prompts.py      # Benchmark stage prompts
│   │   └── dataset.json    # 30 jobs (easy/medium/hard)
│   ├── data/               # Data pipelines
│   │   ├── generate_training_data.py  # Synthetic data generation
│   │   └── upload_to_postgres.py      # PostgreSQL + Feast upload
│   ├── training/           # Model training
│   │   ├── finetune_qlora.py  # QLoRA fine-tuning
│   │   └── evaluate.py     # Baseline vs fine-tuned comparison
│   └── tracking/           # Experiment tracking
│       └── report_to_mlflow.py
├── data/
│   ├── training_data_full.json   # 501 training examples
│   ├── training_data.json        # 20 hand-crafted seed examples
│   ├── seed_descriptions.json    # 40 project description seeds
│   ├── test_descriptions.json    # 5 evaluation test cases
│   └── indiehackers_ideas.json   # 113 scraped startup ideas
├── models/gguf/            # GGUF quantized models (downloaded on first run)
├── results/                # Evaluation results and benchmarks
├── feast/feature_repo/     # Feast feature store definitions
├── ui/                     # Web dashboard
│   ├── server.py           # Dashboard HTTP server
│   └── app.html            # Single-page dashboard app
├── scripts/                # Utility and deployment scripts
├── tools/                  # One-off data generation utilities
├── docs/                   # Documentation
├── evaluate.py             # Standalone evaluator (GGUF)
├── evaluate_fast.py        # Optimized evaluator (compact schema)
├── requirements.txt        # Python dependencies
├── .env.remote             # Remote machine config (CPU_PERCENT)
└── .gitignore
```

---

## Documentation

| Document | Description |
|---|---|
| [Approach](docs/slm-project-evaluator-approach.md) | Full design: dimensions, model candidates, fine-tuning approaches, roadmap |
| [Model Benchmarks](docs/model-benchmarks.md) | Public benchmark scores, composite ranking, selection rationale |
| [Data Requirements](docs/data-requirements.md) | Training data sizing guide: 50/200/500/1500/3000 example tiers |
| [Inference Optimization](docs/inference-optimization.md) | GGUF quantization, thread limiting, compact schema, measured latencies |
| [Implementation Status](docs/implementation-status.md) | Progress log, champion model decision (QLE metric), infrastructure status |
| [Benchmark Roadmap](docs/benchmark_suite_implementation_roadmap.md) | 4-stage pipeline benchmark design |
| [Pipeline Roadmap](docs/local_first_llm_pipeline_project_roadmap.md) | Local-first LLM pipeline project plan |

---

## Infrastructure

| Component | Address | Purpose |
|---|---|---|
| PostgreSQL | localhost:5432 (admin/admin) | Offline feature store, training data, Indie Hackers ideas |
| Feast | feast/feature_repo/ | Feature definitions, online serving via SQLite |
| MLflow | http://mlflow.platform.local | Experiment tracking, model comparison, benchmark results |
| Dashboard | http://localhost:3000 | Web UI for visualization |
| Pipeline API | http://localhost:8000 | REST API for pipeline execution |

---

## Key Results

| Metric | Value |
|---|---|
| Champion model | Qwen2.5-1.5B-Instruct Q4_K_M |
| Champion QLE score | 49.7 |
| Avg latency (local, 4 threads) | 11.7s per evaluation |
| Avg latency (remote, 16 threads) | 2.8s per evaluation |
| Pipeline latency (remote, 16 threads) | 13.7s for 4 stages |
| Pipeline latency (remote, 20% CPU) | 15.0s for 4 stages |
| Training examples | 501 (calibrated score distribution) |
| Indie Hackers ideas scraped | 113 |
| Benchmark jobs | 30 (10 easy, 10 medium, 10 hard) |
| Models evaluated | 6 (0.5B, 1.5B, 1.7B, 2.6B, 3B, 3.8B) |

---

## Remote Machine (Windows)

Workspace: `C:\workspace\slm_evaluator\` on 192.168.1.2

- Python 3.12, 16 CPU threads, RTX 3050 Ti (GPU unusable — driver 462.62 too old)
- To enable GPU: update NVIDIA driver by running `C:\workspace\nvidia_driver.exe` as Administrator, then reboot
- After driver update, reinstall llama-cpp-python with CUDA: `pip install llama-cpp-python --force-reinstall --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122`
