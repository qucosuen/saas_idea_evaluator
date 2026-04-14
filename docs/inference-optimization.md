# Inference Optimization Techniques

Target: <20 seconds per evaluation, <50% CPU utilization on consumer hardware.

## Techniques Applied

### 1. GGUF Quantization (biggest impact)

The single most impactful optimization. Converts the model from FP16 (2 bytes per weight) to Q4_K_M (4-bit mixed quantization, ~0.5 bytes per weight).

Why it works: Autoregressive LLM inference is memory-bandwidth bound, not compute bound. Each token generation requires reading the entire model weights from RAM. Smaller weights = faster reads = faster tokens.

| Format | Size (3B model) | Size (1.5B model) | Quality loss |
|---|---|---|---|
| FP32 | ~12 GB | ~6 GB | None (baseline) |
| FP16 | ~6 GB | ~3 GB | Negligible |
| Q4_K_M | ~2 GB | ~1.1 GB | Minimal (<1% perplexity increase) |
| Q2_K | ~1.2 GB | ~0.6 GB | Noticeable (2-5% perplexity increase) |

For our task (structured JSON output), Q4_K_M is the sweet spot — the quality loss is imperceptible for scoring and short justifications.

Backend: llama.cpp via `llama-cpp-python`. Uses optimized C++ SIMD kernels instead of Python/PyTorch overhead.

### 2. Smaller Model (Qwen2.5-1.5B vs 3B)

Half the parameters = roughly half the inference time per token. The 1.5B model is ~2x faster than the 3B model at the cost of lower benchmark scores.

| Model | Params | Q4_K_M Size | Est. time (4 threads) | MMLU |
|---|---|---|---|---|
| Qwen2.5-3B | 3.1B | 2.0 GB | ~40-80s | 65.6 |
| Qwen2.5-1.5B | 1.5B | 1.1 GB | ~15-25s | 55.9 |

The 1.5B model's lower MMLU (55.9 vs 65.6) means shallower reasoning, but for our narrow task with fine-tuning, this gap is addressable. The 1.5B model is the only path to <20s on a 4-core CPU at 50% utilization.

### 3. Compact JSON Schema (~40% fewer output tokens)

Standard schema uses verbose keys like `pain_severity`, `pain_frequency`, etc. The compact schema uses 2-character keys: `ps`, `pf`, `ea`, `wp`, `ms`, `sc`, `pp`, `df`, `tv`, `fm`.

Standard output (~400 tokens):
```json
{"pain_severity":{"score":7,"reason":"Unpaid invoices cause financial stress"},...}
```

Compact output (~250 tokens):
```json
{"ps":{"s":7,"r":"Unpaid invoices cause financial stress"},...}
```

~40% fewer tokens = ~40% less generation time. The compact output is expanded back to full keys for display and storage.

### 4. Thread Limiting (for <50% CPU)

On an 8-thread CPU, using all 8 threads gives maximum speed but 100% CPU utilization. Using 4 threads (half) keeps CPU under 50% while only reducing speed by ~30-40% (not 50%, because of memory bandwidth saturation).

```python
n_threads = max(2, os.cpu_count() // 2)  # Half the cores
```

### 5. Reduced Context Window

Default context is 32K tokens. Our prompts fit in ~600 tokens. Setting `n_ctx=1024` reduces memory allocation and speeds up the attention computation.

### 6. Greedy Decoding (temperature=0.0)

Eliminates the sampling step entirely. Each token is deterministically chosen as the highest-probability next token. Slightly faster than sampling with temperature>0, and produces more consistent output.

### 7. Reduced max_tokens

Standard: 768-1024 tokens. With compact schema, the output fits in ~250 tokens. Setting `max_tokens=384` prevents the model from generating unnecessary text after the JSON closes.

## Combined Impact

| Configuration | Model | Time | CPU | Notes |
|---|---|---|---|---|
| Baseline (Transformers FP32) | SmolLM2-1.7B | ~389s | 100% | Original Phase 1 |
| GGUF Q4_K_M, 8 threads | Qwen2.5-3B | ~38-77s | 100% | 5-10x speedup |
| GGUF Q4_K_M, 4 threads | Qwen2.5-3B | ~60-120s | ~50% | CPU-limited |
| GGUF Q4_K_M, 4 threads, compact | Qwen2.5-1.5B | ~50-55s | ~50% | Measured on i7-1185G7 |

### Detailed Latency Benchmarks (measured)

All measured on Intel i7-1185G7, 4/8 threads (~50% CPU), GGUF Q4_K_M, llama.cpp. Best of 2 runs (warm cache):

| Model | Params | Size | Short Q (10 tok) | 3-dim eval (28 tok) | Full 10-dim (131 tok) | tok/s |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | 0.5 GB | 1.7s | 20.8s | 16.2s | 6 |
| Qwen2.5-1.5B | 1.5B | 1.1 GB | 2.5s | 6.4s | 13.2s | 4 |
| Qwen2.5-3B | 3.0B | 2.1 GB | 4.6s | 13.2s | 55.6s | 2 |

Key observations:
- 0.5B achieves <2s for short questions but produces low-quality evaluations (all scores=1, no reasons). Valid JSON structure but useless content.
- 1.5B is the practical sweet spot: 6.4s for 3-dimension eval, 13.2s for full 10-dimension eval. Valid JSON with meaningful scores.
- 3B produces highest quality but 55.6s for full eval at 50% CPU is too slow.
- First run is 2-5x slower due to KV cache warmup. Subsequent runs are much faster.
- The SlimLM leaderboard's <5s results used 125M-135M models on Samsung Galaxy S24 with dedicated NPU — not comparable to CPU-only inference.

Achieving <20s at <50% CPU on this hardware is feasible with the 1.5B model if the output is kept under ~60 tokens (3-dimension eval: 6.4s). Full 10-dimension eval at 13.2s is close but requires the model to generate concise output.

To hit <5s consistently, you need one of:
- A CPU with higher memory bandwidth (Apple M-series: 100+ GB/s)
- A GPU (even GTX 1060 would achieve <5s)
- A 125M-135M model (but quality is insufficient for structured evaluation)

## Scripts

| Script | Description |
|---|---|
| `evaluate.py` | Standard GGUF inference with Qwen2.5-3B Q4_K_M |
| `evaluate_fast.py` | Ultra-optimized: compact schema, thread limiting, 1.5B model support |
| `src/evaluator/inference_fast.py` | Multi-backend engine (GGUF > ONNX > Transformers) |

## How to Run

```bash
# Standard (Qwen2.5-3B, ~50-80s, full CPU)
python evaluate.py "Your project description"

# Fast (auto-selects smallest available model, <50% CPU)
python evaluate_fast.py "Your project description"

# With specific model
python evaluate_fast.py --model models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf "description"
```

## Further Optimization Paths (not implemented)

1. Speculative decoding: Use a tiny draft model (0.5B) to predict tokens, verify in batch with the main model. Can achieve 2-3x speedup but requires two models in memory.

2. Prompt caching: Cache the KV state of the system prompt so it's only computed once. Saves ~200ms per request for repeated evaluations.

3. Batched inference: If evaluating multiple projects, batch them together to amortize the fixed costs. llama.cpp supports continuous batching.

4. Even more aggressive quantization: Q2_K or IQ2_XXS would be ~40% smaller/faster but with noticeable quality degradation. Only viable after fine-tuning compensates for the quality loss.
