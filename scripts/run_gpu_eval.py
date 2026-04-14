"""
GPU-enabled evaluation script for the remote Windows machine.
Runs the champion selection benchmark and pipeline benchmark using GPU acceleration.

This script auto-detects CUDA and uses GPU if available, falling back to CPU.
Run: python scripts/run_gpu_eval.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment():
    """Check GPU availability and print system info."""
    print("=" * 60)
    print("  SLM Evaluator — GPU Evaluation")
    print("=" * 60)

    # Check CUDA
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem
            print(f"  GPU: {gpu_name} ({gpu_mem / 1e9:.1f} GB)")
        else:
            print("  GPU: Not available (using CPU)")
    except ImportError:
        has_cuda = False
        print("  GPU: PyTorch not installed")

    print(f"  Python: {sys.version.split()[0]}")
    print(f"  CWD: {os.getcwd()}")
    print("=" * 60)
    return has_cuda


def download_model():
    """Download the champion GGUF model if not present."""
    models_dir = Path("models/gguf")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    if model_file.exists():
        print(f"Model cached: {model_file}")
        return model_file

    from huggingface_hub import hf_hub_download
    print("Downloading Qwen2.5-1.5B-Instruct Q4_K_M...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        local_dir=str(models_dir),
    )
    print(f"Downloaded to {model_file}")
    return model_file


def run_benchmark(model_path, has_cuda):
    """Run the pipeline benchmark."""
    from llama_cpp import Llama

    n_gpu = -1 if has_cuda else 0
    threads = os.cpu_count() or 4

    print(f"\nLoading model: {model_path.name}")
    print(f"  GPU layers: {'all' if has_cuda else 'none'}")
    print(f"  CPU threads: {threads}")

    model = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=threads,
        n_gpu_layers=n_gpu,
        verbose=False,
    )

    # Load benchmark dataset
    ds_path = Path("src/benchmark/dataset.json")
    if not ds_path.exists():
        print("Benchmark dataset not found. Running simple eval instead.")
        return run_simple_eval(model)

    with open(ds_path) as f:
        dataset = json.load(f)

    # Sample 2 per difficulty
    import random
    random.seed(42)
    sampled = []
    for diff in ["easy", "medium", "hard"]:
        pool = [j for j in dataset if j["difficulty"] == diff]
        sampled.extend(random.sample(pool, min(2, len(pool))))

    print(f"\nRunning benchmark: {len(sampled)} jobs")

    from src.benchmark.pipeline import PipelineRunner as BenchRunner
    from src.benchmark.scorer import score_pipeline

    runner = BenchRunner(model, max_tokens=200)
    results = []

    for i, job_data in enumerate(sampled):
        job = job_data["job"]
        print(f"  [{i+1}/{len(sampled)}] {job} ({job_data['difficulty']})", end=" ... ", flush=True)
        result = runner.run(job)
        scores = score_pipeline(result, job_data["expected_patterns"])
        scores["difficulty"] = job_data["difficulty"]
        results.append(scores)
        print(f"score={scores['final_score']:.3f}  latency={scores['latency']['total_ms']:.0f}ms")

    del model
    return results


def run_simple_eval(model):
    """Fallback: run 3 simple evaluations."""
    prompts = [
        ("Data Entry Clerk", ["data entry", "typing", "spreadsheet"]),
        ("Accountant", ["data entry", "verification", "reporting"]),
        ("HR Manager", ["recruitment", "onboarding", "compliance"]),
    ]
    results = []
    for job, patterns in prompts:
        print(f"  Evaluating: {job}", end=" ... ", flush=True)
        start = time.time()
        resp = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "Generate a 5-step daily workflow. Output JSON array."},
                {"role": "user", "content": f"Workflow for: {job}"},
            ],
            max_tokens=200, temperature=0.0,
        )
        elapsed = (time.time() - start) * 1000
        print(f"{elapsed:.0f}ms")
        results.append({"job": job, "latency_ms": elapsed,
                        "output": resp["choices"][0]["message"]["content"][:200]})
    return results


def save_results(results, has_cuda):
    """Save results to JSON."""
    out_dir = Path("results/gpu_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "gpu_available": has_cuda,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }

    out_file = out_dir / "gpu_eval_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


def main():
    has_cuda = check_environment()
    model_path = download_model()
    results = run_benchmark(model_path, has_cuda)
    save_results(results, has_cuda)
    print("\nDone.")


if __name__ == "__main__":
    main()
