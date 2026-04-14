"""
Remote benchmark: runs all model candidates on the Windows machine (CPU, 16 threads).
Compares latency across 0.5B, 1.5B, 3B models.
"""
import json, time, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODELS = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
]

PROMPT_SYS = "Score this project 1-10 on 10 dimensions. Output JSON only."
PROMPT_USER = "Evaluate: An app that helps freelancers track unpaid invoices."

def download(repo, filename):
    d = "models/gguf"
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, filename)
    if os.path.exists(p):
        return p
    print(f"  Downloading {filename}...")
    hf_hub_download(repo_id=repo, filename=filename, local_dir=d)
    return p

def benchmark(name, path, threads):
    print(f"\n{'='*50}")
    print(f"  {name} ({threads} threads)")
    model = Llama(model_path=path, n_ctx=1024, n_threads=threads, verbose=False)
    # Warmup
    model.create_chat_completion(
        messages=[{"role":"user","content":"Hi"}], max_tokens=5, temperature=0.0)
    times = []
    for i in range(3):
        start = time.time()
        r = model.create_chat_completion(
            messages=[
                {"role":"system","content":PROMPT_SYS},
                {"role":"user","content":PROMPT_USER},
            ], max_tokens=300, temperature=0.0)
        elapsed = time.time() - start
        tok = r["usage"]["completion_tokens"]
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.1f}s ({tok} tok, {tok/elapsed:.1f} tok/s)")
    del model
    best = min(times)
    return {"model": name, "best_seconds": round(best, 2),
            "avg_seconds": round(sum(times)/len(times), 2),
            "tokens_per_second": round(tok/best, 1)}

def main():
    threads = os.cpu_count() or 8
    print(f"CPU threads: {threads}")
    print(f"Downloading models...")
    results = []
    for name, repo, filename in MODELS:
        path = download(repo, filename)
        r = benchmark(name, path, threads)
        results.append(r)
    # Find fastest
    fastest = min(results, key=lambda x: x["best_seconds"])
    output = {
        "hardware": {"cpu_threads": threads, "gpu": "N/A (driver too old)"},
        "results": results,
        "fastest": fastest["model"],
        "fastest_latency": fastest["best_seconds"],
    }
    print(f"\n{'='*50}")
    print(f"  FASTEST: {fastest['model']} at {fastest['best_seconds']}s")
    print(f"{'='*50}")
    os.makedirs("results", exist_ok=True)
    with open("results/remote_benchmark.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to results/remote_benchmark.json")

if __name__ == "__main__":
    main()
