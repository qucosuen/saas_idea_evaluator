"""Profile remote inference providers (Groq and HuggingFace) for latency and error rates."""

import json
import time
import os
import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.remote_model import RemoteModel

PROVIDERS = ["groq", "huggingface"]
MODEL = "llama-3.3-70b-versatile"
HF_MODEL = "meta-llama/Llama-3.3-70b-Instruct"
TEST_PROMPT = "List 3 synonyms for 'happy' in JSON format with keys like 'word1', 'word2', 'word3'."
RUNS = 5
WARMUP_RUNS = 2


def create_client(backend: str):
    """Create a remote model client for the given backend."""
    if backend == "groq":
        return RemoteModel(model=MODEL, backend="groq")
    else:
        return RemoteModel(model=HF_MODEL, backend="huggingface")


def run_inference(client) -> tuple[float, str | None]:
    """Run a test inference and return (latency_ms, error_message)."""
    start = time.perf_counter()
    try:
        resp = client.create_chat_completion(
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=100,
            temperature=0.1,
        )
        latency = (time.perf_counter() - start) * 1000
        content = resp["choices"][0]["message"]["content"]
        return latency, None
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        error_str = str(e)
        return latency, error_str


def detect_throttle(error_msg: str) -> bool:
    """Check if the error indicates rate limiting/throttling."""
    error_lower = error_msg.lower()
    throttle_indicators = [
        "rate limit",
        "throttle",
        "too many requests",
        "429",
        "rate_limit_exceeded",
        "requests per minute",
        "requests per day",
        "tokens per minute",
        "tpm limit",
        "rpm limit",
    ]
    return any(indicator in error_lower for indicator in throttle_indicators)


def profile_provider(backend: str) -> dict:
    """Profile a single provider."""
    print(f"\n{'=' * 50}")
    print(f"  Profiling: {backend}")
    print(f"{'=' * 50}")

    try:
        client = create_client(backend)
    except Exception as e:
        print(f"  Failed to create client: {e}")
        return {
            "backend": backend,
            "available": False,
            "error": str(e),
            "latency_avg_ms": None,
            "latency_std_ms": None,
            "error_rate": 1.0,
            "throttle_count": 0,
        }

    # Warmup
    print(f"  Running {WARMUP_RUNS} warmup...")
    for _ in range(WARMUP_RUNS):
        run_inference(client)

    # Actual profiling runs
    print(f"  Running {RUNS} test runs...")
    latencies = []
    errors = []
    throttle_count = 0

    for i in range(RUNS):
        latency, error = run_inference(client)
        latencies.append(latency)

        if error:
            errors.append(error)
            print(f"    Run {i + 1}: {latency:.0f}ms - ERROR: {error[:80]}...")
            if detect_throttle(error):
                throttle_count += 1
                print(f"      -> Throttle detected!")
        else:
            print(f"    Run {i + 1}: {latency:.0f}ms - OK")

    error_rate = len(errors) / RUNS

    stats = {
        "backend": backend,
        "available": True,
        "latency_avg_ms": round(statistics.mean(latencies), 1),
        "latency_std_ms": round(statistics.stdev(latencies), 1)
        if len(latencies) > 1
        else 0,
        "latency_min_ms": round(min(latencies), 1),
        "latency_max_ms": round(max(latencies), 1),
        "error_rate": error_rate,
        "throttle_count": throttle_count,
        "total_runs": RUNS,
        "errors": errors[:5],  # Keep first 5 errors
    }

    print(f"\n  Results:")
    print(f"    Avg latency: {stats['latency_avg_ms']}ms")
    print(f"    Std dev: {stats['latency_std_ms']}ms")
    print(f"    Error rate: {error_rate * 100:.0f}%")
    print(f"    Throttle count: {throttle_count}")

    return stats


def main():
    results = {"providers": {}, "profile_timestamp": time.time()}

    for provider in PROVIDERS:
        results["providers"][provider] = profile_provider(provider)

    # Save results
    output_path = Path("results/provider_profiles.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 50}")
    print(f"  Saved profiles to {output_path}")
    print(f"{'=' * 50}")

    # Print summary
    print("\n=== Provider Summary ===")
    for provider, data in results["providers"].items():
        if data.get("available"):
            score = data["latency_avg_ms"] * (1 + data["error_rate"])
            print(
                f"{provider}: latency={data['latency_avg_ms']}ms, error_rate={data['error_rate'] * 100:.0f}%, score={score:.0f}"
            )
        else:
            print(f"{provider}: UNAVAILABLE - {data.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
