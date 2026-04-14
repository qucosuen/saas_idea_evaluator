"""
CLI interface for the local-first LLM pipeline.

Usage:
  # Analyze a single job
  python -m src.pipeline.cli "Accountant"

  # With custom model
  python -m src.pipeline.cli --model models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf "Accountant"

  # JSON output
  python -m src.pipeline.cli --json "Accountant"

  # Enable caching (faster repeated runs)
  python -m src.pipeline.cli --cache "Accountant"

  # Interactive mode
  python -m src.pipeline.cli
"""

import argparse
import json
import sys
from dataclasses import asdict

from src.pipeline.model import find_model, load_model
from src.pipeline.runner import PipelineRunner


def print_result(output):
    """Pretty-print a pipeline result."""
    print(f"\n{'='*60}")
    print(f"  JOB: {output.job}")
    print(f"  Status: {'✓ Success' if output.success else '✗ Failed'}")
    print(f"  Total latency: {output.total_latency_ms:.0f}ms")
    print(f"{'='*60}")

    for stage in output.stages:
        status = "✓" if stage.valid else "✗"
        print(f"\n  {status} {stage.name} ({stage.latency_ms:.0f}ms, {stage.tokens} tok, {stage.attempts} attempt(s))")

        if stage.valid and stage.parsed:
            for item in stage.parsed:
                d = asdict(item)
                step = d.pop("step_number", "?")
                parts = [f"{k}={v}" for k, v in d.items() if v]
                print(f"    Step {step}: {', '.join(parts[:3])}")
        elif not stage.valid:
            print(f"    Raw: {stage.raw[:150]}...")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Local-first LLM pipeline: Job → Workflow → Problems → Solutions → Evaluation")
    parser.add_argument("job", nargs="?", help="Job title to analyze")
    parser.add_argument("--model", help="Path to GGUF model")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--cache", action="store_true", help="Enable response caching")
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    model_path = find_model(args.model)
    print(f"Loading {model_path.name}...", file=sys.stderr)
    model = load_model(model_path)

    runner = PipelineRunner(
        model,
        max_tokens=args.max_tokens,
        max_retries=args.retries,
        cache_dir=".cache/pipeline" if args.cache else None,
    )

    def run_job(job_title):
        result = runner.run(job_title)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_result(result)

    if args.job:
        run_job(args.job)
    else:
        print("Interactive mode. Enter a job title, or 'quit' to exit.\n", file=sys.stderr)
        while True:
            try:
                job = input("Job > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if job.lower() in ("quit", "exit", "q"):
                break
            if job:
                run_job(job)


if __name__ == "__main__":
    main()
