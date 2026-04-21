#!/usr/bin/env python3
"""
Batch runner for Stage 1 on all jobs in comprehensive_job_list.txt.
Handles rate limits with delays and retries.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.prompts import STAGE1_SYSTEM, STAGE1_USER
from src.pipeline.validators import validate_stage1
from src.config import get_model_config

import os
from openai import OpenAI


def get_model(backend: str = "openrouter"):
    """Load model based on backend choice."""
    if backend == "openrouter":
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    elif backend == "groq":
        from src.pipeline.model import load_remote_model
        return load_remote_model()
    else:
        from src.pipeline.model import load_remote_model
        return load_remote_model()


def load_jobs_from_file(filepath: str) -> list[str]:
    """Extract job titles from comprehensive_job_list.txt."""
    jobs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-"):
                job = line[1:].strip()
                if job:
                    jobs.append(job)
    return jobs


def load_completed_jobs(output_dir: Path) -> set:
    """Load already completed jobs from output directory."""
    completed = set()
    if not output_dir.exists():
        return completed
    for f in output_dir.glob("stage1_*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                completed.add(data.get("job", ""))
        except Exception:
            pass
    return completed


def run_stage1_for_job(job: str, model, backend: str, output_path: Path, force: bool = False) -> bool:
    """Run Stage 1 for a single job. Returns True on success."""
    if output_path.exists() and not force:
        print(f"  Skipping (already exists): {job}")
        return True

    system_msg = STAGE1_SYSTEM.replace("{job}", job)
    user_msg = STAGE1_USER.replace("{job}", job)

    start = time.time()
    try:
        if backend == "openrouter":
            resp = model.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct:free",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=2500,
            )
            content = resp.choices[0].message.content
            tokens = resp.usage.completion_tokens
        else:
            resp = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=2500,
            )
            content = resp["choices"][0]["message"]["content"]
            tokens = resp["usage"]["completion_tokens"]
    except Exception as e:
        print(f"  Error: {e}")
        return False

    latency_ms = (time.time() - start) * 1000

    try:
        parsed = validate_stage1(content)
        result = {"job": job, "stage": "stage1", "tasks": parsed}
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=lambda x: vars(x) if hasattr(x, '__dict__') else str(x))
        print(f"  ✓ {len(parsed)} tasks - {latency_ms:.0f}ms, {tokens} tokens")
        return True
    except Exception as e:
        print(f"  Validation error: {e}")
        with open(output_path.with_suffix(".raw.json"), "w") as f:
            f.write(content)
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch run Stage 1 for all jobs")
    parser.add_argument(
        "--file",
        default="../comprehensive_job_list.txt",
        help="Path to job list file",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between successful requests (seconds)",
    )
    parser.add_argument(
        "--batch-pause",
        type=int,
        default=60,
        help="Pause between batches (seconds)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of jobs per batch before pause",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for rate limit errors",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results exist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of jobs to process",
    )
    parser.add_argument(
        "--backend",
        default="openrouter",
        choices=["openrouter", "groq"],
        help="API backend to use",
    )
    args = parser.parse_args()

    jobs_file = Path(__file__).parent / args.file
    if not jobs_file.exists():
        jobs_file = Path(args.file)
    if not jobs_file.exists():
        print(f"Error: Job file not found: {args.file}")
        return

    jobs = load_jobs_from_file(str(jobs_file))
    print(f"Found {len(jobs)} jobs in {jobs_file}")

    output_dir = Path(__file__).parent.parent / "results" / "stage1_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.force:
        completed = load_completed_jobs(output_dir)
        jobs = [j for j in jobs if j not in completed]
        print(f"Resuming: {len(jobs)} jobs remaining ({len(completed)} done)")

    if args.limit:
        jobs = jobs[:args.limit]
        print(f"Limited to {args.limit} jobs")

    print(f"Loading {args.backend} model...")
    model = get_model(args.backend)

    total = len(jobs)
    success = 0
    failed = 0

    for i, job in enumerate(jobs):
        output_path = output_dir / f"stage1_{job.replace('/', '_').replace(' ', '_')}.json"

        for attempt in range(args.max_retries):
            if run_stage1_for_job(job, model, args.backend, output_path, args.force):
                success += 1
                break
            if attempt < args.max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Retry {attempt + 2}/{args.max_retries} after {wait_time:.1f}s...")
                time.sleep(wait_time)
        else:
            failed += 1
            print(f"  Failed after {args.max_retries} attempts: {job}")

        if (i + 1) % args.batch_size == 0:
            print(f"\n=== Batch {(i+1)//args.batch_size} complete ({i+1}/{total}), pausing {args.batch_pause}s... ===\n")
            time.sleep(args.batch_pause)
        elif i < total - 1:
            time.sleep(args.delay)

    print(f"\n=== Complete: {success} success, {failed} failed ===")


if __name__ == "__main__":
    main()