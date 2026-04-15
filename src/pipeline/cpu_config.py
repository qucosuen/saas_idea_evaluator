"""
CPU thread configuration.
Reads CPU_PERCENT from .env (default 50), computes thread count from system max.
"""

import os
from pathlib import Path


def _load_env(env_path: Path | None = None) -> dict:
    """Parse a .env file into a dict. Ignores comments and blank lines."""
    if env_path is None:
        # Walk up from this file to find .env
        for parent in [Path.cwd(), Path(__file__).parent.parent.parent]:
            candidate = parent / ".env"
            if candidate.exists():
                env_path = candidate
                break
    if env_path is None or not env_path.exists():
        return {}
    result = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip("\"'")
    return result


def get_thread_count(cpu_percent: int | None = None) -> int:
    """
    Compute the number of threads to use based on a percentage of available CPUs.

    Priority:
      1. cpu_percent argument (if provided)
      2. CPU_PERCENT from .env file
      3. CPU_PERCENT environment variable
      4. cpu.percent from config.yaml
      5. Default: 50
    """
    if cpu_percent is None:
        env_vars = _load_env()
        raw = env_vars.get("CPU_PERCENT") or os.environ.get("CPU_PERCENT")
        if raw:
            cpu_percent = int(raw)
        else:
            try:
                from src.config import get_cpu_percent
                cpu_percent = get_cpu_percent()
            except Exception:
                cpu_percent = 50

    cpu_percent = max(10, min(100, cpu_percent))  # clamp to 10-100
    total = os.cpu_count() or 4
    threads = max(1, round(total * cpu_percent / 100))
    return threads


def get_cpu_info(cpu_percent: int | None = None) -> dict:
    """Return a summary of CPU config for logging."""
    total = os.cpu_count() or 4
    threads = get_thread_count(cpu_percent)
    actual_pct = round(threads / total * 100)
    return {
        "total_threads": total,
        "used_threads": threads,
        "target_percent": cpu_percent or int(_load_env().get("CPU_PERCENT", os.environ.get("CPU_PERCENT", "50"))),
        "actual_percent": actual_pct,
    }
