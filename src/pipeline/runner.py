"""
Pipeline runner with retry logic, validation, and caching.
Implements Phase 3 (retry), Phase 4 (structured outputs, guardrails, caching).
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.pipeline.prompts import (
    STAGE1_SYSTEM, STAGE1_USER,
    STAGE2_SYSTEM, STAGE2_USER,
    STAGE3_SYSTEM, STAGE3_USER,
    STAGE4_SYSTEM, STAGE4_USER,
)
from src.pipeline.validators import (
    validate_stage1, validate_stage2, validate_stage3, validate_stage4,
)


@dataclass
class StageOutput:
    name: str
    raw: str
    parsed: object  # list of dataclass instances
    latency_ms: float
    tokens: int
    attempts: int
    valid: bool


@dataclass
class PipelineOutput:
    job: str
    stages: list[StageOutput] = field(default_factory=list)
    total_latency_ms: float = 0.0
    success: bool = False

    def to_dict(self) -> dict:
        return {
            "job": self.job,
            "success": self.success,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "stages": [
                {
                    "name": s.name,
                    "valid": s.valid,
                    "latency_ms": round(s.latency_ms, 1),
                    "tokens": s.tokens,
                    "attempts": s.attempts,
                    "parsed": [asdict(p) for p in s.parsed] if s.valid and s.parsed else None,
                    "raw": s.raw[:500] if not s.valid else None,
                }
                for s in self.stages
            ],
        }


class PipelineRunner:
    """Runs the 4-stage pipeline with validation, retry, and optional caching."""

    def __init__(self, model, max_tokens: int = 200, max_retries: int = 2,
                 cache_dir: str | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, system: str, user: str) -> str:
        h = hashlib.md5((system + user).encode()).hexdigest()
        return h

    def _get_cached(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f).get("output")
        return None

    def _set_cached(self, key: str, output: str):
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump({"output": output}, f)

    def _generate(self, system: str, user: str, stage: str | None = None) -> tuple[str, float, int]:
        """Generate with optional cache lookup. Uses per-stage config from config.yaml."""
        key = self._cache_key(system, user)
        cached = self._get_cached(key)
        if cached:
            return cached, 0.0, 0

        from src.config import get_generation_params
        params = get_generation_params(stage)

        start = time.perf_counter()
        resp = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **params,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        output = resp["choices"][0]["message"]["content"]
        tokens = resp["usage"]["completion_tokens"]

        self._set_cached(key, output)
        return output, elapsed_ms, tokens

    def _run_stage(self, name, system, user, validator, stage: str | None = None) -> StageOutput:
        """Run a single stage with retry on validation failure."""
        last_raw = ""
        total_ms = 0.0
        total_tok = 0
        for attempt in range(1, self.max_retries + 1):
            raw, ms, tok = self._generate(system, user, stage)
            total_ms += ms
            total_tok += tok
            last_raw = raw
            try:
                parsed = validator(raw)
                return StageOutput(name, raw, parsed, total_ms, total_tok, attempt, True)
            except (ValueError, KeyError, TypeError):
                if self.cache_dir:
                    # Invalidate cache on failure so retry generates fresh
                    key = self._cache_key(system, user)
                    path = self.cache_dir / f"{key}.json"
                    if path.exists():
                        path.unlink()
                continue

        return StageOutput(name, last_raw, None, total_ms, total_tok, self.max_retries, False)

    def run(self, job: str, max_stages: int = 4) -> PipelineOutput:
        """Execute the pipeline. Use max_stages to stop early (1-4)."""
        result = PipelineOutput(job=job)

        # Stage 1: Workflow Generator (Job → Tasks with Steps)
        s1 = self._run_stage(
            "workflow_generator",
            STAGE1_SYSTEM, STAGE1_USER.format(job=job),
            validate_stage1,
            stage="stage1_workflow",
        )
        result.stages.append(s1)
        if not s1.valid or max_stages <= 1:
            result.total_latency_ms = s1.latency_ms
            result.success = s1.valid
            return result

        workflow_text = json.dumps([asdict(s) for s in s1.parsed], indent=2)

        # Stage 2: Step Analyzer (Steps → Current Solutions & Problems)
        s2 = self._run_stage(
            "step_analyzer",
            STAGE2_SYSTEM, STAGE2_USER.format(workflow=workflow_text),
            validate_stage2,
            stage="stage2_problems",
        )
        result.stages.append(s2)
        if not s2.valid or max_stages <= 2:
            result.total_latency_ms = sum(s.latency_ms for s in result.stages)
            result.success = all(s.valid for s in result.stages)
            return result

        # Filter to only steps with problems for Stage 3
        analyses_with_problems = [asdict(a) for a in s2.parsed if a.problem.strip()]
        problems_text = json.dumps(analyses_with_problems, indent=2)

        # Stage 3: Solution Mapper (Problems → Automation Solutions)
        s3 = self._run_stage(
            "solution_mapper",
            STAGE3_SYSTEM, STAGE3_USER.format(problems=problems_text),
            validate_stage3,
            stage="stage3_solutions",
        )
        result.stages.append(s3)
        if not s3.valid or max_stages <= 3:
            result.total_latency_ms = sum(s.latency_ms for s in result.stages)
            result.success = all(s.valid for s in result.stages)
            return result

        solutions_text = json.dumps([asdict(s) for s in s3.parsed], indent=2)

        # Stage 4: Evaluator (Solutions → Scores)
        s4 = self._run_stage(
            "evaluator",
            STAGE4_SYSTEM, STAGE4_USER.format(solutions=solutions_text),
            validate_stage4,
            stage="stage4_evaluation",
        )
        result.stages.append(s4)

        result.total_latency_ms = sum(s.latency_ms for s in result.stages)
        result.success = all(s.valid for s in result.stages)
        return result
