"""
Pipeline runner: executes the 4-stage LLM pipeline and collects outputs + timing.
"""

import time
from dataclasses import dataclass, field


@dataclass
class StageResult:
    name: str
    output: str
    latency_ms: float
    tokens: int = 0


@dataclass
class PipelineResult:
    job: str
    stages: list[StageResult] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def workflow(self) -> str:
        return self.stages[0].output if len(self.stages) > 0 else ""

    @property
    def problems(self) -> str:
        return self.stages[1].output if len(self.stages) > 1 else ""

    @property
    def solutions(self) -> str:
        return self.stages[2].output if len(self.stages) > 2 else ""

    @property
    def evaluation(self) -> str:
        return self.stages[3].output if len(self.stages) > 3 else ""


class PipelineRunner:
    """Runs the 4-stage pipeline using a llama.cpp model."""

    def __init__(self, model, max_tokens: int = 300):
        self.model = model
        self.max_tokens = max_tokens

    def _generate(self, system: str, user: str, stage: str | None = None) -> tuple[str, float, int]:
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
        return output, elapsed_ms, tokens

    def run(self, job: str) -> PipelineResult:
        from src.pipeline.prompts import (
            STAGE1_SYSTEM, STAGE1_USER,
            STAGE2_SYSTEM, STAGE2_USER,
            STAGE3_SYSTEM, STAGE3_USER,
            STAGE4_SYSTEM, STAGE4_USER,
        )

        result = PipelineResult(job=job)

        out, ms, tok = self._generate(STAGE1_SYSTEM, STAGE1_USER.format(job=job), "stage1_workflow")
        result.stages.append(StageResult("workflow_generator", out, ms, tok))

        out, ms, tok = self._generate(STAGE2_SYSTEM, STAGE2_USER.format(workflow=result.workflow), "stage2_problems")
        result.stages.append(StageResult("step_analyzer", out, ms, tok))

        out, ms, tok = self._generate(STAGE3_SYSTEM, STAGE3_USER.format(problems=result.problems), "stage3_solutions")
        result.stages.append(StageResult("solution_mapper", out, ms, tok))

        out, ms, tok = self._generate(STAGE4_SYSTEM, STAGE4_USER.format(solutions=result.solutions), "stage4_evaluation")
        result.stages.append(StageResult("evaluator", out, ms, tok))

        result.total_latency_ms = sum(s.latency_ms for s in result.stages)
        return result
