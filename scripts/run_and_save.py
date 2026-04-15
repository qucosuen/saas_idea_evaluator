"""Run the pipeline for a job and save the result to results/pipeline_<job>.json"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.runner import PipelineRunner

job = sys.argv[1] if len(sys.argv) > 1 else "Devops Engineer"
use_remote = "--remote" in sys.argv
print(f"Running pipeline for: {job}")

if use_remote:
    from src.pipeline.model import load_remote_model
    print("Using HuggingFace Inference API...")
    model = load_remote_model()
else:
    from src.pipeline.model import find_model, load_model
    model_path = find_model()
    print(f"Loading {model_path.name}...")
    model = load_model(model_path)

runner = PipelineRunner(model, max_retries=2)
start = time.time()
result = runner.run(job)
elapsed = time.time() - start

print(f"Status: {'Success' if result.success else 'Failed'}")
print(f"Stages: {len(result.stages)}")
print(f"Time: {elapsed:.1f}s")

for s in result.stages:
    status = "✓" if s.valid else "✗"
    print(f"  {status} {s.name}: {s.latency_ms:.0f}ms, {s.tokens} tok")

slug = job.lower().replace(" ", "_")
out_path = f"results/pipeline_{slug}.json"
with open(out_path, "w") as f:
    json.dump(result.to_dict(), f, indent=2)
print(f"\nSaved to {out_path}")
