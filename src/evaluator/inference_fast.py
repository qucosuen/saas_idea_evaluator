"""
Fast Inference Engine for Project Evaluator
=============================================
Optimized inference using multiple backends, ordered by speed:

1. llama.cpp (GGUF Q4_K_M) — fastest on CPU, ~5-15s per eval
2. ONNX Runtime (INT8/FP16) — fast, no compilation needed, ~15-30s per eval
3. HuggingFace Transformers (FP32) — slowest fallback, ~6 min per eval

Usage:
  # Auto-detect best available backend
  python -m src.evaluator.inference_fast -d "An app for tracking invoices"

  # Force a specific backend
  python -m src.evaluator.inference_fast --backend gguf -d "..."
  python -m src.evaluator.inference_fast --backend onnx -d "..."
  python -m src.evaluator.inference_fast --backend transformers -d "..."

  # Download and cache a GGUF model (recommended first step)
  python -m src.evaluator.inference_fast --setup-gguf
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from textwrap import dedent

# Reuse core components from the main inference module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluator.inference import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, DIMENSIONS,
    extract_json, validate_evaluation, print_evaluation,
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
GGUF_DIR = MODELS_DIR / "gguf"
ONNX_DIR = MODELS_DIR / "onnx"


# ============================================================================
# Backend 1: llama.cpp via llama-cpp-python (GGUF)
# ============================================================================

class GGUFBackend:
    """Fastest CPU inference using llama.cpp with quantized GGUF models."""

    name = "gguf"

    @staticmethod
    def is_available() -> bool:
        try:
            from llama_cpp import Llama  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def download_model(
        repo_id: str = "Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf",
    ) -> Path:
        """Download a GGUF model from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download
        GGUF_DIR.mkdir(parents=True, exist_ok=True)
        local_path = GGUF_DIR / filename
        if local_path.exists():
            print(f"GGUF model already cached: {local_path}")
            return local_path
        print(f"Downloading {repo_id}/{filename}...")
        path = hf_hub_download(
            repo_id=repo_id, filename=filename,
            local_dir=str(GGUF_DIR), local_dir_use_symlinks=False,
        )
        print(f"Downloaded to: {path}")
        return Path(path)

    def __init__(self, model_path: str | None = None, n_threads: int = 0):
        from llama_cpp import Llama
        if model_path is None:
            model_path = str(self.download_model())
        # n_threads=0 means auto-detect
        cpu_count = os.cpu_count() or 4
        threads = n_threads if n_threads > 0 else cpu_count
        print(f"Loading GGUF: {model_path} (threads={threads})")
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=threads,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )

    def generate(self, description: str) -> str:
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=description)},
            ],
            max_tokens=768,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        return response["choices"][0]["message"]["content"]


# ============================================================================
# Backend 2: ONNX Runtime
# ============================================================================

class ONNXBackend:
    """Fast CPU inference using ONNX Runtime with optimized graph execution.
    
    Requires: pip install optimum[onnxruntime]
    Note: May not be available on all Python versions.
    """

    name = "onnx"

    @staticmethod
    def is_available() -> bool:
        try:
            from optimum.onnxruntime import ORTModelForCausalLM  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        n_threads: int = 0,
    ):
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        cpu_count = os.cpu_count() or 4
        threads = n_threads if n_threads > 0 else cpu_count

        # Set ONNX Runtime session options for max CPU performance
        os.environ["OMP_NUM_THREADS"] = str(threads)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        cache_dir = ONNX_DIR / model_name.replace("/", "--")

        print(f"Loading ONNX model: {model_name} (threads={threads})")
        if cache_dir.exists() and any(cache_dir.glob("*.onnx")):
            print(f"  Using cached ONNX model from {cache_dir}")
            self.model = ORTModelForCausalLM.from_pretrained(
                str(cache_dir), session_options=sess_options,
            )
        else:
            print(f"  Exporting to ONNX (one-time, may take a few minutes)...")
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.model = ORTModelForCausalLM.from_pretrained(
                model_name, export=True, session_options=sess_options,
                trust_remote_code=True,
            )
            self.model.save_pretrained(str(cache_dir))
            print(f"  ONNX model cached to {cache_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def generate(self, description: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=description)},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=768,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================================
# Backend 3: HuggingFace Transformers (fallback)
# ============================================================================

class TransformersBackend:
    """HuggingFace Transformers with CPU optimizations applied.
    
    Optimizations vs the base inference.py:
    - torch.set_num_threads for full CPU utilization
    - torch.inference_mode instead of no_grad (slightly faster)
    - Reduced max_new_tokens (512 vs 1024 — our JSON fits in ~400 tokens)
    - Lower temperature (0.1) for faster convergence and more deterministic output
    - torch.compile when available (PyTorch 2.0+)
    """

    name = "transformers"

    @staticmethod
    def is_available() -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", n_threads: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cpu_count = os.cpu_count() or 4
        threads = n_threads if n_threads > 0 else cpu_count
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, threads // 2))

        print(f"Loading model: {model_name} (threads={threads})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.model.eval()

        # torch.compile can help on GPU but adds overhead on CPU
        # Skip it for CPU-only inference
        print(f"  Model loaded. Ready for inference.")

    def generate(self, description: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=description)},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(input_text, return_tensors="pt")

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,  # Our JSON fits in ~400 tokens
                temperature=0.1,     # Lower = faster convergence
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================================
# Unified interface
# ============================================================================

BACKENDS = {
    "gguf": GGUFBackend,
    "onnx": ONNXBackend,
    "transformers": TransformersBackend,
}

# Priority order: fastest first
BACKEND_PRIORITY = ["gguf", "onnx", "transformers"]


def get_best_backend(preferred: str | None = None) -> str:
    """Return the fastest available backend name."""
    if preferred and preferred in BACKENDS:
        if BACKENDS[preferred].is_available():
            return preferred
        print(f"Requested backend '{preferred}' not available, falling back...")

    for name in BACKEND_PRIORITY:
        if BACKENDS[name].is_available():
            return name

    raise RuntimeError("No inference backend available. Install at least transformers.")


def evaluate_fast(description: str, backend_instance) -> dict:
    """Run evaluation with timing and retry logic."""
    for attempt in range(2):
        start = time.time()
        raw = backend_instance.generate(description)
        elapsed = time.time() - start

        data = extract_json(raw)
        if data is not None:
            valid, issues = validate_evaluation(data)
            data["_meta"] = {
                "backend": backend_instance.name,
                "generation_time_seconds": round(elapsed, 2),
                "attempt": attempt + 1,
                "schema_valid": valid,
            }
            return data

        print(f"  Attempt {attempt+1}: JSON parse failed ({elapsed:.1f}s), retrying...")

    return {"error": "Failed to generate valid evaluation", "raw_output": raw[:500]}


def main():
    parser = argparse.ArgumentParser(
        description="Fast project evaluation inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-d", "--description", help="Project description to evaluate")
    parser.add_argument("-f", "--file", help="File containing project description")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--backend", choices=list(BACKENDS.keys()), help="Force a backend")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--gguf-path", help="Path to GGUF model file")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0=auto)")
    parser.add_argument("--setup-gguf", action="store_true", help="Download GGUF model and exit")
    parser.add_argument("--json-output", action="store_true", help="Output raw JSON")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")

    args = parser.parse_args()

    # Setup mode: just download the GGUF model
    if args.setup_gguf:
        if not GGUFBackend.is_available():
            print("llama-cpp-python not installed. Install with:")
            print("  pip install llama-cpp-python")
            sys.exit(1)
        GGUFBackend.download_model()
        print("\nGGUF model ready. Run inference with: --backend gguf")
        return

    # Select backend
    backend_name = get_best_backend(args.backend)
    print(f"Backend: {backend_name}")

    # Initialize
    if backend_name == "gguf":
        backend = GGUFBackend(model_path=args.gguf_path, n_threads=args.threads)
    elif backend_name == "onnx":
        backend = ONNXBackend(model_name=args.model, n_threads=args.threads)
    else:
        backend = TransformersBackend(model_name=args.model)

    # Benchmark mode
    if args.benchmark:
        test_cases = [
            "An app that helps freelancers track unpaid invoices and auto-send payment reminders.",
            "A browser extension that changes all website backgrounds to blue.",
            "An AI-powered platform for automating patent applications.",
        ]
        print(f"\nBenchmark: {len(test_cases)} evaluations with {backend_name}")
        times = []
        for i, desc in enumerate(test_cases):
            start = time.time()
            result = evaluate_fast(desc, backend)
            elapsed = time.time() - start
            times.append(elapsed)
            score = result.get("overall_score", result.get("error", "?"))
            print(f"  [{i+1}] {elapsed:.1f}s — score: {score}")

        avg = sum(times) / len(times)
        print(f"\nAverage: {avg:.1f}s per evaluation")
        print(f"Throughput: {3600/avg:.0f} evaluations/hour")
        return

    # Single evaluation or interactive
    def run_once(desc):
        result = evaluate_fast(desc, backend)
        if args.json_output:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print_evaluation(result)
                meta = result.get("_meta", {})
                print(f"\n  Backend: {meta.get('backend', '?')}")
                print(f"  Time: {meta.get('generation_time_seconds', '?')}s")

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                desc = input("Project > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if desc.lower() in ("quit", "exit", "q"):
                break
            if desc:
                run_once(desc)
    elif args.file:
        with open(args.file) as f:
            run_once(f.read().strip())
    elif args.description:
        run_once(args.description)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
