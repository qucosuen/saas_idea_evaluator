"""
Project Idea Evaluator — SLM Inference Script
==============================================
Runs a small language model locally to evaluate project/startup ideas
across multiple business-viability dimensions.

Supports two backends:
  1. HuggingFace Transformers (default, works on CPU and GPU)
  2. llama-cpp-python (optional, faster on CPU with GGUF models)

Usage:
  # Install dependencies
  pip install transformers torch accelerate bitsandbytes

  # Run with default model (Qwen2.5-3B-Instruct)
  python scripts/inference.py --description "An app that helps freelancers track unpaid invoices"

  # Run with a specific model
  python scripts/inference.py --model "microsoft/Phi-3.5-mini-instruct" --description "..."

  # Run with a GGUF model via llama-cpp-python
  pip install llama-cpp-python
  python scripts/inference.py --gguf path/to/model.gguf --description "..."

  # Interactive mode
  python scripts/inference.py --interactive

  # Load a fine-tuned LoRA adapter
  python scripts/inference.py --lora-path ./my-lora-adapter --description "..."
"""

import argparse
import json
import re
import sys
from textwrap import dedent

# ---------------------------------------------------------------------------
# Evaluation schema & system prompt
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "pain_severity",
    "pain_frequency",
    "existing_alternatives",
    "willingness_to_pay",
    "market_size",
    "scalability",
    "profitability_potential",
    "defensibility",
    "time_to_value",
    "founder_market_fit_requirement",
]

SYSTEM_PROMPT = dedent("""\
    You are an expert startup and project evaluator. Given a project description,
    evaluate it across the following 10 business-viability dimensions. For each
    dimension, provide a score from 1 to 10 and a brief one-sentence reason.

    Dimensions:
    - pain_severity: How painful is the problem? (1=mild, 10=critical)
    - pain_frequency: How often do users face this? (1=rarely, 10=daily)
    - existing_alternatives: Are current solutions lacking? (1=well-solved, 10=no good alternatives)
    - willingness_to_pay: Will users pay? (1=expects free, 10=pays premium)
    - market_size: How large is the addressable market? (1=tiny niche, 10=massive)
    - scalability: Can it scale with low marginal cost? (1=hard, 10=near-zero cost)
    - profitability_potential: Revenue/margin potential? (1=low, 10=high)
    - defensibility: How hard to replicate? (1=trivially copyable, 10=strong moat)
    - time_to_value: How fast do users see value? (1=months, 10=instant)
    - founder_market_fit_requirement: Domain expertise needed? (1=anyone, 10=deep domain)

    Also provide:
    - overall_score: A weighted average (1-10)
    - summary: A 2-3 sentence overall assessment

    Respond ONLY with valid JSON. No markdown, no extra text.
    Example format:
    {
      "pain_severity": {"score": 7, "reason": "..."},
      ...
      "overall_score": 6.5,
      "summary": "..."
    }
""")

USER_PROMPT_TEMPLATE = "Evaluate this project:\n\n{description}"

# ---------------------------------------------------------------------------
# JSON extraction & validation
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from model output, tolerating surrounding text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in the text
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"(\{.*\})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return None


def validate_evaluation(data: dict) -> tuple[bool, list[str]]:
    """Check that the evaluation has the expected structure."""
    issues = []
    if not isinstance(data, dict):
        return False, ["Output is not a JSON object"]

    for dim in DIMENSIONS:
        if dim not in data:
            issues.append(f"Missing dimension: {dim}")
        elif not isinstance(data[dim], dict):
            issues.append(f"{dim} should be an object with 'score' and 'reason'")
        else:
            score = data[dim].get("score")
            if not isinstance(score, (int, float)) or not (1 <= score <= 10):
                issues.append(f"{dim}.score should be a number between 1 and 10")
            if "reason" not in data[dim]:
                issues.append(f"{dim} missing 'reason'")

    if "overall_score" not in data:
        issues.append("Missing overall_score")
    if "summary" not in data:
        issues.append("Missing summary")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Backend: HuggingFace Transformers
# ---------------------------------------------------------------------------

def load_hf_model(model_name: str, lora_path: str | None = None):
    """Load a HuggingFace model with optional 4-bit quantization and LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model: {model_name}")

    # Determine device and quantization
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if (has_cuda or has_mps) else torch.float32,
    }

    if has_cuda:
        try:
            import bitsandbytes  # noqa: F401
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
            print("Using 4-bit quantization on CUDA")
        except ImportError:
            load_kwargs["device_map"] = "auto"
            print("bitsandbytes not installed, loading in float16 on CUDA")
    elif has_mps:
        load_kwargs["device_map"] = "mps"
        print("Using MPS (Apple Silicon)")
    else:
        load_kwargs["torch_dtype"] = torch.float32
        print("Using CPU (this will be slower)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Load LoRA adapter if provided
    if lora_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, tokenizer


def generate_hf(model, tokenizer, description: str, max_new_tokens: int = 1024) -> str:
    """Generate evaluation using HuggingFace model."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=description)},
    ]

    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback for models without chat template
        input_text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{USER_PROMPT_TEMPLATE.format(description=description)}\n<|assistant|>\n"

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Backend: llama-cpp-python (GGUF)
# ---------------------------------------------------------------------------

def load_gguf_model(gguf_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
    """Load a GGUF model via llama-cpp-python."""
    from llama_cpp import Llama

    print(f"Loading GGUF model: {gguf_path}")
    model = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    return model


def generate_gguf(model, description: str, max_tokens: int = 1024) -> str:
    """Generate evaluation using a GGUF model."""
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=description)},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.1,
    )
    return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Evaluation runner with retry
# ---------------------------------------------------------------------------

def evaluate_project(
    description: str,
    generate_fn,
    max_retries: int = 2,
    verbose: bool = False,
) -> dict:
    """Run the evaluation with retry logic for JSON parsing failures."""
    for attempt in range(1, max_retries + 1):
        if verbose:
            print(f"\n--- Attempt {attempt}/{max_retries} ---")

        raw_output = generate_fn(description)

        if verbose:
            print(f"Raw output:\n{raw_output}\n")

        data = extract_json(raw_output)
        if data is None:
            print(f"Attempt {attempt}: Failed to parse JSON from output")
            continue

        valid, issues = validate_evaluation(data)
        if not valid:
            print(f"Attempt {attempt}: Validation issues: {issues}")
            if attempt < max_retries:
                continue
            # Return partial result on last attempt
            print("Returning partial result")

        return data

    return {"error": "Failed to generate valid evaluation", "raw_output": raw_output}


def print_evaluation(data: dict):
    """Pretty-print the evaluation result."""
    if "error" in data:
        print(f"\nError: {data['error']}")
        if "raw_output" in data:
            print(f"Raw output:\n{data['raw_output']}")
        return

    print("\n" + "=" * 60)
    print("  PROJECT EVALUATION")
    print("=" * 60)

    for dim in DIMENSIONS:
        if dim in data and isinstance(data[dim], dict):
            score = data[dim].get("score", "?")
            reason = data[dim].get("reason", "")
            bar = "█" * int(score) + "░" * (10 - int(score)) if isinstance(score, (int, float)) else ""
            label = dim.replace("_", " ").title()
            print(f"\n  {label}")
            print(f"    {bar}  {score}/10")
            print(f"    {reason}")

    if "overall_score" in data:
        print(f"\n{'─' * 60}")
        overall = data["overall_score"]
        bar = "█" * int(overall) + "░" * (10 - int(overall))
        print(f"  Overall Score: {bar}  {overall}/10")

    if "summary" in data:
        print(f"\n  Summary: {data['summary']}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a project idea using a small language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              python scripts/inference.py -d "An app that helps freelancers track unpaid invoices"
              python scripts/inference.py --model microsoft/Phi-3.5-mini-instruct -d "..."
              python scripts/inference.py --gguf ./models/qwen2.5-3b-q4_k_m.gguf -d "..."
              python scripts/inference.py --interactive
        """),
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    model_group.add_argument(
        "--gguf",
        help="Path to a GGUF model file (uses llama-cpp-python backend)",
    )

    # LoRA adapter
    parser.add_argument(
        "--lora-path",
        help="Path to a fine-tuned LoRA adapter directory",
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-d", "--description",
        help="Project description to evaluate",
    )
    input_group.add_argument(
        "-f", "--file",
        help="Path to a text file containing the project description",
    )
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: enter descriptions one at a time",
    )

    # Generation options
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens to generate")
    parser.add_argument("--retries", type=int, default=2, help="Max retries on JSON parse failure")
    parser.add_argument("--verbose", action="store_true", help="Print raw model output")
    parser.add_argument("--json-output", action="store_true", help="Output raw JSON instead of formatted")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers for GGUF models (-1=all)")

    args = parser.parse_args()

    # Load model
    if args.gguf:
        model = load_gguf_model(args.gguf, n_gpu_layers=args.n_gpu_layers)
        generate_fn = lambda desc: generate_gguf(model, desc, max_tokens=args.max_tokens)
    else:
        model, tokenizer = load_hf_model(args.model, lora_path=args.lora_path)
        generate_fn = lambda desc: generate_hf(model, tokenizer, desc, max_new_tokens=args.max_tokens)

    print("Model loaded. Ready to evaluate.\n")

    # Run evaluation
    def run_once(description: str):
        result = evaluate_project(
            description, generate_fn, max_retries=args.retries, verbose=args.verbose
        )
        if args.json_output:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_evaluation(result)

    if args.interactive:
        print("Interactive mode. Type a project description and press Enter.")
        print("Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                desc = input("Project description > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if desc.lower() in ("quit", "exit", "q"):
                break
            if not desc:
                continue
            run_once(desc)
    elif args.file:
        with open(args.file, "r") as f:
            description = f.read().strip()
        run_once(description)
    else:
        run_once(args.description)


if __name__ == "__main__":
    main()
