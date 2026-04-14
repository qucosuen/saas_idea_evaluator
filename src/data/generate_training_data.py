"""
Phase 2: Generate Synthetic Training Data
==========================================
Generates structured evaluation training examples from seed project descriptions.

Supports two modes:
  1. Local SLM teacher (slow but free, uses the few-shot baseline)
  2. OpenAI API teacher (fast, requires OPENAI_API_KEY env var)

The script also augments data by generating paraphrased descriptions.

Usage:
  # Local teacher (slow, ~6 min per example on CPU)
  .venv/bin/python scripts/phase2_generate_training_data.py --teacher local

  # OpenAI teacher (fast, requires API key)
  OPENAI_API_KEY=sk-... .venv/bin/python scripts/phase2_generate_training_data.py --teacher openai

  # Resume from where you left off
  .venv/bin/python scripts/phase2_generate_training_data.py --teacher local --resume
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluator.inference import DIMENSIONS, extract_json, validate_evaluation

TEACHER_SYSTEM_PROMPT = """You are an expert startup and project evaluator. Given a project description, evaluate it across 10 business-viability dimensions. For each dimension, provide a score from 1 to 10 and a one-sentence reason.

Respond ONLY with valid JSON. No markdown, no extra text.

Example:
User: Evaluate this project: A mobile app that lets dog owners find and book trusted pet sitters in their neighborhood with real-time GPS tracking.
Assistant: {"pain_severity":{"score":7,"reason":"Finding reliable pet care is stressful for dog owners"},"pain_frequency":{"score":6,"reason":"Needed for vacations, work trips, and busy days several times a year"},"existing_alternatives":{"score":4,"reason":"Rover and Wag already serve this market well"},"willingness_to_pay":{"score":7,"reason":"Pet owners regularly spend on pet care services"},"market_size":{"score":7,"reason":"Large pet ownership market globally"},"scalability":{"score":8,"reason":"Platform model scales with near-zero marginal cost"},"profitability_potential":{"score":6,"reason":"Commission-based revenue but competitive pricing pressure"},"defensibility":{"score":3,"reason":"Low switching costs and established competitors"},"time_to_value":{"score":8,"reason":"Users can book a sitter within minutes"},"founder_market_fit_requirement":{"score":3,"reason":"No deep domain expertise needed"},"overall_score":5.9,"summary":"Validated market with proven demand but faces strong incumbents. Differentiation through GPS tracking is minor. Would need a unique angle to compete."}

Now evaluate the following project. Respond with ONLY valid JSON."""


PARAPHRASE_PROMPT = """Rewrite the following project description in a different style. Keep the core idea the same but change the wording, length, and tone. Output ONLY the rewritten description, nothing else.

Original: {description}

Rewritten:"""

VARIATION_PROMPT = """Generate a new, unique startup/project idea in the {industry} industry. Write a 1-2 sentence description. Output ONLY the description, nothing else."""

# Training data format for chat fine-tuning
TRAINING_SYSTEM_MSG = (
    "You are a startup and project evaluator. Given a project description, "
    "evaluate it across 10 business-viability dimensions, each scored 1-10 "
    "with a brief justification. Output valid JSON."
)


def format_training_example(description: str, evaluation: dict) -> dict:
    """Format a single training example in chat format."""
    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_MSG},
            {"role": "user", "content": f"Evaluate this project: {description}"},
            {"role": "assistant", "content": json.dumps(evaluation, ensure_ascii=False)},
        ]
    }


# ---------------------------------------------------------------------------
# Local teacher (uses the SLM with few-shot prompting)
# ---------------------------------------------------------------------------

class LocalTeacher:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        from src.evaluator.inference import load_hf_model
        self.model, self.tokenizer = load_hf_model(model_name)
        self.model_name = model_name

    def evaluate(self, description: str, max_retries: int = 2) -> dict | None:
        import torch

        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Evaluate this project: {description}"},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        for attempt in range(max_retries):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=768,
                    temperature=0.4 + (attempt * 0.1),
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            data = extract_json(raw)
            if data is not None:
                valid, _ = validate_evaluation(data)
                if valid:
                    return data
        return None

    def paraphrase(self, description: str) -> str | None:
        import torch

        messages = [
            {"role": "user", "content": PARAPHRASE_PROMPT.format(description=description)},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Basic sanity: should be at least 20 chars and not identical
        if len(result) > 20 and result != description:
            return result
        return None

    def generate_variation(self, industry: str) -> str | None:
        import torch

        messages = [
            {"role": "user", "content": VARIATION_PROMPT.format(industry=industry)},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if len(result) > 20:
            return result.split("\n")[0]  # Take first line only
        return None


# ---------------------------------------------------------------------------
# OpenAI API teacher (fast, high quality)
# ---------------------------------------------------------------------------

class OpenAITeacher:
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self.client = OpenAI()
        self.model = model

    def _chat(self, messages: list, temperature: float = 0.4, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def evaluate(self, description: str, max_retries: int = 2) -> dict | None:
        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Evaluate this project: {description}"},
        ]
        for attempt in range(max_retries):
            raw = self._chat(messages, temperature=0.3 + attempt * 0.1)
            data = extract_json(raw)
            if data is not None:
                valid, _ = validate_evaluation(data)
                if valid:
                    return data
        return None

    def paraphrase(self, description: str) -> str | None:
        messages = [
            {"role": "user", "content": PARAPHRASE_PROMPT.format(description=description)},
        ]
        result = self._chat(messages, temperature=0.7, max_tokens=200).strip()
        if len(result) > 20 and result != description:
            return result
        return None

    def generate_variation(self, industry: str) -> str | None:
        messages = [
            {"role": "user", "content": VARIATION_PROMPT.format(industry=industry)},
        ]
        result = self._chat(messages, temperature=0.8, max_tokens=150).strip()
        if len(result) > 20:
            return result.split("\n")[0]
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_existing_data(output_file: Path) -> list:
    """Load existing training data for resume support."""
    if output_file.exists():
        with open(output_file) as f:
            return json.load(f)
    return []


def save_data(data: list, output_file: Path):
    """Save training data incrementally."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--teacher", choices=["local", "openai"], default="local")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                        help="Model for local teacher")
    parser.add_argument("--openai-model", default="gpt-4o-mini",
                        help="OpenAI model for API teacher")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Limit number of seed descriptions to process")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Generate paraphrases and variations")
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    args = parser.parse_args()

    # Paths
    seed_file = Path(__file__).parent.parent / "data" / "seed_descriptions.json"
    output_file = Path(__file__).parent.parent / "data" / "training_data.json"

    # Load seeds
    with open(seed_file) as f:
        seeds = json.load(f)
    if args.max_seeds:
        seeds = seeds[:args.max_seeds]

    # Load existing data if resuming
    training_data = load_existing_data(output_file) if args.resume else []
    existing_ids = {ex.get("source_id") for ex in training_data if "source_id" in ex}

    print(f"=== Phase 2: Synthetic Training Data Generation ===")
    print(f"Teacher: {args.teacher}")
    print(f"Seeds: {len(seeds)}")
    print(f"Existing examples: {len(training_data)}")
    print(f"Augmentation: {args.augment}")
    print()

    # Initialize teacher
    if args.teacher == "openai":
        teacher = OpenAITeacher(model=args.openai_model)
    else:
        teacher = LocalTeacher(model_name=args.model)

    stats = {"evaluated": 0, "failed": 0, "paraphrased": 0, "variations": 0}

    for i, seed in enumerate(seeds):
        sid = seed["id"]
        desc = seed["desc"]
        industry = seed.get("industry", "General")

        # Skip if already processed
        if sid in existing_ids:
            print(f"[{i+1}/{len(seeds)}] {sid} — already processed, skipping")
            continue

        print(f"\n[{i+1}/{len(seeds)}] {sid} ({industry})")
        print(f"  {desc[:80]}...")

        # 1. Evaluate the original description
        start = time.time()
        evaluation = teacher.evaluate(desc)
        elapsed = time.time() - start

        if evaluation:
            example = format_training_example(desc, evaluation)
            example["source_id"] = sid
            example["source_type"] = "original"
            training_data.append(example)
            stats["evaluated"] += 1
            print(f"  ✓ Evaluated ({elapsed:.1f}s) — overall: {evaluation.get('overall_score', '?')}")
        else:
            stats["failed"] += 1
            print(f"  ✗ Failed to generate valid evaluation ({elapsed:.1f}s)")
            continue  # Skip augmentation if original fails

        # 2. Generate a paraphrase and evaluate it
        if args.augment:
            para = teacher.paraphrase(desc)
            if para:
                print(f"  Paraphrase: {para[:80]}...")
                para_eval = teacher.evaluate(para)
                if para_eval:
                    example = format_training_example(para, para_eval)
                    example["source_id"] = f"{sid}_para"
                    example["source_type"] = "paraphrase"
                    training_data.append(example)
                    stats["paraphrased"] += 1
                    print(f"  ✓ Paraphrase evaluated — overall: {para_eval.get('overall_score', '?')}")

            # 3. Generate a variation in the same industry
            variation = teacher.generate_variation(industry)
            if variation:
                print(f"  Variation: {variation[:80]}...")
                var_eval = teacher.evaluate(variation)
                if var_eval:
                    example = format_training_example(variation, var_eval)
                    example["source_id"] = f"{sid}_var"
                    example["source_type"] = "variation"
                    training_data.append(example)
                    stats["variations"] += 1
                    print(f"  ✓ Variation evaluated — overall: {var_eval.get('overall_score', '?')}")

        # Save incrementally
        save_data(training_data, output_file)
        print(f"  Total examples so far: {len(training_data)}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"PHASE 2 SUMMARY")
    print(f"{'='*60}")
    print(f"Original evaluations: {stats['evaluated']}")
    print(f"Failed evaluations:   {stats['failed']}")
    print(f"Paraphrases:          {stats['paraphrased']}")
    print(f"Variations:           {stats['variations']}")
    print(f"Total training examples: {len(training_data)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
