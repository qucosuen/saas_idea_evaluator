"""
Phase 3: QLoRA Fine-Tuning
============================
Fine-tunes a small language model on the synthetic training data using QLoRA.

Requirements:
  pip install peft trl datasets bitsandbytes

Usage:
  # Fine-tune with default settings (Qwen2.5-3B on GPU)
  .venv/bin/python scripts/phase3_finetune_qlora.py

  # Fine-tune SmolLM2-1.7B on CPU (slow but works)
  .venv/bin/python scripts/phase3_finetune_qlora.py \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --no-quantize \
    --batch-size 1 \
    --epochs 3

  # Custom LoRA config
  .venv/bin/python scripts/phase3_finetune_qlora.py \
    --lora-rank 32 --lora-alpha 64 --lr 1e-4
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for project evaluator")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--data", default="data/training_data.json",
                        help="Path to training data JSON")
    parser.add_argument("--output-dir", default="models/project-evaluator-lora",
                        help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit quantization (for CPU training)")
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Fraction of data for evaluation")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()

    # Check dependencies
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install peft trl datasets bitsandbytes")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # 1. Load training data
    # ---------------------------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Training data not found: {data_path}")
        print("Run phase2_generate_training_data.py first.")
        sys.exit(1)

    with open(data_path) as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} training examples from {data_path}")

    # Convert to the format expected by SFTTrainer
    # Each example has a "messages" field with system/user/assistant messages
    conversations = []
    for ex in raw_data:
        if "messages" in ex:
            conversations.append({"messages": ex["messages"]})

    if not conversations:
        print("No valid training examples found.")
        sys.exit(1)

    print(f"Valid conversations: {len(conversations)}")

    # Split into train/eval
    import random
    random.seed(42)
    random.shuffle(conversations)
    split_idx = max(1, int(len(conversations) * (1 - args.eval_split)))
    train_data = conversations[:split_idx]
    eval_data = conversations[split_idx:] if split_idx < len(conversations) else []

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None

    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # ---------------------------------------------------------------------------
    # 2. Load model and tokenizer
    # ---------------------------------------------------------------------------
    has_cuda = torch.cuda.is_available()
    print(f"\nDevice: {'CUDA' if has_cuda else 'CPU'}")
    print(f"Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}

    if has_cuda and not args.no_quantize:
        print("Loading with 4-bit quantization (QLoRA)")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
    elif has_cuda:
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
    else:
        print("CPU training — this will be slow. Consider using a GPU.")
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # ---------------------------------------------------------------------------
    # 3. Configure LoRA
    # ---------------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    print(f"\nLoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout}")

    # ---------------------------------------------------------------------------
    # 4. Training configuration
    # ---------------------------------------------------------------------------
    output_dir = Path(args.output_dir)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        max_length=args.max_seq_length,
        fp16=has_cuda,
        bf16=False,
        gradient_checkpointing=has_cuda,
        report_to="none",
        seed=42,
        packing=False,
    )

    # ---------------------------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")
    print(f"Starting training for {args.epochs} epochs...")
    print()

    trainer.train()

    # ---------------------------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------------------------
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"LoRA adapter saved to: {output_dir}")
    print(f"To use: python scripts/inference.py --lora-path {output_dir} -d '...'")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
