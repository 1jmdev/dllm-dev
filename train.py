import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from dllm.collator import BlockDiffusionCollator
from dllm.data import load_instruction_dataset, tokenize_instruction_dataset
from dllm.tokens import ensure_mask_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-0.6B into a block-diffusion LLM.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Base model or checkpoint path.")
    parser.add_argument("--dataset", default=None, help="HF dataset name. Ignored when --data-files is set.")
    parser.add_argument("--data-files", default="data/raw/train.jsonl", help="Local JSON/JSONL training file.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--output-dir", default="outputs/qwen3-0.6b-block-diffusion", help="Checkpoint output directory.")
    parser.add_argument("--mask-token", default="|<MASK>|", help="Learned mask token string.")
    parser.add_argument("--block-size", type=int, default=32, help="Block diffusion block size.")
    parser.add_argument("--context-length", type=int, default=2048, help="Packed training context length.")
    parser.add_argument("--max-steps", type=int, default=6000, help="Training steps. Use fewer for smoke tests.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup steps.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Packed sequences per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval.")
    parser.add_argument("--logging-steps", type=int, default=1, help="Logging interval.")
    parser.add_argument("--num-proc", type=int, default=1, help="Dataset preprocessing workers.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 training.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 training.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--dataloader-num-workers", type=int, default=4, help="Training dataloader workers.")
    parser.add_argument("--torch-compile", action="store_true", help="Enable torch.compile for long runs.")
    parser.add_argument("--deepspeed", default=None, help="Optional DeepSpeed config path.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    mask_token_id = ensure_mask_token(tokenizer, model, args.mask_token)
    model.config.mask_token_id = mask_token_id
    model.config.bd_size = args.block_size
    model.config.block_diffusion_base_model = args.model
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset_name = args.dataset or "json"
    data_files = args.data_files if dataset_name == "json" or args.data_files else None
    dataset = load_instruction_dataset(dataset_name, split=args.split, data_files=data_files)
    tokenized = tokenize_instruction_dataset(dataset, tokenizer, max_length=args.context_length, num_proc=args.num_proc)

    collator = BlockDiffusionCollator(
        mask_token_id=mask_token_id,
        block_size=args.block_size,
        max_length=args.context_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_strategy="steps",
        save_total_limit=5,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        torch_compile=args.torch_compile,
        include_tokens_per_second=True,
        report_to="wandb" if "WANDB_PROJECT" in os.environ else "none",
        remove_unused_columns=False,
    )

    print(
        "Training config: "
        f"steps={args.max_steps}, context={args.context_length}, block={args.block_size}, "
        f"micro_batch={args.per_device_train_batch_size}, grad_accum={args.gradient_accumulation_steps}, "
        f"bf16={args.bf16}, fp16={args.fp16}, gradient_checkpointing={args.gradient_checkpointing}, "
        f"torch_compile={args.torch_compile}"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved block-diffusion checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
