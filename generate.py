import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dllm.generation import block_diffusion_generate
from dllm.tokens import ensure_mask_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate once with a block-diffusion checkpoint.")
    parser.add_argument("--model", default="outputs/qwen3-0.6b-block-diffusion")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sub-block-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", trust_remote_code=True, attn_implementation="sdpa").to(args.device)
    mask_token_id = ensure_mask_token(tokenizer, model)
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
    output = block_diffusion_generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        mask_token_id=mask_token_id,
        block_size=args.block_size,
        sub_block_size=args.sub_block_size,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
