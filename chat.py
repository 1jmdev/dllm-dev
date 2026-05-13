import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dllm.generation import stream_block_diffusion_generate
from dllm.tokens import ensure_mask_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal chat with streaming diffusion states.")
    parser.add_argument("--model", default="outputs/qwen3-0.6b-block-diffusion")
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
    if getattr(model.config, "block_diffusion_label_shift", None) != "hf_causal_lm_internal_shift":
        print(
            "Warning: this checkpoint does not advertise the fixed block-diffusion label shift. "
            "If it was trained before the collator fix, retrain it before using sub-block sizes above 1."
        )
    mask_token_id = ensure_mask_token(tokenizer, model)
    messages: list[dict[str, str]] = []
    print("Block-diffusion chat. Type 'clear' or 'exit'.")

    while True:
        user = input("User: ").strip()
        if user.lower() == "exit":
            break
        if user.lower() == "clear":
            messages.clear()
            print("History cleared.")
            continue
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)

        last_text = ""
        for state in stream_block_diffusion_generate(
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
        ):
            last_text = state["text"]
            visible = "".join(state["tokens"]) if state["tokens"] else last_text
            sys.stdout.write("\rAI diffusing: " + visible[:200].replace("\n", " "))
            sys.stdout.flush()
        print("\nAI: " + last_text.strip())
        messages.append({"role": "assistant", "content": last_text.strip()})


if __name__ == "__main__":
    main()
