import argparse

from dllm.data import load_instruction_dataset, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/save an instruction dataset for local dLLM training.")
    parser.add_argument("--dataset", default="tatsu-lab/alpaca", help="Hugging Face dataset name.")
    parser.add_argument("--split", default="train", help="Dataset split to save.")
    parser.add_argument("--data-files", default=None, help="Optional local JSON/JSONL files instead of HF dataset.")
    parser.add_argument("--output", default="data/raw/train.jsonl", help="Output JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_instruction_dataset(args.dataset, split=args.split, data_files=args.data_files)
    save_jsonl(dataset, args.output)
    print(f"Saved {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
