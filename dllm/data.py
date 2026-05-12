import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


def load_instruction_dataset(dataset_name: str, split: str = "train", data_files: str | None = None) -> Dataset:
    """Load a Hugging Face or local JSON/JSONL instruction dataset."""

    if data_files:
        return load_dataset("json", data_files=data_files, split=split)

    loaded = load_dataset(dataset_name)
    if isinstance(loaded, DatasetDict):
        return loaded[split]
    return loaded


def _messages_from_example(example: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in example and isinstance(example["messages"], list):
        return [{"role": m["role"], "content": str(m["content"])} for m in example["messages"]]

    if "conversations" in example and isinstance(example["conversations"], list):
        messages = []
        for turn in example["conversations"]:
            role = turn.get("role") or turn.get("from")
            content = turn.get("content") or turn.get("value") or ""
            if role in {"human", "user"}:
                role = "user"
            elif role in {"gpt", "assistant"}:
                role = "assistant"
            else:
                role = "system"
            messages.append({"role": role, "content": str(content)})
        return messages

    instruction = str(example.get("instruction") or example.get("prompt") or example.get("question") or "")
    input_text = str(example.get("input") or "")
    output = str(example.get("output") or example.get("response") or example.get("answer") or "")
    user = instruction if not input_text else f"{instruction}\n\n{input_text}"
    return [{"role": "user", "content": user}, {"role": "assistant", "content": output}]


def tokenize_instruction_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    num_proc: int = 1,
) -> Dataset:
    """Render examples through the model chat template and tokenize them."""

    def tokenize(example: dict[str, Any]) -> dict[str, list[int]]:
        messages = _messages_from_example(example)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        if tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id):
            ids.append(tokenizer.eos_token_id)
        return {"input_ids": ids}

    keep = [column for column in dataset.column_names]
    return dataset.map(tokenize, remove_columns=keep, num_proc=num_proc)


def save_jsonl(dataset: Dataset, path: str) -> None:
    """Save a dataset as JSONL for reproducible local training."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
