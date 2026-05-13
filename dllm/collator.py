from dataclasses import dataclass
import random

import torch


def block_causal_attention_mask(seq_len: int, block_size: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Return an additive mask: previous blocks + current block are visible."""

    positions = torch.arange(seq_len)
    query_blocks = positions[:, None] // block_size
    key_blocks = positions[None, :] // block_size
    allowed = key_blocks <= query_blocks
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    mask.masked_fill_(allowed, 0)
    return mask


@dataclass
class BlockDiffusionCollator:
    """Build Fast-dLLM v2 style complementary masked training batches.

    Each sample is padded to a multiple of `block_size` with mask tokens, then
    duplicated into complementary views. Labels are placed at the masked token
    positions because Hugging Face causal LM heads apply the AR one-token shift
    internally: logit `i - 1` is trained against label `i`.
    """

    mask_token_id: int
    block_size: int = 32
    max_length: int = 2048
    pad_to_context: bool = True

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        packed: list[int] = []
        for feature in features:
            ids = feature["input_ids"][: self.max_length]
            pad_len = (-len(ids)) % self.block_size
            packed.extend(ids + [self.mask_token_id] * pad_len)

        if self.pad_to_context:
            packed = packed[: self.max_length]
            packed.extend([self.mask_token_id] * (self.max_length - len(packed)))

        clean = torch.tensor(packed, dtype=torch.long)
        valid = clean != self.mask_token_id

        mask = torch.zeros_like(clean, dtype=torch.bool)
        for start in range(0, clean.numel(), self.block_size):
            end = min(start + self.block_size, clean.numel())
            for pos in range(start, end):
                if valid[pos]:
                    mask[pos] = bool(random.getrandbits(1))

        views = []
        labels = []
        for view_mask in (mask, valid & ~mask):
            input_ids = clean.clone()
            input_ids[view_mask] = self.mask_token_id
            labels_for_view = torch.full_like(clean, -100)
            target_positions = torch.where(view_mask)[0]
            labels_for_view[target_positions] = clean[target_positions]
            views.append(input_ids)
            labels.append(labels_for_view)

        input_ids = torch.stack(views)
        labels_tensor = torch.stack(labels)
        attention_mask = block_causal_attention_mask(input_ids.shape[1], self.block_size)
        attention_mask = attention_mask[None, None, :, :].expand(input_ids.shape[0], 1, -1, -1).clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels_tensor}


def flatten_trainer_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Flatten `[batch, views, seq]` if a Trainer adds an outer batch dimension."""

    return {key: value.flatten(0, 1) if value.ndim == 3 else value for key, value in batch.items()}
