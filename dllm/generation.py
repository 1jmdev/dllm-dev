from collections.abc import Generator
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _inference_attention_mask(seq_len: int, current_block: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Allow causal prefix attention and bidirectional current-block attention."""

    positions = torch.arange(seq_len, device=device)
    causal = positions[None, :] <= positions[:, None]
    block_start = seq_len - current_block
    current_queries = positions[:, None] >= block_start
    current_keys = positions[None, :] >= block_start
    allowed = causal | (current_queries & current_keys)
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask.masked_fill_(allowed, 0)
    return mask[None, None, :, :]


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        probs = F.softmax(logits, dim=-1)
        tokens = probs.argmax(dim=-1)
        return tokens, probs

    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    probs_sorted = F.softmax(sorted_logits, dim=-1)
    sampled_sorted = torch.multinomial(probs_sorted.view(-1, probs_sorted.shape[-1]), 1).view(logits.shape[:-1])
    tokens = sorted_indices.gather(-1, sampled_sorted.unsqueeze(-1)).squeeze(-1)
    probs = torch.zeros_like(logits).scatter(-1, sorted_indices, probs_sorted)
    return tokens, probs


@torch.no_grad()
def block_diffusion_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    mask_token_id: int,
    block_size: int = 32,
    sub_block_size: int = 8,
    max_new_tokens: int = 256,
    threshold: float = 0.9,
    temperature: float = 0.0,
    top_p: float = 0.95,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate text block-by-block with confidence-based unmasking."""

    states = stream_block_diffusion_generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        mask_token_id=mask_token_id,
        block_size=block_size,
        sub_block_size=sub_block_size,
        max_new_tokens=max_new_tokens,
        threshold=threshold,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        emit_text=False,
    )
    final = None
    for final in states:
        pass
    if final is None:
        return input_ids
    return final["input_ids"]


@torch.no_grad()
def stream_block_diffusion_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    mask_token_id: int,
    block_size: int = 32,
    sub_block_size: int = 8,
    max_new_tokens: int = 256,
    threshold: float = 0.9,
    temperature: float = 0.0,
    top_p: float = 0.95,
    eos_token_id: int | None = None,
    emit_text: bool = True,
) -> Generator[dict[str, Any], None, None]:
    """Yield denoising states while generating a response.

    The model is autoregressive between blocks and diffusion-like inside each
    block: masked slots are repeatedly predicted, and high-confidence positions
    are finalized in parallel. This mirrors the public Fast-dLLM v2 sampler but
    stays generic enough for a locally fine-tuned Qwen3 checkpoint.
    """

    model.eval()
    device = next(model.parameters()).device
    x = input_ids.to(device)
    prompt_len = x.shape[1]
    eos = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    new_tokens = 0

    while new_tokens < max_new_tokens:
        current_block = min(block_size, max_new_tokens - new_tokens)
        block = torch.full((x.shape[0], current_block), mask_token_id, dtype=torch.long, device=device)
        x = torch.cat([x, block], dim=1)

        while (x[:, -current_block:] == mask_token_id).any():
            for start in range(0, current_block, sub_block_size):
                end = min(start + sub_block_size, current_block)
                local = x[:, -(current_block - start) : -(current_block - end) if end < current_block else None]
                if not (local == mask_token_id).any():
                    continue

                attention_mask = _inference_attention_mask(
                    seq_len=x.shape[1],
                    current_block=current_block,
                    device=device,
                    dtype=torch.float32,
                )
                logits = model(input_ids=x, attention_mask=attention_mask).logits
                shifted = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                slice_start = x.shape[1] - current_block + start
                slice_end = x.shape[1] - current_block + end
                candidate_logits = shifted[:, slice_start:slice_end, :].clone()
                candidate_logits[..., mask_token_id] = -torch.inf
                candidates, probs = _sample(candidate_logits, temperature=temperature, top_p=top_p)
                confidence = probs.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
                mask_positions = x[:, slice_start:slice_end] == mask_token_id
                confidence = confidence.masked_fill(~mask_positions, -torch.inf)

                unmask = confidence > threshold
                best = confidence.argmax(dim=-1)
                unmask[torch.arange(x.shape[0], device=device), best] = True
                unmask &= mask_positions
                x[:, slice_start:slice_end][unmask] = candidates[unmask]

                if emit_text:
                    generated = x[0, prompt_len:]
                    yield {
                        "input_ids": x.clone(),
                        "tokens": [
                            "[MASK]" if token.item() == mask_token_id else tokenizer.decode([token.item()], skip_special_tokens=True)
                            for token in generated
                        ],
                        "text": tokenizer.decode(generated[generated != mask_token_id], skip_special_tokens=True),
                    }

                if eos is not None and (x[:, prompt_len:] == eos).any():
                    eos_pos = (x[0, prompt_len:] == eos).nonzero(as_tuple=False)[0].item()
                    yield {"input_ids": x[:, : prompt_len + eos_pos + 1], "tokens": [], "text": tokenizer.decode(x[0, prompt_len : prompt_len + eos_pos], skip_special_tokens=True)}
                    return

        new_tokens += current_block

    yield {"input_ids": x, "tokens": [], "text": tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)}
