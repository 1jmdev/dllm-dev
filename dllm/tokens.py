from transformers import PreTrainedModel, PreTrainedTokenizerBase


def ensure_mask_token(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel | None = None,
    mask_token: str = "|<MASK>|",
) -> int:
    """Ensure the tokenizer/model have the block-diffusion mask token."""

    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer.convert_tokens_to_ids(mask_token)
