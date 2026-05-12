from dataclasses import dataclass


@dataclass
class BlockDiffusionConfig:
    """Runtime settings shared by training and generation."""

    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    mask_token: str = "|<MASK>|"
    block_size: int = 32
    sub_block_size: int = 8
    max_new_tokens: int = 256
    threshold: float = 0.9
    temperature: float = 0.0
    top_p: float = 0.95
