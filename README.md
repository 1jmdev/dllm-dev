# Qwen3-0.6B Block-Diffusion LLM

This repo is a clean, focused rebuild of NVIDIA Fast-dLLM v2 ideas for converting `Qwen/Qwen3-0.6B` into a block-diffusion language model.

It keeps the codebase intentionally small:

```text
.
├── app.py                  # Gradio chat with live denoising view
├── chat.py                 # terminal chat with streaming diffusion states
├── download_dataset.py     # dataset downloader/exporter
├── generate.py             # one-shot generation
├── train.py                # Qwen3 block-diffusion fine-tuning
├── configs/
│   └── deepspeed_zero2.json
├── dllm/                   # small core library
│   ├── collator.py         # complementary mask + token-shift training batch
│   ├── config.py
│   ├── data.py
│   ├── generation.py       # block diffusion sampler
│   └── tokens.py
├── requirements.txt
└── README.md
```

There is no `src/` layout, no `pyproject.toml`, and no conda/venv files.

## What It Implements

- Starts from `Qwen/Qwen3-0.6B`.
- Adds a learned `|<MASK>|` token.
- Pads samples to block boundaries so examples do not leak across diffusion blocks.
- Trains with Fast-dLLM v2 style complementary masking.
- Uses token-shift labels so masked token `i` is predicted by logit `i - 1`, preserving the autoregressive shape of Qwen.
- Uses a block-causal 4D attention mask: tokens see previous blocks plus all positions inside their current block.
- Generates block-by-block with parallel confidence-based unmasking inside each block.
- Streams the denoising process so you can see `[MASK]` tokens turn into text.

This is a compact implementation based on the paper and the original `original/v2` code. It does not include NVIDIA's custom remote model kernels or exact hierarchical KV cache implementation; the sampler is generic PyTorch/Transformers and prioritizes clarity for local Qwen3-0.6B fine-tuning.

## Install

Use your preferred Python environment. The repo does not create or require a venv or conda environment.

```bash
pip install -r requirements.txt
```

Install CUDA PyTorch first if your machine needs a specific CUDA wheel, for example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Download Dataset

Default Alpaca download:

```bash
python download_dataset.py \
  --dataset tatsu-lab/alpaca \
  --split train \
  --output data/raw/train.jsonl
```

Use any Hugging Face instruction dataset:

```bash
python download_dataset.py \
  --dataset yahma/alpaca-cleaned \
  --split train \
  --output data/raw/train.jsonl
```

Use a local JSON/JSONL file:

```bash
python download_dataset.py \
  --data-files /path/to/train.jsonl \
  --output data/raw/train.jsonl
```

Supported row formats include `messages`, `conversations`, or Alpaca-style `instruction`, `input`, `output`.

## Train Qwen3-0.6B

Single-process training:

```bash
python train.py \
  --model Qwen/Qwen3-0.6B \
  --data-files data/raw/train.jsonl \
  --output-dir outputs/qwen3-0.6b-block-diffusion \
  --block-size 32 \
  --context-length 2048 \
  --max-steps 6000 \
  --learning-rate 2e-5 \
  --warmup-steps 500 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --bf16 \
  --gradient-checkpointing
```

DeepSpeed ZeRO-2 training:

```bash
deepspeed train.py \
  --deepspeed configs/deepspeed_zero2.json \
  --model Qwen/Qwen3-0.6B \
  --data-files data/raw/train.jsonl \
  --output-dir outputs/qwen3-0.6b-block-diffusion \
  --block-size 32 \
  --context-length 2048 \
  --max-steps 6000 \
  --learning-rate 2e-5 \
  --warmup-steps 500 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --bf16 \
  --gradient-checkpointing
```

For a quick smoke test, lower `--max-steps`:

```bash
python train.py \
  --model Qwen/Qwen3-0.6B \
  --data-files data/raw/train.jsonl \
  --output-dir outputs/smoke \
  --max-steps 5 \
  --context-length 512 \
  --block-size 32 \
  --fp16 \
  --gradient-checkpointing
```

On an 8GB GPU, use a smaller context for smoke tests if needed:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
  --model Qwen/Qwen3-0.6B \
  --data-files data/raw/train.jsonl \
  --output-dir outputs/smoke \
  --max-steps 5 \
  --context-length 256 \
  --block-size 32 \
  --gradient-accumulation-steps 16 \
  --fp16 \
  --gradient-checkpointing
```

Model loading is fixed to `dtype="auto"` in the Python entrypoints, so there is no `--dtype` CLI flag. Use `--bf16` on GPUs that support bf16, or `--fp16` on smaller/older CUDA GPUs. Use `--gradient-checkpointing` when VRAM is tight.

## Generate

```bash
python generate.py \
  --model outputs/qwen3-0.6b-block-diffusion \
  --prompt "Explain block diffusion language models in three sentences." \
  --max-new-tokens 256 \
  --block-size 32 \
  --sub-block-size 8 \
  --threshold 0.9
```

`--threshold 1.0` is more conservative. Lower values finalize more tokens per denoising step and show faster parallel unmasking.

## Terminal Chat With Streaming Diffusion

```bash
python chat.py \
  --model outputs/qwen3-0.6b-block-diffusion \
  --max-new-tokens 256 \
  --block-size 32 \
  --sub-block-size 8 \
  --threshold 0.9
```

Commands inside chat:

- `clear` resets conversation history.
- `exit` quits.

While generating, the terminal prints `AI diffusing:` and updates the partially denoised block so you can watch `[MASK]` positions resolve.

## Web Chat With Denoising Visualization

```bash
python app.py
```

Open the Gradio URL printed in the terminal. The UI shows the current response block as highlighted tokens while diffusion is happening.

## Notes

- The training objective is intentionally simple and readable: complementary masked views plus shifted labels.
- The generation path recomputes the block with standard Transformers forward passes. This is easier to understand and works with local Qwen3 checkpoints, but it is slower than NVIDIA's custom cached implementation.
- For better results, train on a larger instruction corpus than Alpaca and keep block size consistent between training and inference.
