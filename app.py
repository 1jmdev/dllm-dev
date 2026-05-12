import time

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dllm.generation import stream_block_diffusion_generate
from dllm.tokens import ensure_mask_token


MODEL_PATH = "outputs/qwen3-0.6b-block-diffusion"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype="auto", trust_remote_code=True).to(DEVICE)
mask_token_id = ensure_mask_token(tokenizer, model)


def _messages(history, message):
    messages = []
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    return messages


def user_submit(message, history):
    if not message.strip():
        return "", history, [], "waiting"
    history = history + [[message, ""]]
    return "", history, [], "diffusing..."


def bot_reply(history, max_new_tokens, block_size, sub_block_size, threshold, temperature, top_p):
    message = history[-1][0]
    prompt = tokenizer.apply_chat_template(_messages(history[:-1], message), tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    start = time.time()
    final = ""

    for state in stream_block_diffusion_generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        mask_token_id=mask_token_id,
        block_size=int(block_size),
        sub_block_size=int(sub_block_size),
        max_new_tokens=int(max_new_tokens),
        threshold=float(threshold),
        temperature=float(temperature),
        top_p=float(top_p),
    ):
        final = state["text"].strip()
        history[-1][1] = final
        diffusion = [(token, 0.5 if token == "[MASK]" else -0.5) for token in state["tokens"]]
        yield history, diffusion, f"{time.time() - start:.2f}s"


with gr.Blocks() as demo:
    gr.Markdown("# Qwen3-0.6B Block-Diffusion LLM")
    gr.Markdown("Watch the current response block denoise from `[MASK]` tokens into text.")
    chatbot = gr.Chatbot(height=480)
    with gr.Row():
        user_box = gr.Textbox(placeholder="Ask something...", scale=8, show_label=False)
        send = gr.Button("Send", scale=1)
        clear = gr.Button("Clear", scale=1)
    with gr.Row():
        diffusion_view = gr.HighlightedText(label="Diffusion stream", combine_adjacent=False)
        elapsed = gr.Textbox(label="Elapsed", value="waiting", interactive=False)
    with gr.Accordion("Generation settings", open=True):
        max_new_tokens = gr.Slider(32, 1024, value=256, step=32, label="Max new tokens")
        block_size = gr.Slider(8, 64, value=32, step=8, label="Block size")
        sub_block_size = gr.Slider(1, 32, value=8, step=1, label="Sub-block size")
        threshold = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Confidence threshold")
        temperature = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")

    submit = user_box.submit(user_submit, [user_box, chatbot], [user_box, chatbot, diffusion_view, elapsed])
    click = send.click(user_submit, [user_box, chatbot], [user_box, chatbot, diffusion_view, elapsed])
    for event in (submit, click):
        event.then(
            bot_reply,
            [chatbot, max_new_tokens, block_size, sub_block_size, threshold, temperature, top_p],
            [chatbot, diffusion_view, elapsed],
        )
    clear.click(lambda: ([], [], "waiting"), outputs=[chatbot, diffusion_view, elapsed])


if __name__ == "__main__":
    demo.queue().launch(server_port=10086)
