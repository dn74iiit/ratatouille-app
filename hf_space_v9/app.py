"""
Ratatouille — HF Space (Free CPU, Transformers Direct)
======================================================
Loads the merged 16-bit model directly with transformers.
No GGUF. No llama-cpp-python. No compilation.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

MODEL_ID = "nd1490/ratatouille-llama3-3b-v9-MERGED"

# Load tokenizer directly from tokenizer.json (bypasses Unsloth's broken config)
print("📥 Loading tokenizer...")
tokenizer_path = hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded.")

# Load model on CPU in bfloat16 (~6GB RAM)
print("📥 Loading model (this takes ~2-3 min on CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    ignore_mismatched_sizes=True,
)
model.eval()
print("✅ Model loaded and ready.")


def generate(prompt: str, max_new_tokens: int = 350,
             temperature: float = 0.7, top_p: float = 0.9,
             repetition_penalty: float = 1.1,
             do_sample: bool = True) -> str:
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=6),
        gr.Slider(10, 512, value=350, step=10, label="Max New Tokens"),
        gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p"),
        gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty"),
        gr.Checkbox(value=True, label="Do Sample"),
    ],
    outputs=gr.Textbox(label="Generated Recipe", lines=15),
    title="🐀 Ratatouille Inference API",
    description="Full-precision recipe generation on CPU. Called by the Ratatouille FastAPI backend.",
)

demo.launch()
