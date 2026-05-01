# =============================================================
# RATATOUILLE — Convert Merged 16-bit Model → F16 GGUF
# =============================================================
# Run this in Google Colab (any runtime, GPU not needed).
# It downloads your merged model, converts to F16 GGUF,
# and uploads the result to a new HuggingFace repo.
#
# STEPS:
#   1. Open Google Colab
#   2. Make sure your HF_TOKEN is saved in Colab Secrets
#      (Settings gear icon → Secrets → Add HF_TOKEN)
#   3. Copy each cell below into separate Colab cells and run them in order
# =============================================================


# ===================== CELL 1 =====================
# Install dependencies (takes ~60 seconds)

# !pip install huggingface_hub sentencepiece protobuf transformers -q
# !git clone https://github.com/ggerganov/llama.cpp.git
# !pip install -r llama.cpp/requirements.txt -q
# print("✅ Dependencies installed.")


# ===================== CELL 2 =====================
# Authenticate with Hugging Face

# from google.colab import userdata
# from huggingface_hub import login
# HF_TOKEN = userdata.get('HF_TOKEN')
# login(token=HF_TOKEN)
# print("✅ Logged in to Hugging Face.")


# ===================== CELL 3 =====================
# Download merged model & convert to F16 GGUF
# This takes ~5-8 minutes (downloads 6GB, then converts)

# import subprocess
# 
# MODEL_ID = "nd1490/ratatouille-llama3-3b-v8-MERGED"
# OUTPUT_FILE = "ratatouille-llama3-3b-v8-F16.gguf"
# 
# print("📥 Downloading model and converting to F16 GGUF...")
# print("   (This downloads ~6GB and converts — be patient)")
# 
# result = subprocess.run([
#     "python", "llama.cpp/convert_hf_to_gguf.py",
#     MODEL_ID,                    # reads directly from HF
#     "--outfile", OUTPUT_FILE,
#     "--outtype", "f16",          # ← Full 16-bit, ZERO quality loss
# ], capture_output=True, text=True)
# 
# print(result.stdout[-500:] if result.stdout else "")
# if result.returncode != 0:
#     print("❌ ERROR:", result.stderr[-500:])
# else:
#     import os
#     size_gb = os.path.getsize(OUTPUT_FILE) / (1024**3)
#     print(f"✅ Conversion complete! File: {OUTPUT_FILE} ({size_gb:.2f} GB)")


# ===================== CELL 4 =====================
# Upload the F16 GGUF to a new HuggingFace repository

# from huggingface_hub import HfApi
# 
# api = HfApi()
# REPO_ID = "nd1490/ratatouille-llama3-3b-v8-F16-GGUF"
# OUTPUT_FILE = "ratatouille-llama3-3b-v8-F16.gguf"
# 
# # Create the repo (will skip if it already exists)
# api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
# 
# print(f"📤 Uploading {OUTPUT_FILE} to {REPO_ID}...")
# print("   (Uploading ~6GB — this takes 5-10 minutes)")
# 
# api.upload_file(
#     path_or_fileobj=OUTPUT_FILE,
#     path_in_repo=OUTPUT_FILE,
#     repo_id=REPO_ID,
#     repo_type="model",
# )
# 
# print(f"✅ DONE! Your F16 GGUF is live at: https://huggingface.co/{REPO_ID}")
# print(f"   File: {OUTPUT_FILE}")
