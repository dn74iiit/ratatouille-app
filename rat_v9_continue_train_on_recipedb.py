# -*- coding: utf-8 -*-
"""RAT_V9_CONTINUE_TRAIN_ON_RECIPEDB.ipynb

Ratatouille V9 — Continue training the 50k-trained adapter on RecipeDB data.
Pushes updated adapter back to: nd1490/ratatouille-llama3-3b-v8-50k

HOW TO USE:
  Session 1 : Run all cells top to bottom. Upload CSVs when Cell 2 prompts.
  Session 2+ : Run all cells top to bottom. Cell 2 is skipped (data already on HF).
               Cell 3 will automatically skip rows already trained on.
"""

# ============================================================
# CELL 0: INSTALL DEPENDENCIES
# xformers pin removed — unsloth uses its own Flash Attention
# ============================================================
!pip install unsloth -q
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
!pip install "trl==0.24.0" peft accelerate bitsandbytes -q
!pip install evaluate rouge_score nltk huggingface_hub datasets -q

import torch
print("=" * 40)
print(f"GPU available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 40)

# ============================================================
# CELL 1: CONFIGURATION & HF AUTH
# Run every session.
# ============================================================
import os, json
from huggingface_hub import login, HfApi, hf_hub_download
from datasets import load_dataset, Dataset
import pandas as pd

from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')   # Set via Colab Secrets (key icon in left sidebar)

HF_USERNAME          = 'nd1490'
V8_SLUG              = 'ratatouille-llama3-3b-v8-50k'       # model repo — same as V8
V8_REPO_ID           = f'{HF_USERNAME}/{V8_SLUG}'
DATASET_REPO_ID      = f'{HF_USERNAME}/recipedb-v9-processed'  # dataset repo
BASE_MODEL           = 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit'

BATCH_SIZE           = 4
GRAD_ACCUM           = 4
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM
BLOCK_SIZE           = 512
MAX_TRAIN_HOURS      = 2.5

login(token=HF_TOKEN)
api = HfApi()
print(f'Authenticated as : {HF_USERNAME}')
print(f'Model repo       : {V8_REPO_ID}')
print(f'Dataset repo     : {DATASET_REPO_ID}')

# ── Resume detection ───────────────────────────────────────────────────────
# Two counters in training_metadata.json:
#   samples_seen          -> cumulative total from all training (V8 + V9)
#   recipedb_samples_seen -> RecipeDB-specific skip counter for resuming V9
is_resume              = False
previous_steps         = 0
samples_seen           = 0
recipedb_samples_seen  = 0
model_load_source      = BASE_MODEL

print('\nChecking HF for existing training metadata...')
try:
    info_path = hf_hub_download(
        repo_id=V8_REPO_ID, filename='training_metadata.json',
        token=HF_TOKEN, local_dir='/tmp'
    )
    with open(info_path) as f:
        meta = json.load(f)

    previous_steps        = meta.get('total_steps', 0)
    samples_seen          = meta.get('samples_seen', 0)
    recipedb_samples_seen = meta.get('recipedb_samples_seen', 0)
    model_load_source     = V8_REPO_ID
    is_resume             = True

    print(f'  Previous total steps         : {previous_steps}')
    print(f'  Old V8 samples (historical)  : {samples_seen:,}')
    print(f'  RecipeDB rows trained so far : {recipedb_samples_seen:,}')
    if recipedb_samples_seen > 0:
        print(f'  -> Will SKIP first {recipedb_samples_seen:,} rows (already trained)')
    else:
        print(f'  -> First RecipeDB session — training from row 0')

except Exception as e:
    print(f'  No metadata found ({e}). Loading adapter from {V8_REPO_ID}.')
    model_load_source = V8_REPO_ID
    is_resume         = True

print(f'\nWill load model from: {model_load_source}')

# ============================================================
# CELL 2: FIRST TIME ONLY — process CSVs and push to HF
# On session 2+, this cell is skipped automatically.
# ============================================================
import requests

def dataset_exists_on_hf(repo_id, token):
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    return r.status_code == 200

if dataset_exists_on_hf(DATASET_REPO_ID, HF_TOKEN):
    print(f'Dataset already on HF: {DATASET_REPO_ID}')
    print('Skipping Cell 2 — proceeding to Cell 3.')
else:
    print('Dataset not found on HF. Running first-time setup...')
    print('Upload both files:')
    print('  1. RecipeDB_general - RecipeDB_general.csv')
    print('  2. RecipeDB_instructions.csv')

    from google.colab import files
    uploaded = files.upload()
    print('Uploaded:', list(uploaded.keys()))

    GENERAL_PATH      = '/content/RecipeDB_general - RecipeDB_general.csv'
    INSTRUCTIONS_PATH = '/content/RecipeDB_instructions.csv'
    READ_OPTS = dict(engine='python', on_bad_lines='skip',
                     encoding='utf-8', encoding_errors='ignore')

    df_gen  = pd.read_csv(GENERAL_PATH,      **READ_OPTS)
    df_inst = pd.read_csv(INSTRUCTIONS_PATH, **READ_OPTS)
    print(f'General      : {len(df_gen):,} rows')
    print(f'Instructions : {len(df_inst):,} rows')

    df_gen ['Recipe_id'] = df_gen ['Recipe_id'].astype(str).str.strip()
    df_inst['recipe_id'] = df_inst['recipe_id'].astype(str).str.strip()

    # INNER JOIN — only rows where BOTH title and steps exist (no fake/mismatched data)
    df = pd.merge(
        df_gen [['Recipe_id', 'Recipe_title', 'Region']],
        df_inst[['recipe_id', 'steps']],
        left_on='Recipe_id', right_on='recipe_id', how='inner'
    ).rename(columns={'Recipe_title': 'title', 'steps': 'directions_raw'})

    df = df.dropna(subset=['title', 'directions_raw'])
    df = df[df['directions_raw'].str.strip().ne('')]
    print(f'Merged       : {len(df):,} clean recipes (title + steps both present)')

    import re
    UNIT_TO_G = {
        'cup': 240.0, 'cups': 240.0,
        'tablespoon': 14.79, 'tablespoons': 14.79, 'tbsp': 14.79, 'tbs': 14.79,
        'teaspoon': 4.93, 'teaspoons': 4.93, 'tsp': 4.93,
        'fluid ounce': 29.57, 'fluid ounces': 29.57, 'fl oz': 29.57,
        'pint': 473.18, 'pints': 473.18,
        'quart': 946.35, 'quarts': 946.35,
        'gallon': 3785.41, 'gallons': 3785.41,
        'liter': 1000.0, 'liters': 1000.0, 'litre': 1000.0, 'litres': 1000.0,
        'milliliter': 1.0, 'milliliters': 1.0, 'ml': 1.0,
        'pound': 453.59, 'pounds': 453.59, 'lb': 453.59, 'lbs': 453.59,
        'ounce': 28.35, 'ounces': 28.35, 'oz': 28.35,
        'kilogram': 1000.0, 'kilograms': 1000.0, 'kg': 1000.0,
        'gram': 1.0, 'grams': 1.0, 'g': 1.0,
        'milligram': 0.001, 'milligrams': 0.001, 'mg': 0.001,
        'clove': 5.0, 'cloves': 5.0, 'slice': 25.0, 'slices': 25.0,
        'piece': 30.0, 'pieces': 30.0, 'can': 400.0, 'cans': 400.0,
        'package': 250.0, 'packages': 250.0, 'pkg': 250.0,
        'bunch': 100.0, 'bunches': 100.0, 'sprig': 2.0, 'sprigs': 2.0,
        'pinch': 0.3, 'pinches': 0.3, 'dash': 0.6, 'dashes': 0.6,
        'drop': 0.05, 'drops': 0.05, 'handful': 40.0, 'handfuls': 40.0,
        'stick': 113.0, 'sticks': 113.0,
    }
    _UNITS_SORTED = sorted(UNIT_TO_G.keys(), key=len, reverse=True)
    _UNIT_PATTERN = '|'.join(re.escape(u) for u in _UNITS_SORTED)
    _INGR_RE = re.compile(
        r'(?P<qty>\d+\s*/\s*\d+|\d+\.\d+|\d+)'
        r'\s*(?P<unit>' + _UNIT_PATTERN + r')'
        r'\s*(?P<name>[a-z][a-z ,\-()]{1,60})',
        re.IGNORECASE
    )

    def parse_fraction(s):
        s = s.strip()
        if '/' in s:
            num, den = s.split('/')
            return float(num.strip()) / float(den.strip())
        return float(s)

    def extract_gram_ingredients(steps_text):
        seen, result = set(), []
        for m in _INGR_RE.finditer(steps_text):
            qty_str  = m.group('qty').replace(' ', '')
            unit_str = m.group('unit').lower().strip()
            name_str = m.group('name').strip().rstrip(' ,')
            name_key = re.sub(r'\s+', ' ', name_str.lower())
            if name_key in seen:
                continue
            seen.add(name_key)
            try:
                qty_g = parse_fraction(qty_str) * UNIT_TO_G.get(unit_str, 1.0)
                result.append(f'{qty_g:.1f}g {name_str}')
            except ValueError:
                result.append(name_str)
        return result

    def split_directions(steps_text):
        raw = re.split(r'\s\.\s', steps_text.strip())
        out = []
        for s in raw:
            s = s.strip()
            if len(s) > 10:
                out.append(s[0].upper() + s[1:] + '.')
        return out

    def format_v9_recipe(row):
        try:
            title     = str(row['title']).strip()
            steps_raw = str(row['directions_raw']).strip()
            ingr_list = extract_gram_ingredients(steps_raw)
            dir_list  = split_directions(steps_raw)
            if not ingr_list or not dir_list:
                return {'text': ''}
            ingr_text = '\n'.join(f'- {i}' for i in ingr_list)
            dir_text  = '\n'.join(f'{idx+1}. {s}' for idx, s in enumerate(dir_list))
            text = (
                f'### INGREDIENTS:\n{ingr_text}\n'
                f'### TITLE:\n{title}\n'
                f'### DIRECTIONS:\n{dir_text}\n'
                f'<|end_of_text|>'
            )
            return {'text': text[:2000]}
        except Exception:
            return {'text': ''}

    ds = Dataset.from_pandas(df.reset_index(drop=True))
    ds = ds.map(format_v9_recipe, num_proc=2)
    ds = ds.filter(lambda x: len(x['text']) > 100)
    print(f'Formatted : {len(ds):,} samples')
    print(f'Pushing to HF: {DATASET_REPO_ID} ...')
    ds.push_to_hub(DATASET_REPO_ID, token=HF_TOKEN)
    print(f'Done! Dataset live at: https://huggingface.co/datasets/{DATASET_REPO_ID}')

# ============================================================
# CELL 3: LOAD DATASET FROM HF + APPLY RESUME SKIP
# Run every session. Fast — no re-processing.
# ============================================================
print(f'Loading dataset from HF: {DATASET_REPO_ID} ...')
full_dataset    = load_dataset(DATASET_REPO_ID, split='train', token=HF_TOKEN)
total_available = len(full_dataset)
print(f'Total samples in dataset : {total_available:,}')

# Skip rows already trained on in previous V9 sessions
if recipedb_samples_seen > 0:
    if recipedb_samples_seen >= total_available:
        print('All RecipeDB samples already trained on! Nothing left.')
        raise SystemExit('Training complete for this dataset.')
    dataset = full_dataset.select(range(recipedb_samples_seen, total_available))
    print(f'Skipped first {recipedb_samples_seen:,} rows (already trained).')
else:
    dataset = full_dataset
    print('First RecipeDB session — starting from row 0.')

print(f'Rows queued for THIS session : {len(dataset):,}')
print('\n--- First sample this session ---')
print(dataset[0]['text'][:500])
print('---')

# ============================================================
# CELL 4: LOAD MODEL + EXISTING ADAPTER
# ============================================================
import os, torch

# Disable Unsloth telemetry ping (avoids TimeoutError if HF stats endpoint is slow)
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

from unsloth import FastLanguageModel

print(f'Loading from: {model_load_source}')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_load_source,
    max_seq_length=BLOCK_SIZE,
    load_in_4bit=True,
    token=HF_TOKEN,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'right'

if not is_resume:
    print('Attaching fresh LoRA adapters (fallback)...')
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=16, lora_dropout=0.05,
        bias='none', use_gradient_checkpointing='unsloth',
        random_state=3407, use_rslora=False,
    )
else:
    print('Using existing LoRA adapters from HuggingFace.')

FastLanguageModel.for_training(model)
print('Model ready.')

# ============================================================
# CELL 5: TRAIN + PUSH TO HF
# ============================================================
import time, json
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datetime import datetime

class SafeTimeoutCallback(TrainerCallback):
    def __init__(self, max_hours):
        self.max_seconds = (max_hours * 3600) - 300   # 5-min buffer before Colab cuts session
        self.start = None
    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        print(f'Timer started. Hard cut-off in {MAX_TRAIN_HOURS}h.')
    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.start > self.max_seconds:
            print('Time limit reached — saving and stopping...')
            control.should_save = True
            control.should_training_stop = True
        return control

# TrainingArguments works on ALL trl/transformers versions
training_args = TrainingArguments(
    output_dir                  = V8_SLUG,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = 2e-4,
    lr_scheduler_type           = 'constant',
    warmup_steps                = 0,
    num_train_epochs            = 1,
    logging_steps               = 10,
    save_strategy               = 'no',
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    optim                       = 'adamw_8bit',
    weight_decay                = 0.01,
    seed                        = 3407,
    report_to                   = 'none',
)

trainer = SFTTrainer(
    model              = model,
    processing_class   = tokenizer,
    train_dataset      = dataset,
    dataset_text_field = 'text',
    max_seq_length     = BLOCK_SIZE,
    packing            = True,
    args               = training_args,
    callbacks          = [SafeTimeoutCallback(MAX_TRAIN_HOURS)],
)

print('Starting training on RecipeDB data...')
train_result = trainer.train()

# ── Compute stats ──────────────────────────────────────────────────────────
session_steps     = train_result.global_step
new_total_steps   = previous_steps  + session_steps
session_samples   = session_steps   * EFFECTIVE_BATCH_SIZE
new_recipedb_seen = recipedb_samples_seen + session_samples   # skip counter
new_total_samples = samples_seen    + session_samples         # historical total

remaining = max(0, total_available - new_recipedb_seen)
print(f'\nSession steps                   : {session_steps}')
print(f'RecipeDB rows trained this run  : {session_samples:,}')
print(f'RecipeDB rows trained total     : {new_recipedb_seen:,} / {total_available:,}')
print(f'RecipeDB rows still remaining   : {remaining:,}')
if remaining > 0:
    print('  -> Come back and run this notebook again to continue.')
else:
    print('  -> All RecipeDB samples trained! Dataset complete.')

# ── Push model + tokenizer to HF ──────────────────────────────────────────
print(f'\nPushing adapter to {V8_REPO_ID} ...')
model.push_to_hub(
    V8_REPO_ID, token=HF_TOKEN,
    commit_message=f'V9 RecipeDB: {new_recipedb_seen:,}/{total_available:,} rows seen'
)
tokenizer.push_to_hub(V8_REPO_ID, token=HF_TOKEN)

# ── Save metadata — recipedb_samples_seen is the key resume counter ────────
new_meta = {
    'version'              : 'V9-RecipeDB',
    'total_steps'          : new_total_steps,
    'samples_seen'         : new_total_samples,        # historical running total
    'recipedb_samples_seen': new_recipedb_seen,         # WHERE TO RESUME NEXT SESSION
    'recipedb_total'       : total_available,
    'effective_batch_size' : EFFECTIVE_BATCH_SIZE,
    'last_updated'         : datetime.now().isoformat(),
    'notes'                : 'V9 continued training on RecipeDB (inner join, gram-normalised)'
}
with open('training_metadata.json', 'w') as f:
    json.dump(new_meta, f, indent=4)

api.upload_file(
    path_or_fileobj='training_metadata.json',
    path_in_repo='training_metadata.json',
    repo_id=V8_REPO_ID, token=HF_TOKEN, repo_type='model'
)
print(f'Done! Next session resumes from row {new_recipedb_seen:,}.')

# ============================================================
# CELL 6 (OPTIONAL): QUICK INFERENCE CHECK
# ============================================================
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

test_prompt = (
    '### INGREDIENTS:\n'
    '- 907.2g chicken breast, cubed\n'
    '- 240.0g coconut milk\n'
    '- 14.79g curry powder\n'
    '- 120.0g onion, diced\n'
    '### TITLE:\n'
    '### DIRECTIONS:\n'
)
inputs = tokenizer(test_prompt, return_tensors='pt').to('cuda')
out    = model.generate(
    **inputs, max_new_tokens=300, temperature=0.7,
    do_sample=True, eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(out[0], skip_special_tokens=True))