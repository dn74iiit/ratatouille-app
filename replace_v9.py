with open('api.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('V9 API', 'V10 API')
text = text.replace('V8 + V9', 'V8 + V10')
text = text.replace('HF_SPACE_V9', 'HF_SPACE_V10')
text = text.replace('HF_SPACE_URL_V9', 'HF_SPACE_URL_V10')
text = text.replace('gradio_client_v9', 'gradio_client_v10')
text = text.replace('version == "v9"', 'version == "v10"')
text = text.replace('"v9": lambda: _get_client("v9")', '"v10": lambda: _get_client("v10")')
text = text.replace('model_version: str = "v8"', 'model_version: str = "v10"')
text = text.replace('"v9" = new', '"v10" = new')
text = text.replace('"v9": HF_SPACE_V9', '"v10": HF_SPACE_V10')
text = text.replace('V9 PROMPT', 'V10 PROMPT')
text = text.replace('from V9', 'from V10')

with open('api.py', 'w', encoding='utf-8') as f:
    f.write(text)
