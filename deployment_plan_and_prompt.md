# Phase 3: The Deployment Master Plan
This is the roadmap to take the Ratatouille project from local experiments to a fully deployed, production-ready FARM stack application.

## Step 1: Model Fusion (Google Colab)
We must convert the fragmented PEFT adapter into a standalone, unified 16-bit model to utilize Hugging Face's Free Serverless GPUs.
1. Open your main training notebook (`RAT V8 JUST LOAD TRAIN AND SAVE TO HF (UPDATED).ipynb`) in Colab.
2. **Run Cell 1:** Installs Unsloth and environment dependencies.
3. **Run Cell 2:** Loads your `HF_TOKEN` from Colab secrets.
4. **Skip Cell 3:** (The massive CSV data loader is not needed).
5. **Run Cell 4 & 5:** This loads the Llama-3 base model and attaches your PEFT adapters into GPU memory.
6. **🛑 SKIP THE TRAINING & EVALUATION CELLS! 🛑**
7. Create a brand new code cell at the very bottom of the notebook and run this exact command:
   ```python
   # This automatically creates a NEW repository on Hugging Face and uploads the fused 6GB model
   model.push_to_hub_merged(
       "nd1490/ratatouille-llama3-3b-v8-MERGED", 
       tokenizer, 
       save_method="merged_16bit", 
       token=userdata.get('HF_TOKEN')
   )
   ```

## Step 2: Environment Security
1. Create a file named exactly `.env` in your project folder.
2. Add your secure tokens:
   ```text
   HF_TOKEN=hf_your_huggingface_write_token_here
   GITHUB_PAT=ghp_your_github_read_token_here
   MONGO_URI=mongodb+srv://... (we will get this later)
   ```
3. Ensure `.env` is listed in your `.gitignore` file so it is never pushed to public GitHub.

## Step 3: Backend Rerouting (FastAPI)
Update the `api.py` logic to stop using the local CPU-bound `llama-cpp-python` library.
1. Refactor `deconstruct_ingredient`, `get_recipe_archetype`, and `generate_final_recipe` to use the `requests` library or `huggingface_hub.InferenceClient`.
2. Point the requests to `https://api-inference.huggingface.co/models/nd1490/ratatouille-llama3-3b-v8-MERGED`.
3. Test the endpoints locally to ensure Scipy math still routes correctly and inference speed is restored to 16-bit quality.

## Step 4: The React Frontend & Database (FARM Stack)
1. Set up a free MongoDB Atlas M0 cluster to store User Accounts and Saved Recipes.
2. Create a React app using Vite (`npm create vite@latest ratatouille-ui`).
3. Build the UI components: 
   - A login/signup screen.
   - A dashboard to input ingredients, budget, and state.
   - A display window to show the generated recipe.
4. Deploy the React app to Vercel/Netlify for free.
5. Deploy the Python `api.py` server to Render.com for free.

---

# Context Prompt for Fresh AI Session
*Copy and paste everything below this line into a brand new chat session when you are ready to begin work again next week.*

**PROMPT:**
"Act as my Senior AI Software Engineer. We are building the final deployment phase of my Capstone Project called 'Ratatouille'. 
Here is the context of our system:
1. **The Model:** I have fine-tuned a Llama-3 3B model to generate highly detailed recipes based on mathematically optimized ingredients. I just merged my PEFT adapters into the base model at 16-bit precision and uploaded it to Hugging Face as `nd1490/ratatouille-llama3-3b-v8-MERGED`.
2. **The Backend:** I have a Python `FastAPI` server (`api.py`) that uses `pandas` to fetch live market prices from my GitHub, and `scipy.optimize.linprog` to calculate exact ingredient weights based on a user's budget.
3. **The Goal Today:** I need to update my `api.py` file to delete all local `llama-cpp-python` (GGUF) logic. Instead, I want it to ping my new Hugging Face Serverless API endpoint using my `HF_TOKEN` from a `.env` file. After the backend is successfully rerouted to the cloud GPU, we need to begin scaffolding a React frontend and a MongoDB Atlas connection to satisfy my professor's FARM stack requirement.
Let's start by looking at my `api.py` file and rewriting it to use Hugging Face cloud inference. Are you ready?"
