# Technical Architecture: Ratatouille Full-Stack Application (Post-Deployment)

## Part 1: High-Level Architecture (Short Version)

Ratatouille is a fully deployed, cost-aware recipe generation system built on a distributed **FARM Stack** (FastAPI, React, MongoDB) integrated with a remote AI Inference pipeline. 

The system operates across four distinct deployment zones, optimized to bypass free-tier hardware limitations:
1. **Frontend (Vercel):** A React 18 / Vite Single Page Application (SPA) featuring a responsive, glassmorphic UI. It serves as the primary user presentation layer.
2. **Backend Orchestrator (Render):** A FastAPI Python server acting as the central nervous system. It handles real-time market data ingestion, executes linear programming (SciPy) for budget optimization, and acts as the secure gateway to the AI and Database layers.
3. **AI Inference (Hugging Face Spaces):** A remote Free CPU environment running a merged Llama 3 (3B) model. It loads `safetensors` directly via the `transformers` library in `bfloat16` precision, completely bypassing GGUF or llama.cpp for a streamlined Python-native deployment.
4. **Data Persistence (MongoDB Atlas):** A cloud NoSQL database managing user recipe histories and saved configurations, accessed via asynchronous database drivers.

---

## Part 2: Detailed Component Breakdown & Code Explanations (Long Version)

### 1. Frontend Layer (React + Vite -> Deployed on Vercel)
**Architecture & Implementation:**
* **Core Framework:** React 18 built with Vite for optimized asset chunking, fast Hot Module Replacement (HMR) during dev, and highly compressed production builds.
* **State Management:** Utilizes React's native `useState` and `useEffect` hooks. State manages the dynamic form inputs (budget, family size, dietary preferences) and transitions between the "Recipe Generator" and "Saved Recipes" views seamlessly without page reloads.
* **Styling (Glassmorphism):** The UI completely avoids generic component libraries, relying on custom CSS. It utilizes `backdrop-filter: blur(12px)` and semi-transparent `rgba` backgrounds placed over vibrant, high-resolution background imagery to create a premium, modern aesthetic. 
* **API Integration:** The frontend uses standard async `fetch` API calls to communicate with the Render backend. The API base URL is dynamically injected using Vite environment variables (`import.meta.env.VITE_API_URL`), allowing the same codebase to switch automatically between `localhost:10000` during development and the secure Render `https://...` URL in production.

### 2. Orchestration Backend (FastAPI -> Deployed on Render)
**Architecture & Implementation:**
* **Core Framework:** Python FastAPI running on a high-performance ASGI server (Uvicorn). It is deployed on Render's Web Service tier.
* **CORS Configuration:** The `CORSMiddleware` is explicitly configured to whitelist the Vercel frontend domain, preventing Cross-Origin Resource Sharing (CORS) blocks in the browser while securing the API from unauthorized origins.
* **Optimization Engine (`scipy.optimize`):** 
  * The backend dynamically fetches live Mandi market prices (e.g., from a raw GitHub CSV URL).
  * It translates the user's budget and family size into a Linear Programming (LP) mathematical matrix using `scipy.optimize.linprog`. 
  * **Code Logic:** The algorithm defines constraints where the sum product of `(ingredient_weight * market_price)` must strictly be `<= User Budget`, while ensuring minimum caloric and nutritional floors per family member.
  * **Fallback Mechanisms:** If the LP solver fails (i.e., the user's budget is mathematically too low for current market prices), the code triggers sequential fallbacks (Super Fallback -> Nuclear Fallback), aggressively relaxing nutritional constraints until a valid mathematical solution is forced.
* **Asynchronous Execution:** FastAPI's `async def` and `await` syntax are heavily utilized. While the backend waits for the slow Hugging Face AI response or MongoDB network I/O, the server thread is yielded, keeping the application non-blocking and capable of serving multiple concurrent users on a single Render instance.

### 3. AI Inference Layer (Llama 3 -> Deployed on Hugging Face Spaces)
**Architecture & Implementation:**
* **Model Deployment:** The deployment utilizes a merged Meta Llama 3 (3B) model (`nd1490/ratatouille-llama3-3b-v8-MERGED`). It runs natively on a Free CPU environment using Hugging Face `transformers` and `torch`.
* **Direct Safetensors Loading:** It explicitly avoids `GGUF` and `llama-cpp-python`. Instead, the `app.py` script loads `.safetensors` weights directly into memory using `torch.bfloat16` (`device_map="cpu"`). This prevents the need for C++ compilation overhead in the container.
* **Integration Logic (`gradio_client`):**
  * The FastAPI backend uses the Python `gradio_client` library to stream prompt payloads to the Hugging Face Space via a hidden REST API route, bypassing the public web GUI.
  * **Prompt Engineering:** The backend code constructs a highly rigid prompt. It injects the exact SciPy-optimized ingredient list and enforces strict output formatting rules using Llama 3 specific tokens like `<|begin_of_text|>` and `<|start_header_id|>assistant`.
  * **Output Sanitization:** LLMs naturally hallucinate conversational padding (e.g., "Here is your recipe!"). The backend code implements aggressive string slicing techniques (`.split()`, `.strip()`) and Regex to surgically remove this padding, ensuring only clean Markdown or JSON is passed back to the frontend.

### 4. Persistence Layer (MongoDB Atlas)
**Architecture & Implementation:**
* **Database Schema:** A flexible NoSQL approach. Documents in the `recipes` collection contain the final generated recipe markdown, the calculated ingredient weights, user metadata, and a UTC timestamp.
* **Driver Integration (`motor`):**
  * The backend connects to the remote Atlas cluster using `motor` (`AsyncIOMotorClient`). This is a critical architectural decision: using synchronous `pymongo` would block FastAPI's async event loop, whereas `motor` allows high-concurrency database I/O.
* **Authentication/User Tracking Code:**
  * To minimize user friction, complex JWT/OAuth authentication flows were intentionally omitted for this deployment. Instead, the system uses a localized "Username Tagging" paradigm.
  * **State Flow:** The frontend prompts the user for a username, saving it persistently in the browser's `localStorage`.
  * **Saving (`POST /save-recipe`):** The backend executes `db.recipes.insert_one(payload)`, appending the username to the document.
  * **Retrieval (`GET /my-recipes/{username}`):** The backend executes `db.recipes.find({"username": requested_username}).sort("created_at", -1)`. This retrieves the user's chronological history, instantly populating the frontend's "Saved Recipes" interface.
