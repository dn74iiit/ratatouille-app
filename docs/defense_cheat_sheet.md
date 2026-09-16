# Capstone Defense: Q&A Cheat Sheet

Use this sheet to quickly answer technical questions during your project defense or viva.

### Q: Why did you separate the Frontend (Vercel) and Backend (Render)?
**A:** "We separated the frontend and backend to follow modern microservice architecture. React is entirely static, so Vercel can cache it globally on their Edge Network for instant load times. Render is used strictly as an API orchestration layer because it supports Python and heavy computation like SciPy."

### Q: How does the AI generate recipes? Why didn't you just use the OpenAI API?
**A:** "Instead of paying for OpenAI, we built our own localized inference pipeline. We quantized a 6GB Meta Llama 3 8B model to run on a Hugging Face Space. Our backend communicates with it via the Gradio API. This proves we can build entirely free, open-source AI pipelines."

### Q: How does the Cost-Optimization work? What happens if the budget is too low?
**A:** "We use `scipy.optimize.linprog` to perform Linear Programming. It takes real-time Mandi market prices and creates boundaries (e.g., maximum 400g of potato). If the user provides an impossibly low budget, our system catches the `ValueError` and activates a 'Super Fallback' that dynamically lowers the required boundaries to 5g, ensuring the math *always* returns a feasible solution."

### Q: How are user recipes saved without passwords or a complex login system?
**A:** "To optimize for prototype speed, we implemented a 'Username Tagging' system using MongoDB Atlas and the `motor` asynchronous Python driver. When a recipe is saved, it is attached to a string identifier (the username). The React app then uses `fetch()` to query MongoDB for all documents matching that tag."

### Q: Why does the system sometimes take a few minutes to generate the first recipe?
**A:** "Because we are utilizing free-tier cloud infrastructure, the Hugging Face GPU space 'spins down' when idle. The 3-minute delay is the 'Cold Start' phase where the 6GB Llama 3 model is being loaded into server RAM. Subsequent requests take less than 30 seconds."

### Q: How do you prevent the AI from hallucinating or rambling?
**A:** "We implemented strict Prompt Engineering using the `<|begin_of_text|>` BOS token. Furthermore, the Python backend intercepts the AI's raw output and uses aggressive text-slicing (`ai_text.split()`) to cut off standard Llama 3 hallucinations like 'Bon Appetit!' or 'Variations:'."
