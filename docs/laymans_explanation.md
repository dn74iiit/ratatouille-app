# Ratatouille: The Elevator Pitch

*Use this document when explaining the project to non-technical audiences, friends, or family.*

### What is Ratatouille?
Ratatouille is an AI-powered smart kitchen assistant. You tell it what ingredients you have in your fridge, how much money you want to spend, and what state you live in. In seconds, it calculates exactly how much of each ingredient you can afford at today's market prices, and then uses Artificial Intelligence to write a custom, perfectly portioned recipe for you.

### How does it work? (The Simple Version)
Think of the app as having three different "brains" that talk to each other:

1. **The Interface (React/Vercel):** 
   This is the beautiful website you see on your phone or computer. It’s the waiter that takes your order (your ingredients and budget) and hands it to the kitchen.

2. **The Calculator (Python/Render):** 
   This is the manager in the back office. Before any cooking happens, it checks a real-world database of vegetable prices in India (Mandi prices). It uses advanced math to figure out exactly how many grams of chicken or potato you can afford without going over your budget.

3. **The Master Chef (Hugging Face AI):** 
   Once the Calculator figures out the exact amounts, it hands the list to the AI Chef. The AI acts just like a human chef—it looks at the ingredients and writes a brand new, step-by-step recipe from scratch, ensuring everything is cooked perfectly.

4. **The Filing Cabinet (MongoDB):**
   If you like a recipe, you can save it! We attached a cloud database to the system that acts like a digital recipe book. You just type your name, and it saves your custom recipe forever so you can view it later.

### Why is this impressive?
Most AI apps just ask ChatGPT to write a recipe, which costs money and often ignores real-world math. We built our **own** AI brain and our **own** math calculator from scratch, connected them together, and hosted them on the internet for completely free!
