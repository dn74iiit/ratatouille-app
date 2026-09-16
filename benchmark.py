import time
import asyncio
from api import generate_recipe
from pydantic import BaseModel

class RecipeRequest(BaseModel):
    ingredients: list[str]
    is_vegan: bool = False
    budget: float = 200.0
    servings: int = 2
    state: str = "Maharashtra"
    model_version: str = "v10"

request = RecipeRequest(ingredients=["chicken", "onion", "tomato"], is_vegan=True)

async def run():
    start = time.time()
    resp = generate_recipe(request)
    async for chunk in resp.body_iterator:
        pass
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")

asyncio.run(run())
