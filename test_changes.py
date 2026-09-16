import sys
sys.path.insert(0, '.')
import vegan_engine
from api import canonicalize_ingredient

print('--- Test 1: Keyword Detection ---')
res1 = vegan_engine.classify_by_keyword('egg')
print(f"egg: {res1['culinary_role'] if res1 else None}")

res2 = vegan_engine.classify_by_keyword('egg yolk')
print(f"egg yolk: {res2['culinary_role'] if res2 else None}")

res3 = vegan_engine.classify_by_keyword('eggplant')
print(f"eggplant (should be None): {res3}")

print('\n--- Test 2: Canonical Normalization ---')
tests = ['chicken breast', 'egg yolk', 'eggs', 'heavy cream', 'lamb chop', 'garlic', 'eggplant']
for t in tests:
    print(f"{t:22} -> {canonicalize_ingredient(t)}")
