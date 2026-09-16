import unittest
import os
import json
import numpy as np

import vegan_engine

class TestVeganEngine(unittest.TestCase):
    def setUp(self):
        self.features = vegan_engine.load_features()

    def test_database_loads(self):
        self.assertIn("paneer", self.features)
        self.assertIn("tofu", self.features)
        self.assertIn("chicken", self.features)
        self.assertIn("soya chunks", self.features)
        self.assertFalse(self.features["paneer"]["is_vegan"])
        self.assertTrue(self.features["tofu"]["is_vegan"])

    def test_jaccard_similarity(self):
        set_a = {"diacetyl", "butanoic_acid"}
        set_b = {"diacetyl", "lactic_acid"}
        # intersection = 1 (diacetyl), union = 3 (diacetyl, butanoic_acid, lactic_acid)
        # sim = 1/3 = 0.3333
        self.assertAlmostEqual(vegan_engine.jaccard_similarity(set_a, set_b), 1/3)

    def test_texture_similarity(self):
        # Test identical vectors
        self.assertAlmostEqual(vegan_engine.texture_similarity([3,3,1,2,3,2], [3,3,1,2,3,2]), 1.0)
        # Test different vectors (should be bounded between 0 and 1)
        sim = vegan_engine.texture_similarity([3,3,1,2,3,2], [2,2,0,4,2,1])
        self.assertTrue(0.0 <= sim <= 1.0)

    def test_paneer_to_tofu_blueprint(self):
        blueprint = vegan_engine.generate_vegan_blueprint("paneer", "Curry")
        self.assertEqual(blueprint["status"], "success")
        self.assertEqual(blueprint["original_ingredient"], "paneer")
        self.assertEqual(blueprint["best_vegan_substitute"], "tofu")
        
        # Check delta values
        self.assertGreater(blueprint["chemical_delta"]["lipid_deficit"], 0.1) # Paneer fat 0.22 - Tofu fat 0.04 = 0.18
        
        # Check text recommendations (guardrails)
        comp = blueprint["compensation_blueprint"]
        
        # We should have a lipid addition recommendation
        has_fat_recom = any("oil" in add["name"] for add in comp["auxiliary_additions"])
        self.assertTrue(has_fat_recom)
        
        # We should have a moisture pressing technique
        has_press = any("press" in tech.lower() for tech in comp["techniques"])
        self.assertTrue(has_press)
        
        # We should have a dairy-like flavor yeast recommendation
        has_yeast = any("yeast" in add["name"].lower() for add in comp["auxiliary_additions"])
        self.assertTrue(has_yeast)

    def test_chicken_to_soya_chunks_blueprint(self):
        blueprint = vegan_engine.generate_vegan_blueprint("chicken", "Curry")
        self.assertEqual(blueprint["status"], "success")
        self.assertEqual(blueprint["original_ingredient"], "chicken")
        self.assertEqual(blueprint["best_vegan_substitute"], "soya chunks")
        
        # Soya chunks should have a hot water soaking technique recommendation
        comp = blueprint["compensation_blueprint"]
        has_soak = any("soak" in tech.lower() for tech in comp["techniques"])
        self.assertTrue(has_soak)
        
        # Smoked paprika or spice bridges should be recommended
        self.assertGreater(len(comp["spice_bridge"]), 0)
        self.assertEqual(comp["spice_bridge"][0]["spice"], "smoked paprika")

    def test_shrimp_to_king_oyster_blueprint(self):
        blueprint = vegan_engine.generate_vegan_blueprint("shrimp", "Curry")
        self.assertEqual(blueprint["status"], "success")
        self.assertEqual(blueprint["original_ingredient"], "shrimp")
        self.assertEqual(blueprint["best_vegan_substitute"], "king oyster mushroom")
        
        comp = blueprint["compensation_blueprint"]
        has_cross_hatch = any("cross-hatch" in tech.lower() for tech in comp["techniques"])
        self.assertTrue(has_cross_hatch)

    def test_mutton_fallback_blueprint(self):
        blueprint = vegan_engine.generate_vegan_blueprint("mutton", "Curry")
        self.assertEqual(blueprint["status"], "success")
        self.assertEqual(blueprint["original_ingredient"], "mutton")
        self.assertEqual(blueprint["best_vegan_substitute"], "soya chunks")
        self.assertGreater(blueprint["chemical_delta"]["lipid_deficit"], 0.1)

    def test_duck_fallback_blueprint(self):
        blueprint = vegan_engine.generate_vegan_blueprint("duck", "Curry")
        self.assertEqual(blueprint["status"], "success")
        self.assertEqual(blueprint["original_ingredient"], "duck")
        self.assertEqual(blueprint["best_vegan_substitute"], "soya chunks")

    def test_unmatched_fallback_blueprint(self):
        blueprint = vegan_engine.generate_vegan_blueprint("xyz_unknown", "Curry")
        self.assertEqual(blueprint["status"], "error")
        self.assertIn("not found in the database", blueprint["message"])

if __name__ == "__main__":
    unittest.main()
