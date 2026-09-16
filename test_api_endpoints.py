import unittest
import unittest.mock
from fastapi.testclient import TestClient

# Import the FastAPI app from api
from api import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_get_vegan_blueprint_single(self):
        payload = {
            "ingredient": "paneer",
            "archetype": "Curry"
        }
        response = self.client.post("/get-vegan-blueprint", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["results"]), 1)
        
        result = data["results"][0]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["original_ingredient"], "paneer")
        self.assertEqual(result["best_vegan_substitute"], "tofu")
        self.assertIn("compensation_blueprint", result)

    def test_get_vegan_blueprint_multiple(self):
        payload = {
            "ingredients": ["paneer", "chicken", "honey"],
            "archetype": "Salad"
        }
        response = self.client.post("/get-vegan-blueprint", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["results"]), 3)
        
        # Check first result (paneer)
        self.assertEqual(data["results"][0]["best_vegan_substitute"], "tofu")
        # Check second result (chicken)
        self.assertEqual(data["results"][1]["best_vegan_substitute"], "soya chunks")
        # Check third result (honey)
        self.assertEqual(data["results"][2]["best_vegan_substitute"], "maple syrup")

    @unittest.mock.patch("api.bootstrap_ingredient_profile")
    def test_get_vegan_blueprint_dynamic_bootstrap(self, mock_bootstrap):
        # Mock successful LLM bootstrapping
        mock_bootstrap.return_value = {
            "is_vegan": False,
            "macros": { "fat": 0.1, "protein": 0.2, "carb": 0.0, "water": 0.7 },
            "texture": [3, 4, 4, 3, 2, 1],
            "flavor_molecules": ["hydrogen_sulfide", "methyl_furan"],
            "role": "bulk_protein"
        }
        
        payload = {
            "ingredient": "ostrich",
            "archetype": "Curry"
        }
        response = self.client.post("/get-vegan-blueprint", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        result = data["results"][0]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["original_ingredient"], "ostrich")
        self.assertEqual(result["best_vegan_substitute"], "soya chunks")
        
        # Verify it got cached in the JSON database
        import vegan_engine
        import json
        features = vegan_engine.load_features()
        self.assertIn("ostrich", features)
        
        # Clean up
        try:
            del features["ostrich"]
            with open(vegan_engine.DB_PATH, "w") as f:
                json.dump(features, f, indent=2)
        except Exception as e:
            pass

    @unittest.mock.patch("api.bootstrap_ingredient_profile")
    def test_get_vegan_blueprint_fallback(self, mock_bootstrap):
        # Mock failed/offline LLM bootstrapping
        mock_bootstrap.return_value = None
        
        payload = {
            "ingredient": "quail",
            "archetype": "Curry"
        }
        response = self.client.post("/get-vegan-blueprint", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        result = data["results"][0]
        
        # Should fall back to static 'poultry' template and return soya chunks
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["original_ingredient"], "quail")
        self.assertEqual(result["best_vegan_substitute"], "soya chunks")
        
        # Verify it was NOT cached
        import vegan_engine
        features = vegan_engine.load_features()
        self.assertNotIn("quail", features)

if __name__ == "__main__":
    import unittest.mock
    unittest.main()
