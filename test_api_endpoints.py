import unittest
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

if __name__ == "__main__":
    unittest.main()
