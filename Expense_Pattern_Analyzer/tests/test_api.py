import os
import tempfile
import unittest

from backend.app import create_app


class ApiBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "t.db")
        self.model = os.path.join(self.tmp.name, "m.joblib")
        self.c = create_app(self.db, self.model).test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, **kw):
        d = {"date": "2026-10-05", "amount": 100, "category": "Transport", "description": "x"}
        d.update(kw)
        return self.c.post("/api/expenses", json=d)


class TestExpenses(ApiBase):
    def test_invalid_inputs_rejected(self):
        for bad in ({"amount": -5}, {"amount": 0}, {"amount": "abc"}, {"amount": True}, {"amount": float("nan")},
                    {"amount": 1e12}, {"date": "2026-13-45"}, {"date": ""}, {"category": ""}, {"category": "Nope"},
                    {"description": "x" * 300}):
            self.assertEqual(self.add(**bad).status_code, 400, bad)
        self.assertEqual(self.c.post("/api/expenses", data="notjson", content_type="text/plain").status_code, 400)
        self.assertEqual(self.c.get("/api/expenses").get_json()["count"], 0)

    def test_crud(self):
        r = self.add(amount=12.345)
        self.assertEqual(r.status_code, 201)
        eid = r.get_json()["id"]
        self.assertEqual(self.c.get("/api/expenses").get_json()["count"], 1)
        u = self.c.put(f"/api/expenses/{eid}", json={"date": "2026-10-06", "amount": 20, "category": "Health", "description": "doc"})
        self.assertEqual(u.get_json()["category"], "Health")
        self.assertEqual(self.c.put("/api/expenses/999", json={"date": "2026-10-06", "amount": 1, "category": "Health"}).status_code, 404)
        self.assertEqual(self.c.delete(f"/api/expenses/{eid}").status_code, 200)
        self.assertEqual(self.c.delete(f"/api/expenses/{eid}").status_code, 404)

    def test_search_filter_sort(self):
        self.add(amount=10, description="Coffee beans", category="Groceries")
        self.add(amount=300, description="Bus pass", category="Transport")
        g = lambda q: [e["amount"] for e in self.c.get("/api/expenses?" + q).get_json()["expenses"]]
        self.assertEqual(g("q=coffee"), [10])
        self.assertEqual(g("q=transport"), [300])
        self.assertEqual(g("category=Groceries"), [10])
        self.assertEqual(g("sort=amount&order=desc"), [300, 10])
        self.assertEqual(g("q=%25"), [])           # wildcard is escaped
        self.assertEqual(self.c.get("/api/expenses?sort=id;drop").status_code, 400)


class TestDashboardBudget(ApiBase):
    def test_empty_dashboard(self):
        d = self.c.get("/api/dashboard").get_json()
        self.assertEqual((d["total_spent"], d["transaction_count"], d["top_category"]), (0, 0, None))
        self.assertIsNone(d["utilization_percent"])

    def test_totals_match_records(self):
        self.add(amount=100, category="Food and dining"); self.add(amount=50.5, category="Transport")
        self.add(amount=25, category="Transport", date="2026-09-10")
        d = self.c.get("/api/dashboard?month=2026-10").get_json()
        self.assertEqual(d["total_spent"], 150.5)
        self.assertEqual(d["transaction_count"], 2)
        self.assertEqual(d["top_category"]["category"], "Food and dining")
        self.assertAlmostEqual(sum(c["percent"] for c in d["category_totals"]), 100, delta=0.2)
        self.assertEqual(sum(x["total"] for x in d["daily"]), 150.5)
        self.assertEqual(d["predictor_defaults"]["previous_period_spending"], 25)
        self.c.post("/api/budget", json={"monthly_budget": 100})
        d = self.c.get("/api/dashboard?month=2026-10").get_json()
        self.assertEqual(d["remaining"], -50.5)
        self.assertEqual(d["utilization_percent"], 150.5)
        self.assertEqual(self.c.get("/api/dashboard?month=bad").status_code, 400)

    def test_budget_validation_and_persistence(self):
        for bad in (0, -1, "x", None, True):
            self.assertEqual(self.c.post("/api/budget", json={"monthly_budget": bad}).status_code, 400)
        self.c.post("/api/budget", json={"monthly_budget": 2500})
        self.add()
        c2 = create_app(self.db, self.model).test_client()     # "restart"
        self.assertEqual(c2.get("/api/budget").get_json()["monthly_budget"], 2500)
        self.assertEqual(c2.get("/api/expenses").get_json()["count"], 1)


class TestModelApi(ApiBase):
    GOOD = {"monthly_budget": 1000, "spent_so_far": 400, "days_remaining": 10, "avg_daily_spending": 40, "previous_period_spending": 900}

    def test_predict_without_model(self):
        self.assertEqual(self.c.post("/api/predict", json=self.GOOD).status_code, 409)
        self.assertEqual(self.c.post("/api/model/save").status_code, 409)
        self.assertEqual(self.c.post("/api/model/load").status_code, 404)
        self.assertFalse(self.c.get("/api/status").get_json()["model"]["trained"])

    def test_invalid_training_and_prediction(self):
        for bad in ({"optimizer": "x"}, {"activation": "x"}, {"learning_rate": -1}, {"learning_rate": 5},
                    {"epochs": 0}, {"epochs": 99999}, {"epochs": 2.5}):
            self.assertEqual(self.c.post("/api/train", json=bad).status_code, 400, bad)
        for k in self.GOOD:
            for v in (None, "abc", -1):
                self.assertEqual(self.c.post("/api/predict", json={**self.GOOD, k: v}).status_code, 400, (k, v))
        self.assertEqual(self.c.post("/api/predict", json={**self.GOOD, "days_remaining": 40}).status_code, 400)

    def test_train_predict(self):
        r = self.c.post("/api/train", json={"epochs": 30, "optimizer": "sgd", "activation": "tanh", "learning_rate": 0.1})
        self.assertEqual(r.status_code, 200)
        m = r.get_json()["model"]
        self.assertEqual(len(m["loss_history"]), 30)
        self.assertLess(m["final_loss"], m["initial_loss"])
        p = self.c.post("/api/predict", json=self.GOOD).get_json()
        self.assertTrue(0 <= p["probability"] <= 1)
        self.assertIn(p["risk"], ("low", "moderate", "high"))
        self.assertIn("synthetic", p["disclaimer"])

    def test_unknown_route_json(self):
        r = self.c.get("/api/nothing")
        self.assertEqual((r.status_code, "error" in r.get_json()), (404, True))


if __name__ == "__main__":
    unittest.main()
