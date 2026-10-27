import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from backend.app import create_app
from backend.model_deployment import FORMAT_VERSION, ModelService, load_model, save_model, train_model

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = {"monthly_budget": 1500, "spent_so_far": 700, "days_remaining": 12, "avg_daily_spending": 55, "previous_period_spending": 1400}


class TestDeployment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "model.joblib")

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_predictions_match(self):
        for opt, act in (("adam", "relu"), ("sgd", "tanh")):
            net, scaler, meta = train_model(opt, act, 0.05, 20)
            X = np.array([list(SAMPLE.values())], dtype=float)
            before = net.predict_proba(scaler.transform(X))
            save_model(self.path, net, scaler, meta)
            net2, scaler2, meta2 = load_model(self.path)
            after = net2.predict_proba(scaler2.transform(X))
            self.assertTrue(np.allclose(before, after, atol=1e-10))
            self.assertEqual((meta2["optimizer"], net2.activation), (opt, act))

    def test_file_contents_are_versioned_dict(self):
        net, scaler, meta = train_model("adam", "relu", 0.01, 5)
        save_model(self.path, net, scaler, meta)
        d = joblib.load(self.path)
        self.assertEqual(d["format_version"], FORMAT_VERSION)
        for key in ("architecture", "hidden_activation", "params", "feature_order", "scaler", "metadata"):
            self.assertIn(key, d)
        self.assertEqual(set(d["params"]), {"W1", "b1", "W2", "b2"})
        self.assertEqual(d["architecture"]["n_hidden"], 8)

    def test_fresh_process_load_matches(self):
        svc = ModelService(self.path)
        svc.train(optimizer="adam", activation="relu", learning_rate=0.01, epochs=20)
        expected = svc.predict(SAMPLE)
        svc.save()
        code = ("import sys;from backend.model_deployment import ModelService;"
                f"s=ModelService({self.path!r});s.load();print(repr(s.predict({SAMPLE!r})))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertAlmostEqual(float(out.stdout.strip()), expected, places=9)

    def test_bad_files(self):
        with self.assertRaises(FileNotFoundError):
            load_model(self.path)
        joblib.dump({"format_version": 999}, self.path)
        with self.assertRaises(ValueError):
            load_model(self.path)

    def test_api_train_save_load_predict(self):
        c = create_app(os.path.join(self.tmp.name, "a.db"), self.path).test_client()
        c.post("/api/train", json={"epochs": 20})
        p1 = c.post("/api/predict", json=SAMPLE).get_json()["probability"]
        self.assertEqual(c.post("/api/model/save").status_code, 200)
        c2 = create_app(os.path.join(self.tmp.name, "b.db"), self.path).test_client()   # clean state
        self.assertFalse(c2.get("/api/status").get_json()["model"]["trained"])
        self.assertTrue(c2.get("/api/status").get_json()["model"]["model_file_exists"])
        self.assertEqual(c2.post("/api/model/load").status_code, 200)
        p2 = c2.post("/api/predict", json=SAMPLE).get_json()["probability"]
        self.assertAlmostEqual(p1, p2, places=9)

    def test_corrupt_file_api(self):
        Path(self.path).write_bytes(b"garbage")
        c = create_app(os.path.join(self.tmp.name, "a.db"), self.path).test_client()
        self.assertEqual(c.post("/api/model/load").status_code, 422)

    def test_training_metrics_are_real(self):
        _, _, meta = train_model("adam", "relu", 0.01, 60)
        self.assertGreater(meta["metrics"]["accuracy"], 0.8)
        self.assertTrue(meta["weights_updated"])
        self.assertEqual(sum(meta["split"].values()), 3000)


if __name__ == "__main__":
    unittest.main()
