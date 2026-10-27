import re
import tempfile
import unittest
from pathlib import Path

from backend.app import create_app
from backend.synthetic_data import FEATURES

FRONT = Path(__file__).resolve().parent.parent / "frontend"


class TestContract(unittest.TestCase):
    def test_frontend_paths_exist_in_backend(self):
        js = (FRONT / "script.js").read_text()
        called = set(re.findall(r'api\(\s*(?:id \? )?"(/[a-z/]+\??)', js)) | set(re.findall(r'"(/expenses/)"', js))
        self.assertGreaterEqual(len(called), 10)
        with tempfile.TemporaryDirectory() as d:
            rules = [re.sub(r"<[^>]+>", "", r.rule) for r in create_app(d + "/a.db", d + "/m").url_map.iter_rules()]
        for path in called:
            self.assertTrue(any(r.startswith("/api" + path.rstrip("?")) for r in rules), path)

    def test_json_fields_match(self):
        js = (FRONT / "script.js").read_text()
        for f in FEATURES + ["monthly_budget", "optimizer", "activation", "learning_rate", "epochs", "amount", "date", "category", "description"]:
            self.assertIn(f, js, f)
        html = (FRONT / "index.html").read_text()
        for f in FEATURES:
            self.assertIn(f'id="p-{f}"', html)
        for el in re.findall(r"\$\(\"([\w-]+)\"\)", js):
            self.assertIn(f'id="{el}"', html, el)


if __name__ == "__main__":
    unittest.main()
