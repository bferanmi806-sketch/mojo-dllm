"""Keep independent checkpoint results from overwriting or hiding each other."""

import datetime as dt
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parity_run = load_module("parity_run", ROOT / "bench/parity.py")
tables = load_module("parity_tables", ROOT / "scripts/bench_table.py")


class ParityEvidenceTests(unittest.TestCase):
    def test_result_paths_distinguish_models_and_devices_on_the_same_day(self):
        day = dt.date(2026, 10, 30)
        a = parity_run.result_path("models/Dream-Coder.Q4_K_M.gguf", "dream", "cpu", day)
        b = parity_run.result_path("models/DiffuCoder.Q4_K_M.gguf", "dream", "cpu", day)
        gpu = parity_run.result_path("models/Dream-Coder.Q4_K_M.gguf", "dream", "gpu", day)
        self.assertEqual(a.name, "2026-10-30-dream-Dream-Coder.Q4_K_M.json")
        self.assertEqual(gpu.name, "2026-10-30-dream-Dream-Coder.Q4_K_M-gpu.json")
        self.assertEqual(len({a, b, gpu}), 3)
        self.assertEqual(a.parent, ROOT / "bench/parity")

    def test_two_same_day_cli_runs_save_both_checkpoints(self):
        # Exercise the writer in main(), with inference omitted: the regression
        # happened after the expensive model runs had already finished.
        models = ["Dream-Coder.Q4_K_M.gguf", "DiffuCoder.Q4_K_M.gguf"]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with (
                patch.object(parity_run, "ROOT", root),
                patch.object(parity_run, "LOGIT_CASES", []),
                patch.object(parity_run, "GEN_CASES", []),
                patch.object(parity_run, "run", return_value="test-head"),
                patch.dict(sys.modules, {"numpy": ModuleType("numpy")}),
                redirect_stdout(io.StringIO()),
            ):
                for model in models:
                    with patch.object(sys, "argv", ["parity.py", "--arch", "dream", "--model", model]):
                        self.assertEqual(parity_run.main(), 0)
            files = list((root / "bench/parity").glob("*.json"))
            self.assertEqual(len(files), 2)
            self.assertEqual({json.loads(f.read_text())["model"] for f in files}, set(models))

    def evidence(self, model, arch="llada", device="cpu", date="2026-10-30T10:00:00"):
        doc = json.loads((ROOT / "bench/parity/2026-10-03-llada.json").read_text())
        doc.update(model=model, arch=arch, device=device, date=date)
        return doc

    def write(self, root, name, doc):
        folder = root / "bench/parity"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(json.dumps(doc))

    def test_tables_keep_every_checkpoint_and_device(self):
        cases = [
            ("llada", "cpu", "LLaDA-8B.gguf"),
            ("llada", "cpu", "LLaDA-1.5.gguf"),
            ("dream", "cpu", "Dream-v0.gguf"),
            ("dream", "cpu", "Dream-Coder.gguf"),
            ("dream", "cpu", "DiffuCoder.gguf"),
            ("dream", "gpu", "Dream-v0.gguf"),
        ]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for i, (arch, device, model) in enumerate(cases):
                self.write(root, f"2026-10-30-{i}.json", self.evidence(model, arch, device))
            with patch.object(tables, "ROOT", root):
                logit_table, generation_table = tables.parity()
            for table in (logit_table, generation_table):
                for _, device, model in cases:
                    heading = f"**{model}{', GPU' if device == 'gpu' else ''}**"
                    self.assertEqual(table.count(heading), 1, heading)

    def test_newest_run_uses_recorded_time_not_filename_sort_order(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.write(root, "z-old.json", self.evidence("Dream.gguf", "dream", date="2026-10-30T09:00:00"))
            self.write(root, "a-new.json", self.evidence("Dream.gguf", "dream", date="2026-10-30T10:00:00"))
            with patch.object(tables, "ROOT", root):
                for table in tables.parity():
                    self.assertIn("a-new.json", table)
                    self.assertNotIn("z-old.json", table)
                    self.assertEqual(table.count("**Dream.gguf**"), 1)

    def test_legacy_records_default_to_llada_cpu(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            doc = self.evidence("LLaDA-8B.gguf")
            del doc["arch"]
            del doc["device"]
            self.write(root, "legacy.json", doc)
            with patch.object(tables, "ROOT", root):
                for table in tables.parity():
                    self.assertIn("**LLaDA-8B.gguf**", table)


if __name__ == "__main__":
    unittest.main()
