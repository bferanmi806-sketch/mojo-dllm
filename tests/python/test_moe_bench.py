"""The timing recorder must reject partial or invalid benchmark evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "run_moe", Path(__file__).resolve().parents[2] / "bench/run_moe.py"
)
run_moe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_moe)
parse_run = run_moe.parse_run


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"threads": 2, "tokens": 16, "hidden": 256, "ffn": 32,
                         "experts": 4, "top_k": 2, "down_type": 6}
        self.rows = [
            {"kind": "setup", **self.expected},
            *({"kind": "sample", "index": i, "warmup": i == -1,
               "seconds": seconds, "checksum": 42.0, "max_abs": 2.0}
              for i, seconds in ((-1, 3.0), (0, 2.0), (1, 1.0))),
            {"kind": "routing", "token_expert_pairs": 32,
             "active_experts": 4, "max_tokens_per_expert": 12},
        ]

    def parse(self, rows):
        return parse_run("\n".join(json.dumps(row) for row in rows), 2, self.expected)

    def test_keeps_raw_samples_and_excludes_warmup_from_median(self):
        result = self.parse(self.rows)
        self.assertEqual(result["median_seconds"], 1.5)
        self.assertEqual(len(result["samples"]), 3)

    def test_rejects_missing_duplicate_or_changed_shape(self):
        for mutation in ("missing", "duplicate", "shape"):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(self.rows)
                if mutation == "missing":
                    del rows[2]
                elif mutation == "duplicate":
                    rows[2]["index"] = -1
                else:
                    rows[0]["tokens"] = 8
                with self.assertRaises(ValueError):
                    self.parse(rows)

    def test_rejects_nonfinite_changed_or_empty_outputs(self):
        for key, value in (("seconds", float("nan")), ("seconds", 0),
                           ("checksum", 43.0), ("max_abs", 0)):
            with self.subTest(key=key, value=value):
                rows = copy.deepcopy(self.rows)
                rows[2][key] = value
                with self.assertRaises(ValueError):
                    self.parse(rows)

    def test_rejects_dropped_pairs_and_overfull_expert(self):
        for key, value in (("token_expert_pairs", 31), ("max_tokens_per_expert", 17)):
            with self.subTest(key=key):
                rows = copy.deepcopy(self.rows)
                rows[-1][key] = value
                with self.assertRaises(ValueError):
                    self.parse(rows)


if __name__ == "__main__":
    unittest.main()
