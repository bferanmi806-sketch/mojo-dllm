"""The GPU step-time ratio line is computed from the runs, not typed."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tables = load_module("bench_table_ratio", ROOT / "scripts/bench_table.py")


def run(runtime, rep, prompt, ms, ok=True):
    return dict(runtime=runtime, rep=rep, prompt_index=prompt, ms_per_step=ms, ok=ok, cpu_busy_before=0.01)


def summary(ms, runs_ok=6):
    return dict(
        runs_ok=runs_ok, runs_failed=0, ms_per_step_median=ms, tokens_per_s_median=128 / (32 * ms / 1000),
        startup_s_median=2.0, peak_rss_gb_max=1.0, forward_ms_median=ms,
    )


class PairedRatioTests(unittest.TestCase):
    def test_median_range_and_count_over_pairs(self):
        runs = []
        for rep, (mine, theirs) in enumerate([(90, 100), (80, 100), (95, 100)]):
            runs += [run("a", rep, 0, mine), run("b", rep, 0, theirs)]
        got = tables.paired_ratio(runs, "a", "b")
        self.assertEqual((got.median, got.low, got.high, got.pairs, got.below), (0.90, 0.80, 0.95, 3, 3))

    def test_pairs_below_one_are_counted_even_when_they_print_as_one(self):
        runs = [run("a", 0, 0, 99.98), run("b", 0, 0, 100), run("a", 1, 0, 101), run("b", 1, 0, 100)]
        got = tables.paired_ratio(runs, "a", "b")
        self.assertEqual((got.pairs, got.below), (2, 1))
        self.assertEqual("%.3f" % got.low, "1.000")   # the three decimals hide it, the count does not

    def test_a_repeated_rep_and_prompt_is_refused(self):
        runs = [run("a", 0, 0, 10), run("a", 0, 0, 12), run("b", 0, 0, 20)]
        with self.assertRaises(SystemExit):
            tables.paired_ratio(runs, "a", "b")

    def test_numerator_is_the_first_runtime(self):
        runs = [run("a", 0, 0, 50), run("b", 0, 0, 100)]
        self.assertEqual(tables.paired_ratio(runs, "a", "b").median, 0.5)
        self.assertEqual(tables.paired_ratio(runs, "b", "a").median, 2.0)

    def test_pairs_match_on_rep_and_prompt_together(self):
        runs = [run("a", 0, 0, 10), run("b", 0, 0, 20), run("a", 0, 1, 30), run("b", 0, 1, 40)]
        got = tables.paired_ratio(runs, "a", "b")
        self.assertEqual((got.low, got.high, got.pairs), (0.5, 0.75, 2))

    def test_failed_and_unpaired_runs_are_left_out(self):
        runs = [run("a", 0, 0, 10), run("b", 0, 0, 20), run("a", 1, 0, 1, ok=False), run("b", 1, 0, 100), run("a", 2, 0, 5)]
        got = tables.paired_ratio(runs, "a", "b")
        self.assertEqual((got.median, got.pairs), (0.5, 1))

    def test_no_pair_gives_none(self):
        self.assertIsNone(tables.paired_ratio([run("a", 0, 0, 10)], "a", "b"))

    def test_median_is_not_the_mean(self):
        runs = []
        for rep, mine in enumerate([10, 10, 100]):
            runs += [run("a", rep, 0, mine), run("b", rep, 0, 100)]
        self.assertEqual(tables.paired_ratio(runs, "a", "b").median, 0.1)


class RenderedLineTests(unittest.TestCase):
    def result(self, mine, theirs):
        runs = []
        for rep in range(3):
            for prompt in range(2):
                runs += [run("mojo-dllm-gpu", rep, prompt, mine), run("llama.cpp-cuda", rep, prompt, theirs)]
        return dict(
            partial=False, date="2026-11-03T00:00:00",
            machine=dict(gpu="Test GPU, 1.0, 6144 MiB", cpu="Test CPU", power_profile="performance", ram_gb=16),
            versions={"mojo-dllm": "abc", "mojo": "Mojo 1.1.0", "llama.cpp": "def"},
            config=dict(models=dict(gguf="~/models/Test-Q4_K_M.gguf"), gen_length=128, steps=32,
                        block_length=32, threads=12, prompts=["p0", "p1"], busy_threshold=0.1),
            reps=3, runs=runs,
            summary={"mojo-dllm-gpu": summary(mine), "llama.cpp-cuda": summary(theirs)},
        )

    def render(self, doc):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "bench/results").mkdir(parents=True)
            (root / "bench/results/gpu.json").write_text(json.dumps(doc))
            with patch.object(tables, "ROOT", root):
                return tables.render()

    def test_gpu_table_carries_the_ratio_line(self):
        text = self.render(self.result(90, 100))
        self.assertIn("Step time of mojo-dllm over llama.cpp: median 0.900, range 0.900 to 0.900 across 6 paired runs", text)
        self.assertIn("with mojo-dllm's step shorter in 6 of them", text)

    def split(self, doc, names):
        """One result file per runtime, as `run_bench.py --only` writes them."""
        out = []
        for name in names:
            d = json.loads(json.dumps(doc))
            d["runs"] = [r for r in d["runs"] if r["runtime"] == name]
            d["summary"] = {name: d["summary"][name]}
            out.append(d)
        return out

    def render_files(self, docs):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "bench/results").mkdir(parents=True)
            for i, d in enumerate(docs):
                (root / f"bench/results/{i}.json").write_text(json.dumps(d))
            with patch.object(tables, "ROOT", root):
                return tables.render()

    def test_runtimes_from_two_result_files_get_no_ratio(self):
        docs = self.split(self.result(90, 100), ["mojo-dllm-gpu", "llama.cpp-cuda"])
        with self.assertRaises(SystemExit) as cm:
            self.render_files(docs)
        self.assertIn("different result files", str(cm.exception))

    def test_a_table_whose_runtimes_never_finished_together_is_refused(self):
        doc = self.result(90, 100)
        for r in doc["runs"]:
            r["ok"] = (r["rep"] % 2 == 0) == (r["runtime"] == "mojo-dllm-gpu")   # never both in one pair
        doc["summary"]["mojo-dllm-gpu"]["runs_ok"] = 4
        doc["summary"]["llama.cpp-cuda"]["runs_ok"] = 2
        with self.assertRaises(SystemExit) as cm:
            self.render(doc)
        self.assertIn("no run finished in both runtimes", str(cm.exception))

    def test_a_summary_that_disagrees_with_its_runs_is_refused(self):
        doc = self.result(90, 100)
        doc["summary"]["mojo-dllm-gpu"]["ms_per_step_median"] = 1.0
        with self.assertRaises(SystemExit) as cm:
            self.render(doc)
        self.assertIn("ms_per_step_median", str(cm.exception))
        doc = self.result(90, 100)
        doc["summary"]["llama.cpp-cuda"]["runs_ok"] = 5
        with self.assertRaises(SystemExit) as cm:
            self.render(doc)
        self.assertIn("runs_ok", str(cm.exception))

    def test_cpu_tables_do_not_get_one(self):
        doc = self.result(90, 100)
        doc["runs"] = [dict(r, runtime={"mojo-dllm-gpu": "mojo-dllm", "llama.cpp-cuda": "llama.cpp"}[r["runtime"]]) for r in doc["runs"]]
        doc["summary"] = {"mojo-dllm": doc["summary"]["mojo-dllm-gpu"], "llama.cpp": doc["summary"]["llama.cpp-cuda"]}
        self.assertNotIn("Step time of mojo-dllm over", self.render(doc))


if __name__ == "__main__":
    unittest.main()
