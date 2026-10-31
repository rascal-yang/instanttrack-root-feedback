import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_published_numbers_recompute_from_per_seed_evidence(self):
        spec = importlib.util.spec_from_file_location("report_results", ROOT / "tools/report_results.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.recompute()
        self.assertEqual(result["root_position"]["completed_clips_per_seed"], [40, 40, 40])

    def test_training_entry_dry_run_selects_root_and_explicit_baseline(self):
        for flags, expected in (([], "root"), (["--baseline"], "none")):
            output = subprocess.check_output([sys.executable, str(ROOT / "scripts/train_root.py"),
                "--dry-run", "--seed", "2", *flags], text=True)
            args = json.loads(output)["arguments"]
            self.assertEqual(args[args.index("--position_obs") + 1], expected)
            self.assertEqual(args[args.index("--iterations") + 1], "6000")
            self.assertEqual(args[args.index("--seed") + 1], "2")


if __name__ == "__main__":
    unittest.main()
