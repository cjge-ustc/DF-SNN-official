import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_source_integrity(self):
        manifest = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text())
        self.assertEqual(len(manifest), 52)
        for name, metadata in manifest.items():
            with self.subTest(source=name):
                data = (ROOT / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), metadata["export_sha256"])
                ast.parse(data, filename=name)
                self.assertNotIn(b"spikeDE_dis2", data)

    def test_eight_launch_commands(self):
        configs = json.loads((ROOT / "configs" / "experiments.json").read_text())
        self.assertEqual(len(configs), 8)
        with tempfile.TemporaryDirectory() as temp:
            for name, config in configs.items():
                with self.subTest(experiment=name):
                    source = ROOT / "examples" / config["snapshot"]
                    self.assertTrue((source / "train_ablation.py").is_file())
                    run = subprocess.run(
                        [sys.executable, str(ROOT / "run_experiment.py"), name,
                         "--data-path", temp, "--output-dir", str(Path(temp) / "output"),
                         "--print-command"], capture_output=True, text=True, check=True,
                    )
                    self.assertIn(str(source / "train_ablation.py"), run.stdout)
                    self.assertIn("--policy-json", run.stdout)
                    self.assertNotIn("{DATA}", run.stdout)
                    self.assertNotIn("{OUTPUT}", run.stdout)
                    self.assertFalse((Path(temp) / "output").exists())

    def test_local_imports_resolve(self):
        for directory in (ROOT / "examples").iterdir():
            if not directory.is_dir():
                continue
            local = {file.stem for file in directory.glob("*.py")}
            required = {"train_transformer", "ablation_models", "ablation_io", "model",
                        "aligned_glmulti", "autoaugment", "utils", "hardvs", "dvs_lip"}
            self.assertTrue(required.issubset(local), directory.name)
        self.assertTrue((ROOT / "spikeDE_dis" / "__init__.py").is_file())

    def test_resume_and_evaluation_flags(self):
        command = [sys.executable, str(ROOT / "run_experiment.py"), "spikformer_n101",
                   "--data-path", "/tmp/data", "--output-dir", "/tmp/output",
                   "--print-command", "--test-only"]
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        run = subprocess.run(command + ["--resume", "/tmp/trusted.pth"],
                             capture_output=True, text=True, check=True)
        self.assertIn("--test-only", run.stdout)
        self.assertIn("--resume " + str(Path("/tmp/trusted.pth").resolve()), run.stdout)

    def test_empty_paths_rejected(self):
        for flag in ("--data-path", "--output-dir", "--resume"):
            command = [sys.executable, str(ROOT / "run_experiment.py"), "cnn128_hardvs",
                       "--data-path", "/tmp/data", "--output-dir", "/tmp/output",
                       "--print-command", flag, ""]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("path is empty", result.stderr)

    def test_output_file_rejected_without_importing_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "file"
            output.touch()
            result = subprocess.run(
                [sys.executable, str(ROOT / "run_experiment.py"), "cnn128_hardvs",
                 "--data-path", temp, "--output-dir", str(output)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not a file", result.stderr)


if __name__ == "__main__":
    unittest.main()
