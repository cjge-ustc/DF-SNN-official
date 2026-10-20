"""Launch one frozen configuration without scheduling other GPUs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def nonempty_path(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("path is empty; set the dataset environment variable first")
    return Path(value)


def main():
    configs = json.loads((ROOT / "configs" / "experiments.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=sorted(configs))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--data-path", type=nonempty_path)
    parser.add_argument("--output-dir", type=nonempty_path)
    parser.add_argument("--resume", type=nonempty_path)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(sorted(configs)))
        return 0
    if not args.experiment or args.data_path is None or args.output_dir is None:
        parser.error("experiment, --data-path and --output-dir are required")
    config = configs[args.experiment]
    source = ROOT / "examples" / config["snapshot"]
    command = [sys.executable, "-u", str(source / "train_ablation.py")]
    replacements = {"{DATA}": str(args.data_path.expanduser().resolve()),
                    "{OUTPUT}": str(args.output_dir.expanduser().resolve())}
    command += [replacements.get(value, value) for value in config["argv"]]
    if args.resume is not None:
        command += ["--resume", str(args.resume.expanduser().resolve())]
    if args.test_only:
        if args.resume is None:
            parser.error("--test-only requires --resume")
        command.append("--test-only")
    print(shlex.join(command), flush=True)
    if args.print_command:
        return 0
    if not (source / "train_ablation.py").is_file():
        parser.error("training source is missing; use a complete repository checkout")
    if not args.data_path.expanduser().is_dir():
        parser.error("--data-path must be an existing dataset directory")
    if args.resume is not None and not args.resume.expanduser().is_file():
        parser.error("--resume must be an existing trusted checkpoint")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not output.is_dir():
        parser.error("--output-dir must be a directory, not a file")
    if not args.test_only and args.resume is None and output.exists() and any(output.iterdir()):
        parser.error("output is not empty; choose a new directory or explicitly --resume")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(name, "1")
    return subprocess.call(command, cwd=source, env=env)


if __name__ == "__main__":
    sys.exit(main())
