"""Read-only check of an existing frame cache; never download or preprocess data."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    configs = json.loads((ROOT / "configs/experiments.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(configs))
    parser.add_argument("--data-path", required=True, type=Path)
    args = parser.parse_args()
    config = configs[args.experiment]
    argv = config["argv"]
    steps = int(argv[argv.index("--T") + 1])
    classes = int(argv[argv.index("--num-classes") + 1])
    dataset = argv[argv.index("--dataset") + 1]
    root = args.data_path.expanduser().resolve()
    cache = root / f"frames_number_{steps}_split_by_number"
    if not (root / "events_np").is_dir() or not cache.is_dir():
        parser.error(f"prepare events_np/ and {cache.name}/ first; no files were changed")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path[:0] = [str(ROOT / "examples" / config["snapshot"]), str(ROOT)]
    import numpy as np
    import torch
    import train_transformer as training
    torch.set_num_threads(1)
    replacements = {"{DATA}": str(root), "{OUTPUT}": str(root)}
    sys.argv = [sys.argv[0]] + [replacements.get(item, item) for item in argv]
    # The entry script consumes these two arguments before the shared parser.
    for flag in ("--implementation", "--policy-json"):
        index = sys.argv.index(flag)
        del sys.argv[index:index + 2]
    parsed = training.parse_args()
    train, test = training.build_datasets(parsed)
    expected = {"hardvs": (64519, 32386), "dvs128gesture": (1176, 288),
                "ncaltech101": (6967, 1742), "dvslips": (14896, 4975)}
    assert (len(train), len(test)) == expected[dataset], (len(train), len(test))
    for split in (train, test):
        for index in sorted({0, len(split) - 1}):
            frame, label = split[index]
            assert frame.ndim == 4 and frame.shape[:2] == (steps, 2), frame.shape
            assert np.isfinite(frame).all() and 0 <= int(label) < classes
    assert not torch.cuda.is_initialized()
    print("DATA_OK", args.experiment, "train=", len(train), "test=", len(test),
          "T=", steps, "sampled first/last item per split (not full inference)")


if __name__ == "__main__":
    main()
