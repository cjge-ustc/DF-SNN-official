"""CPU-only dependency, preset/model construction, and checkpoint checks (no training)."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check_one(name, config):
    import cupy
    import torch
    from timm.optim import create_optimizer
    from timm.scheduler import create_scheduler

    torch.set_num_threads(1)
    source = ROOT / "examples" / config["snapshot"]
    sys.path[:0] = [str(source), str(ROOT)]
    import train_transformer as training
    from ablation_io import load_model, rng_state, restore_rng
    from ablation_models import effective_masses

    # Intercept only the training entry; execute the real parser and model factory.
    training.main = lambda args: None
    with tempfile.TemporaryDirectory() as temp:
        replacements = {"{DATA}": temp, "{OUTPUT}": temp}
        sys.argv = [str(source / "train_ablation.py")] + [
            replacements.get(arg, arg) for arg in config["argv"]
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")
        args = training.args
        args.device = "cpu"
        model = training.create_model(args.model)
        assert torch.isfinite(effective_masses(model)).all()
        assert len([p for n, p in model.named_parameters() if "multi_coefficient" in n]) == 1
        optimizer = create_optimizer(args, model)
        scheduler, epochs = create_scheduler(args, optimizer)
        assert epochs == args.epochs
        train_gen, test_gen = torch.Generator(), torch.Generator()
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          lr_scheduler=scheduler.state_dict(), rng=rng_state(train_gen, test_gen))
        path = Path(temp) / "check.pth"
        torch.save(checkpoint, path)
        restored = torch.load(path, map_location="cpu", weights_only=False)
        load_model(model, restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["lr_scheduler"])
        restore_rng(restored["rng"], train_gen, test_gen)
        assert not torch.cuda.is_initialized(), "CPU check must not initialize CUDA"
        print("RUNTIME_OK", name, "classes=", args.num_classes, "T=", args.T,
              "epochs=", epochs, "parameters=", sum(p.numel() for p in model.parameters()),
              "cupy=", cupy.__version__)


def main():
    configs = json.loads((ROOT / "configs/experiments.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(configs))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    if args.experiment:
        check_one(args.experiment, configs[args.experiment])
    else:
        for name in sorted(configs):
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--experiment", name], check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
