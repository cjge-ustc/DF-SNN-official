# Source Layout

The repository separates the shared neuron library from the original experiment drivers.

| Directory | Contents |
|:---|:---|
| `spikeDE_dis/` | Shared neurons, surrogate gradients, solvers, and CuPy kernels |
| `examples/cnn128/` | CNN128 training for all four datasets |
| `examples/spikformer_har_lip/` | Spikformer256 HAR-DVS and DVS-Lip training |
| `examples/spikformer_gesture/` | Spikformer256 DVS128 Gesture training |
| `examples/spikformer_n101/` | Spikformer256 N-Caltech101 training |
| `configs/` | Eight original experiment policies and command arguments |
| `tests/` | Source integrity, layout, and command-expansion checks |

## Preservation Rules

The four original `spikeDE_dis2` packages were byte-identical after the previous path sanitization. They are consolidated into `spikeDE_dis/`, distinct from the original spikeDE library, and experiment imports are renamed accordingly. Dataset-specific trainers and model definitions remain separate to preserve their original assertions and schedules.

`SOURCE_PROVENANCE.json` maps each exported source file to the original archived snapshot and SHA-256 hashes. No model arithmetic, parameter initialization, optimizer policy, or data-split policy is changed by this directory migration.

## Run From the Repository Root

```bash
python -m pip install -e .
python run_experiment.py --list
python -m unittest discover -s tests -v
```

Use the root launcher rather than invoking a trainer from an arbitrary working directory. It supplies the selected source directory, dataset root, output directory, and repository import path. A command is printed before it is executed. `--print-command` does not import PyTorch or start training.

The selected training paths use direct CuPy `LIFNeuronFDE`; `SNNWrapper` is retained in the underlying library for compatibility, not substituted for these runs. Source and command checks do not replace GPU numerical validation of a fresh installation.
