<div align="center">

<h1>DF-SNN: Distributed-Order Fractional Spiking Neural Network</h1>

<p><strong>Official Implementation &middot; NeurIPS 2026</strong></p>

<p>
  <img src="https://img.shields.io/badge/NeurIPS-2026-222222?style=flat-square" alt="NeurIPS 2026">
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9">
  <img src="https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.7.1">
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 12.6">
</p>

<p>
  <a href="#installation">Installation</a> &middot;
  <a href="#dataset-preparation">Datasets</a> &middot;
  <a href="#cnn128">CNN128</a> &middot;
  <a href="#spikformer256">Spikformer256</a> &middot;
  <a href="#inspect-commands">Commands</a>
</p>

</div>

---

This repository provides the official implementation of **Distributed-Order Fractional Spiking Neural Network (NeurIPS 2026)**, with training commands for **CNN128** and **Spikformer256** on four neuromorphic vision datasets.

The supplied configurations use CuPy full-history `LIFNeuronFDE` and shared, unnormalized Softplus coefficients. The instructions below cover setup, dataset preparation, and training.

## Installation

**Requirements:** Linux, an NVIDIA GPU, and a CUDA-compatible driver. Recorded environment: Python 3.9, PyTorch 2.7.1, CUDA 12.6, and CuPy 13.6.0.

### 1. Clone and Create the Environment

```bash
git clone git@github.com:cjge-ustc/DF-SNN.git
cd DF-SNN
conda create -n df-snn python=3.9 -y
conda activate df-snn
```

This repository is private. Cloning requires access to the repository and a GitHub SSH key configured on the machine. Alternatively, use GitHub's **Code > Download ZIP**, extract it, and enter the extracted directory.

### 2. Install Dependencies

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

### 3. Install the Local Library

```bash
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
python tests/check_runtime.py
```

The runtime check disables CUDA and checks all eight real argument parsers, model constructors, and optimizer/scheduler construction. It does not train, evaluate accuracy, or validate CUDA kernels. Run `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"` before training. The latter must print `True`; use the CUDA 12.x CuPy wheel above, not a second `cupy` wheel alongside it.

Source files are directly browsable on GitHub. The shared neuron library is in [`spikeDE_dis/`](spikeDE_dis/), distinct from the original spikeDE library, and dataset-specific training implementations are in [`examples/`](examples/). No archive extraction is required.

## Dataset Preparation

Prepare the original datasets or compatible event caches using the directory layouts below. HAR-DVS uses native T=8 frames, not truncated T=16 frames; other datasets use T=16. Integration uses `split_by='number'`.

| Dataset | Official source / download | Environment variable | Time steps |
|:---|:---|:---|:---:|
| HAR-DVS | [Authors' repository and downloads](https://github.com/Event-AHU/HARDVS#dataset-download) | `HAR_DVS` | 8 |
| DVS128 Gesture | [IBM dataset link](https://ibm.biz/EventCameraData), [paper/project](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system) | `DVS_GESTURE` | 16 |
| N-Caltech101 | [Author's dataset page](https://www.garrickorchard.com/datasets/n-caltech101), [Mendeley download](https://data.mendeley.com/datasets/cy6cvx3ryv/1) | `NCALTECH101` | 16 |
| DVS-Lip | [Authors' project](https://sites.google.com/view/event-based-lipreading), [DVS-Lip.zip](https://drive.google.com/file/d/1dBEgtmctTTWJlWnuWxFtk8gfOdVVpkQ0/view) | `DVS_LIP` | 16 |

```bash
export HAR_DVS=/path/to/HAR-DVS
export DVS_GESTURE=/path/to/DVS128Gesture
export NCALTECH101=/path/to/N-Caltech101
export DVS_LIP=/path/to/DVS-Lip
```

These variables point to the **dataset root**, not the archive, `train/`, or frame-cache directory. Follow [the exact archive names and directory layouts](docs/datasets.md) before launching. Downloads are manual and may require the provider's login/permission; the IBM short link could not be fetched during this audit. First-time event conversion/frame integration can take considerable CPU time and disk space: prepare each dataset once before launching concurrent experiments, and do not interrupt preprocessing.

For an already prepared cache, check its counts and first/last samples without downloading, preprocessing, or using a GPU:

```bash
python tests/check_data.py cnn128_hardvs --data-path "$HAR_DVS"
python tests/check_data.py cnn128_gesture --data-path "$DVS_GESTURE"
python tests/check_data.py cnn128_n101 --data-path "$NCALTECH101"
python tests/check_data.py cnn128_dvslip --data-path "$DVS_LIP"
```

The same checks accept the four `spikformer_*` preset names. They check dataset indexing and sampled frames, not complete inference.

## CNN128

**400 epochs &middot; Batch size 16**

Each command launches one experiment. Set `CUDA_VISIBLE_DEVICES` to a free GPU; these are independent commands, not a multi-GPU scheduler.

### HAR-DVS

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py cnn128_hardvs --data-path "$HAR_DVS" --output-dir runs/cnn128_hardvs
```

<details>
<summary><strong>DVS128 Gesture, N-Caltech101, and DVS-Lip</strong></summary>

```bash
# DVS128 Gesture
CUDA_VISIBLE_DEVICES=0 python run_experiment.py cnn128_gesture --data-path "$DVS_GESTURE" --output-dir runs/cnn128_gesture

# N-Caltech101
CUDA_VISIBLE_DEVICES=0 python run_experiment.py cnn128_n101 --data-path "$NCALTECH101" --output-dir runs/cnn128_n101

# DVS-Lip
CUDA_VISIBLE_DEVICES=0 python run_experiment.py cnn128_dvslip --data-path "$DVS_LIP" --output-dir runs/cnn128_dvslip
```

</details>

## Spikformer256

**Embedding dimension 256 &middot; Depth 2 &middot; Batch size 32**

HAR-DVS and DVS-Lip: **500 epochs**. DVS128 Gesture and N-Caltech101: **1000 epochs**.

### HAR-DVS

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py spikformer_hardvs --data-path "$HAR_DVS" --output-dir runs/spikformer_hardvs
```

<details>
<summary><strong>DVS128 Gesture, N-Caltech101, and DVS-Lip</strong></summary>

```bash
# DVS128 Gesture
CUDA_VISIBLE_DEVICES=0 python run_experiment.py spikformer_gesture --data-path "$DVS_GESTURE" --output-dir runs/spikformer_gesture

# N-Caltech101
CUDA_VISIBLE_DEVICES=0 python run_experiment.py spikformer_n101 --data-path "$NCALTECH101" --output-dir runs/spikformer_n101

# DVS-Lip
CUDA_VISIBLE_DEVICES=0 python run_experiment.py spikformer_dvslip --data-path "$DVS_LIP" --output-dir runs/spikformer_dvslip
```

</details>

## Inspect Commands

Print expanded arguments without importing PyTorch or starting a GPU process:

```bash
python run_experiment.py --list
python run_experiment.py spikformer_n101 --data-path "$NCALTECH101" --output-dir runs/spikformer_n101 --print-command
```

All eight CPU model-construction checks have passed in the recorded 4090 software environment; a fresh GPU training run of this exported package has not been performed.

## Configuration and Source

| File | Purpose |
|:---|:---|
| [`run_experiment.py`](run_experiment.py) | Unified training launcher |
| [`spikeDE_dis/`](spikeDE_dis/) | Shared neurons, solvers, surrogate gradients, and CuPy kernels |
| [`examples/`](examples/) | CNN128 and Spikformer256 training implementations |
| [`configs/experiments.json`](configs/experiments.json) | Eight experiment policies and original command arguments |
| [`requirements.txt`](requirements.txt) | Recorded dependency versions |
| [`setup.py`](setup.py) | Editable installation of the local library |
| [`docs/`](docs/) | Directory layout and implementation notes |
| [`tests/`](tests/) | Source integrity and launch-command checks |
| [`SOURCE_PROVENANCE.json`](SOURCE_PROVENANCE.json) | Original and exported source SHA-256 hashes |

