from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
setup(
    name="df-snn",
    version="0.1.0",
    description="Distributed-Order Fractional Spiking Neural Network",
    packages=find_packages(include=["spikeDE_dis", "spikeDE_dis.*"]),
    python_requires=">=3.9",
    install_requires=(ROOT / "requirements.txt").read_text().splitlines(),
)
