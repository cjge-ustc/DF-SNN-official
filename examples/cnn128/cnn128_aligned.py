"""Original CNN128 topology with per-frame layers and aligned multistep FDE neurons."""
import torch
from torch import nn
import aligned_glmulti as alg
from spikeDE_dis.neuron import LIFNeuronFDE


class CNN128(nn.Module):
    def __init__(self, beta, masses, learn, batch, T, num_classes):
        super().__init__()
        raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(masses)))) if learn else None
        layers = []
        def neuron(name):
            return LIFNeuronFDE(tau=2., threshold=1., surrogate_grad_scale=5.,
                surrogate_function="arctan", beta=beta[0], T=T, method="glmulti",
                step_size=.1, batchsize=batch, name=name, backend="cupy", step_mode="m",
                multi_beta=beta, multi_coefficient=masses, learn_coefficient=learn,
                coefficient_constraint="softplus", shared_multi_coefficient=raw,
                coefficient_update_interval=1, glmulti_history_grad="full")
        for i in range(5):
            layers.extend([nn.Conv2d(2 if i == 0 else 128, 128, 3, padding=1, bias=False),
                           nn.BatchNorm2d(128), neuron("conv" + str(i)), nn.MaxPool2d(2)])
        layers.extend([nn.Flatten(), nn.Dropout(.5), nn.Linear(128 * 4 * 4, 512),
                       neuron("fc0"), nn.Dropout(.5), nn.Linear(512, 256),
                       neuron("fc1"), nn.Linear(256, num_classes)])
        self.backbone = nn.Sequential(*layers)
        self.ablation_implementation = "cupy"
        self.fde_backend = "cupy"
        self.time_steps = T

    def forward(self, x, *args, **kwargs):
        if x.ndim != 5 or x.shape[0] != self.time_steps:
            raise ValueError("CNN128 expects [T,N,2,128,128]")
        for layer in self.backbone:
            if isinstance(layer, LIFNeuronFDE):
                x = layer(x)
            else:
                x = torch.stack([layer(frame) for frame in x])
        return x.mean(0)


def make_model(backend, beta, learn, masses, batch=32, T=16, fixed_gain=False, num_classes=101):
    if backend != "cupy" or fixed_gain:
        raise ValueError("CNN128 requires unnormalized aligned CuPy")
    alg.install("cupy")
    return CNN128(beta, masses, learn, batch, T, num_classes)
