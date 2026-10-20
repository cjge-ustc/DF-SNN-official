import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from model import Spikformer, DirectFDESpikformer
from spikeDE_dis import SNNWrapper
from spikeDE_dis.neuron import LIFNeuronFDE


class SoftplusMass(nn.Module):
    def forward(self, raw):
        return F.softplus(raw)

    def right_inverse(self, mass):
        return torch.log(torch.expm1(mass))


def make_model(implementation, beta, learn, device='cuda:0', batch=32, T=16, masses=None):
    masses = masses if masses is not None else [0.5] * len(beta)
    if implementation == 'wrapper':
        backbone = Spikformer(patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
            in_channels=2, num_classes=101, qkv_bias=False, depths=2, sr_ratios=1,
            tau=2., threshold=1., surrogate_grad_scale=5., drop_path_rate=0.,
            drop_rate=0., attn_drop_rate=0.)
        net = SNNWrapper(backbone, integrator='fdeint', alpha=list(beta),
            multi_coefficient=list(masses), learn_coefficient=learn, time_steps=T,
            compile_ode=True).to(device)
        if learn:
            parametrize.register_parametrization(net, 'multi_coefficient', SoftplusMass())
        net._set_neuron_shapes(input_shape=(1, 2, 128, 128))
        net.fde_backend = 'torch'
    else:
        if implementation == 'native':
            from aligned_glmulti import make_model as aligned_model
            net = aligned_model('native', list(beta), learn, list(masses), batch=batch, T=T,
                                fixed_gain=False).to(device)
            net.ablation_implementation = implementation
            return net
        shared = nn.Parameter(torch.log(torch.expm1(torch.tensor(masses)))) if learn else None
        def neuron_factory(name):
            return LIFNeuronFDE(tau=2., threshold=1., surrogate_grad_scale=5.,
                surrogate_function='arctan', beta=beta[0], T=T, method='glmulti', step_size=.1,
                batchsize=batch, name=name, backend='cupy' if implementation == 'native' else 'torch',
                step_mode='m', multi_beta=list(beta), multi_coefficient=list(masses),
                learn_coefficient=learn, coefficient_constraint='softplus',
                shared_multi_coefficient=shared, coefficient_update_interval=1,
                glmulti_history_grad='full')
        net = DirectFDESpikformer(neuron_factory=neuron_factory, in_channels=2,
            num_classes=101, embed_dims=256, depths=2, time_steps=T).to(device)
    net.ablation_implementation = implementation
    return net


def effective_masses(net):
    if isinstance(net, SNNWrapper):
        return net.multi_coefficient
    for module in net.modules():
        if isinstance(module, LIFNeuronFDE):
            return module.get_multi_coefficient()
    raise RuntimeError('No distributed-order masses found')
