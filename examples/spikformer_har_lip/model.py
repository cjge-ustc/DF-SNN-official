import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
# from spikeDE_0907.neuron import LIFNeuron
# from spikeDE_0907 import SNNWrapper
from spikeDE_dis.neuron import LIFNeuron, LIFNeuronFDE
from spikeDE_dis import SNNWrapper
from spikeDE_dis.layer import ClassificationHead

__all__ = [
    'spikformer',
    'spikformer_e256_lif_fde',
    'spikformer_e256_lif_fde_vogl',
    'spikformer_e384_lif_fde',
    'spikformer_e512_lif_fde',
    'sew_resnet18',
    'sew_resnet18_lif_fde',
    'sew_resnet18_lif_fde_vo',
    'sew_resnet18_lif_fde_vogl',
    'sew_resnet18_lif_fde_fagl',
    'sew_resnet34',
    'sew_resnet34_lif_fde',
    'sew_resnet50',
    'sew_resnet50_lif_fde',
]


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.,
                 tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

    def forward(self, x):
        # Input: [B, C, N]
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        x = self.fc1_lif(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x)
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., sr_ratio=1, tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

        self.attn_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

    def forward(self, x):
        # Input: [B, C, N]
        B, C, N = x.shape

        # Q path
        q = self.q_conv(x)
        q = self.q_bn(q)
        q = self.q_lif(q)
        q = q.transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3).contiguous()  # [B, num_heads, N, C//num_heads]

        # K path
        k = self.k_conv(x)
        k = self.k_bn(k)
        k = self.k_lif(k)
        k = k.transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3).contiguous()  # [B, num_heads, N, C//num_heads]

        # V path
        v = self.v_conv(x)
        v = self.v_bn(v)
        v = self.v_lif(v)
        v = v.transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3).contiguous()  # [B, num_heads, N, C//num_heads]

        # Attention
        attn = (q @ k.transpose(-2, -1)) @ v  # [B, num_heads, N, C//num_heads]
        x = attn * self.scale

        # Reshape back
        x = x.transpose(2, 3).reshape(B, C, N).contiguous()
        x = self.attn_lif(x)

        # Output projection
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 sr_ratio=1, tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                        tau=tau, threshold=threshold, surrogate_grad_scale=surrogate_grad_scale)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop,
                       tau=tau, threshold=threshold, surrogate_grad_scale=surrogate_grad_scale)
        self.drop_path = DropPath(drop_path)
        self.drop_path._snnwrapper_leaf_module = True

    def forward(self, x):
        # Input: [B, C, N]
        identity = x
        x = self.drop_path(self.attn(x))
        x = x + identity  # First residual

        identity = x
        x = self.drop_path(self.mlp(x))
        # Don't add residual here, let MLP output directly
        # x = x + identity  # Remove this line
        return x + identity  # Keep the residual
    #
    # def forward(self, x):
    #     # Input: [B, C, N]
    #     x = x + self.attn(x)
    #     x = x + self.mlp(x)
    #     return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2,
                 embed_dims=256, tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()

        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

    def forward(self, x):
        # Input: [B, C, H, W]
        B, C, H, W = x.shape

        # Stage 1
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        x = self.maxpool(x)

        # Stage 2
        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.proj_lif1(x)
        x = self.maxpool1(x)

        # Stage 3
        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.proj_lif2(x)
        x = self.maxpool2(x)

        # Stage 4
        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.proj_lif3(x)
        x = self.maxpool3(x)

        # Relative position encoding
        x_rpe = self.rpe_conv(x)
        x_rpe = self.rpe_bn(x_rpe)
        x_rpe = self.rpe_lif(x_rpe)
        x = x + x_rpe

        # Reshape to [B, embed_dims, num_patches]
        B, C, H_out, W_out = x.shape
        x = x.reshape(B, C, H_out * W_out).contiguous()

        return x


def sew_residual(x, identity, connect_f):
    connect_f = connect_f.upper()
    if connect_f == 'ADD':
        return x + identity
    if connect_f == 'AND':
        return x * identity
    if connect_f == 'IAND':
        return identity * (1. - x)
    raise NotImplementedError(f'Unsupported SEW connect function: {connect_f}')


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEWBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 connect_f='ADD', tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.downsample = downsample
        self.connect_f = connect_f

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lif2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return sew_residual(out, identity, self.connect_f)


class SEWBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 connect_f='ADD', tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()
        outplanes = planes * self.expansion
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.conv3 = conv1x1(planes, outplanes)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.lif3 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.downsample = downsample
        self.connect_f = connect_f

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lif2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.lif3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return sew_residual(out, identity, self.connect_f)


class SEWClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = x.mean((-1, -2))
        x = self.fc(x)
        x = x.mean(0)
        return x


class SEWResNet(nn.Module):
    def __init__(self, block, layers, in_channels=2, num_classes=101,
                 base_channels=64, connect_f='ADD', tau=2.0, threshold=1.0,
                 surrogate_grad_scale=5.0):
        super().__init__()
        self.inplanes = base_channels
        self.connect_f = connect_f.upper()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.lif1 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1,
                                       tau=tau, threshold=threshold,
                                       surrogate_grad_scale=surrogate_grad_scale)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2,
                                       tau=tau, threshold=threshold,
                                       surrogate_grad_scale=surrogate_grad_scale)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2,
                                       tau=tau, threshold=threshold,
                                       surrogate_grad_scale=surrogate_grad_scale)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2,
                                       tau=tau, threshold=threshold,
                                       surrogate_grad_scale=surrogate_grad_scale)
        out_channels = base_channels * 8 * block.expansion
        self.head = SEWClassificationHead(out_channels, num_classes)

        self.apply(self._init_weights)

    def _make_layer(self, block, planes, blocks, stride=1,
                    tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample, self.connect_f,
                  tau=tau, threshold=threshold,
                  surrogate_grad_scale=surrogate_grad_scale)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, connect_f=self.connect_f,
                      tau=tau, threshold=threshold,
                      surrogate_grad_scale=surrogate_grad_scale)
            )
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return self.head(x)


def _time_distributed(module, x):
    """Apply a regular PyTorch layer to a [T, B, ...] tensor."""
    if x.dim() < 3:
        raise ValueError(f'Expected a time-major tensor [T, B, ...], got {tuple(x.shape)}')
    time_steps, batch_size = x.shape[:2]
    y = module(x.flatten(0, 1))
    return y.reshape(time_steps, batch_size, *y.shape[1:]).contiguous()


def _direct_fde_method(method):
    aliases = {
        'pred': 'predictor',
        'predictor': 'predictor',
        'vopred': 'vopredictor',
        'vopredictor': 'vopredictor',
        'vogl': 'vogl1',
        'vogl1': 'vogl1',
        'fagl': 'fagl1',
        'fagl1': 'fagl1',
        'gl': 'gl',
        'gl1': 'gl1',
        'trap': 'trap',
        'euler': 'euler',
        'glmulti': 'glmulti',
    }
    if method not in aliases:
        raise ValueError(
            f"Direct LIFNeuronFDE does not support method={method!r}. "
            "Use one of: pred, predictor, vopred, vopredictor, vogl, vogl1, fagl, fagl1, gl, trap, euler, glmulti."
        )
    return aliases[method]


class DirectFDEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, neuron_factory, stride=1,
                 downsample=None, connect_f='ADD'):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = neuron_factory('lif1')
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = neuron_factory('lif2')
        self.downsample = downsample
        self.connect_f = connect_f

    def forward(self, x, beta_schedule=None):
        identity = x
        out = _time_distributed(self.conv1, x)
        out = _time_distributed(self.bn1, out)
        out = self.lif1(out, beta_schedule=beta_schedule)
        out = _time_distributed(self.conv2, out)
        out = _time_distributed(self.bn2, out)
        out = self.lif2(out, beta_schedule=beta_schedule)
        if self.downsample is not None:
            identity = _time_distributed(self.downsample, identity)
        return sew_residual(out, identity, self.connect_f)


class DirectFDEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, neuron_factory, stride=1,
                 downsample=None, connect_f='ADD'):
        super().__init__()
        outplanes = planes * self.expansion
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = neuron_factory('lif1')
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = neuron_factory('lif2')
        self.conv3 = conv1x1(planes, outplanes)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.lif3 = neuron_factory('lif3')
        self.downsample = downsample
        self.connect_f = connect_f

    def forward(self, x):
        identity = x
        out = _time_distributed(self.conv1, x)
        out = _time_distributed(self.bn1, out)
        out = self.lif1(out)
        out = _time_distributed(self.conv2, out)
        out = _time_distributed(self.bn2, out)
        out = self.lif2(out)
        out = _time_distributed(self.conv3, out)
        out = _time_distributed(self.bn3, out)
        out = self.lif3(out)
        if self.downsample is not None:
            identity = _time_distributed(self.downsample, identity)
        return sew_residual(out, identity, self.connect_f)


class DirectFDEClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = x.mean((-1, -2))
        x = _time_distributed(self.fc, x)
        return x.mean(0)


class DirectFDEResNet(nn.Module):
    """SEW-ResNet with explicit multi-step LIFNeuronFDE modules."""

    def __init__(self, block, layers, neuron_factory, in_channels=2, num_classes=11,
                 base_channels=64, connect_f='ADD', time_steps=16,
                 shared_beta_init=None, beta_mode='grid', beta_min=0.02,
                 beta_max=0.30, beta_anchor=0.1, beta_residual_scale=0.05,
                 beta_residual_init='zero', beta_anchor_weight=0.0, beta_smooth_weight=0.0):
        super().__init__()
        self.inplanes = base_channels
        self.connect_f = connect_f.upper()
        self.time_steps = int(time_steps)
        self.beta_mode = beta_mode
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_anchor = float(beta_anchor)
        self.beta_residual_scale = float(beta_residual_scale)
        self.beta_residual_init = beta_residual_init
        self.beta_anchor_weight = float(beta_anchor_weight)
        self.beta_smooth_weight = float(beta_smooth_weight)
        self._last_beta_schedule = None
        if shared_beta_init is None:
            self.beta_raw = None
        else:
            beta_init = min(max(float(shared_beta_init), self.beta_min + 1e-6), self.beta_max - 1e-6)
            normalized = (beta_init - self.beta_min) / (self.beta_max - self.beta_min)
            initial_raw = torch.logit(torch.tensor(normalized))
            if beta_mode == 'fixed_per_time':
                self.register_buffer('beta_raw', initial_raw.repeat(self.time_steps))
            elif beta_mode == 'shared_scalar':
                self.beta_raw = nn.Parameter(initial_raw.reshape(1))
            elif beta_mode == 'grid':
                self.beta_raw = nn.Parameter(initial_raw.repeat(self.time_steps))
            elif beta_mode == 'residual_grid4':
                patterns = {
                    'zero': [0.0, 0.0, 0.0, 0.0],
                    'increasing': [-0.75, -0.25, 0.25, 0.75],
                    'decreasing': [0.75, 0.25, -0.25, -0.75],
                    'middle_high': [0.0, 0.75, 0.75, 0.0],
                    'middle_low': [0.0, -0.75, -0.75, 0.0],
                    'zigzag': [-0.75, 0.75, -0.75, 0.75],
                }
                if beta_residual_init not in patterns:
                    raise ValueError(f'Unsupported beta_residual_init={beta_residual_init!r}')
                initial_residual = torch.tensor(patterns[beta_residual_init]).clamp(-0.999, 0.999)
                self.beta_raw = nn.Parameter(torch.atanh(initial_residual))
            elif beta_mode == 'residual_grid16':
                self.beta_raw = nn.Parameter(torch.zeros(self.time_steps))
            elif beta_mode == 'time_mlp':
                self.beta_raw = None
                self.beta_net = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
                nn.init.zeros_(self.beta_net[-1].weight)
                nn.init.constant_(self.beta_net[-1].bias, float(initial_raw))
            elif beta_mode == 'input_time':
                self.beta_raw = None
                self.beta_net = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))
                nn.init.zeros_(self.beta_net[-1].weight)
                nn.init.constant_(self.beta_net[-1].bias, float(initial_raw))
            else:
                raise ValueError(f'Unsupported beta_mode={beta_mode!r}')
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.lif1 = neuron_factory('stem_lif')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, base_channels, layers[0], neuron_factory, stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], neuron_factory, stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], neuron_factory, stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], neuron_factory, stride=2)
        self.head = DirectFDEClassificationHead(
            base_channels * 8 * block.expansion, num_classes
        )
        self.apply(self._init_weights)
        if hasattr(self, 'beta_net'):
            nn.init.zeros_(self.beta_net[-1].weight)
            nn.init.constant_(self.beta_net[-1].bias, float(initial_raw))

    def _make_layer(self, block, planes, blocks, neuron_factory, stride):
        downsample = None
        outplanes = planes * block.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, outplanes, stride),
                nn.BatchNorm2d(outplanes),
            )
        layers = [block(
            self.inplanes, planes, neuron_factory, stride, downsample, self.connect_f
        )]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, neuron_factory, connect_f=self.connect_f
            ))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(module):
        if getattr(module, '_skip_direct_fde_init', False):
            return
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, x_time=None, output_time=None, method=None, options=None):
        if x.dim() != 5:
            raise ValueError(f'Expected [T, B, C, H, W], got {tuple(x.shape)}')
        for module in self.modules():
            if isinstance(module, LIFNeuronFDE):
                module.reset()
        beta_schedule = None
        if self.beta_mode in ('fixed_per_time', 'shared_scalar', 'grid') and self.beta_raw is not None:
            if x.shape[0] != self.time_steps:
                raise ValueError(
                    f'Expected T={self.time_steps} for shared beta, got T={x.shape[0]}'
                )
            beta_schedule = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.beta_raw)
            if beta_schedule.numel() == 1:
                beta_schedule = beta_schedule.expand(self.time_steps)
        elif self.beta_mode in ('residual_grid4', 'residual_grid16'):
            residual = torch.tanh(self.beta_raw)
            if residual.numel() != self.time_steps:
                sample_pos = torch.linspace(
                    0.0, float(residual.numel() - 1), self.time_steps,
                    device=residual.device, dtype=residual.dtype,
                )
                control_pos = torch.arange(
                    residual.numel(), device=residual.device, dtype=residual.dtype,
                )
                weights = (1.0 - (sample_pos[:, None] - control_pos[None, :]).abs()).clamp_min(0.0)
                residual = (
                    weights[:, 0] * residual[0]
                    + weights[:, 1] * residual[1]
                    + weights[:, 2] * residual[2]
                    + weights[:, 3] * residual[3]
                )
            beta_schedule = self.beta_anchor + self.beta_residual_scale * residual
        elif self.beta_mode == 'time_mlp':
            t = torch.linspace(-1.0, 1.0, self.time_steps, device=x.device, dtype=x.dtype)[:, None]
            beta_schedule = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.beta_net(t).squeeze(-1))
        elif self.beta_mode == 'input_time':
            t = torch.linspace(-1.0, 1.0, self.time_steps, device=x.device, dtype=x.dtype)[:, None, None]
            mean = x.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
            std = x.std(dim=(2, 3, 4), keepdim=False, unbiased=False).unsqueeze(-1)
            features = torch.cat((t.expand(-1, x.shape[1], -1), mean, std), dim=-1)
            beta_schedule = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.beta_net(features).squeeze(-1))
        if beta_schedule is not None:
            self._last_beta_schedule = beta_schedule
            if getattr(self, 'fde_backend', 'torch') == 'cupy':
                schedule_builder = (
                    LIFNeuronFDE.build_vogl1_schedule
                    if self.lif1.method == 'vogl1'
                    else LIFNeuronFDE.build_vopredictor_schedule
                )
                beta_schedule = schedule_builder(beta_schedule, self.lif1.h)

        x = _time_distributed(self.conv1, x)
        x = _time_distributed(self.bn1, x)
        x = self.lif1(x, beta_schedule=beta_schedule)
        x = _time_distributed(self.maxpool, x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x = block(x, beta_schedule=beta_schedule)
        return self.head(x)

    def get_alpha_value(self):
        if self._last_beta_schedule is None:
            return None
        beta = self._last_beta_schedule.detach()
        if beta.dim() == 2:
            beta = beta.mean(dim=1)
        return beta.cpu().tolist()

    def beta_regularization(self):
        beta = self._last_beta_schedule
        if beta is None:
            return next(self.parameters()).new_zeros(())
        anchor = (beta - self.beta_anchor).square().mean()
        smooth = (beta[1:] - beta[:-1]).square().mean() if beta.shape[0] > 1 else beta.new_zeros(())
        return self.beta_anchor_weight * anchor + self.beta_smooth_weight * smooth

    def get_attention_stats(self):
        entropies = []
        for module in self.modules():
            if isinstance(module, LIFNeuronFDE):
                stats = module.get_attention_stats()
                if stats is not None:
                    entropies.append(stats['entropy'])
        if not entropies:
            return None
        return {'entropy': sum(entropies) / len(entropies)}


class DirectFDEMLP(nn.Module):
    def __init__(self, in_features, hidden_features, neuron_factory):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = neuron_factory('mlp_lif1')
        self.fc2_conv = nn.Conv1d(hidden_features, in_features, kernel_size=1)
        self.fc2_bn = nn.BatchNorm1d(in_features)
        self.fc2_lif = neuron_factory('mlp_lif2')

    def forward(self, x, beta_schedule=None):
        x = _time_distributed(self.fc1_conv, x)
        x = _time_distributed(self.fc1_bn, x)
        x = self.fc1_lif(x, beta_schedule=beta_schedule)
        x = _time_distributed(self.fc2_conv, x)
        x = _time_distributed(self.fc2_bn, x)
        return self.fc2_lif(x, beta_schedule=beta_schedule)


class DirectFDESSA(nn.Module):
    def __init__(self, dim, num_heads, neuron_factory):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}')
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron_factory('q_lif')
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron_factory('k_lif')
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron_factory('v_lif')
        self.attn_lif = neuron_factory('attn_lif')
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron_factory('proj_lif')

    def _qkv(self, conv, bn, lif, x, beta_schedule=None):
        x = _time_distributed(conv, x)
        x = _time_distributed(bn, x)
        x = lif(x, beta_schedule=beta_schedule)
        time_steps, batch_size, channels, tokens = x.shape
        x = x.transpose(-1, -2).reshape(
            time_steps, batch_size, tokens, self.num_heads,
            channels // self.num_heads,
        )
        return x.permute(0, 1, 3, 2, 4).contiguous()

    def forward(self, x, beta_schedule=None):
        q = self._qkv(self.q_conv, self.q_bn, self.q_lif, x, beta_schedule)
        k = self._qkv(self.k_conv, self.k_bn, self.k_lif, x, beta_schedule)
        v = self._qkv(self.v_conv, self.v_bn, self.v_lif, x, beta_schedule)
        x = ((q @ k.transpose(-2, -1)) @ v) * self.scale
        time_steps, batch_size, _, tokens, _ = x.shape
        x = x.permute(0, 1, 2, 4, 3).reshape(
            time_steps, batch_size, self.dim, tokens
        ).contiguous()
        x = self.attn_lif(x, beta_schedule=beta_schedule)
        x = _time_distributed(self.proj_conv, x)
        x = _time_distributed(self.proj_bn, x)
        return self.proj_lif(x, beta_schedule=beta_schedule)


class DirectFDETransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, neuron_factory, drop_path=0.):
        super().__init__()
        self.attn = DirectFDESSA(dim, num_heads, neuron_factory)
        self.mlp = DirectFDEMLP(dim, int(dim * mlp_ratio), neuron_factory)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, beta_schedule=None):
        x = x + _time_distributed(self.drop_path, self.attn(x, beta_schedule=beta_schedule))
        return x + _time_distributed(self.drop_path, self.mlp(x, beta_schedule=beta_schedule))


class DirectFDESPS(nn.Module):
    def __init__(self, in_channels, embed_dims, neuron_factory):
        super().__init__()
        dims = [embed_dims // 8, embed_dims // 4, embed_dims // 2, embed_dims]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lifs = nn.ModuleList()
        self.pools = nn.ModuleList()
        current = in_channels
        for index, channels in enumerate(dims):
            self.convs.append(nn.Conv2d(
                current, channels, kernel_size=3, stride=1, padding=1, bias=False
            ))
            self.bns.append(nn.BatchNorm2d(channels))
            self.lifs.append(neuron_factory(f'patch_lif{index + 1}'))
            self.pools.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            current = channels
        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = neuron_factory('rpe_lif')

    def forward(self, x, beta_schedule=None):
        for conv, bn, lif, pool in zip(self.convs, self.bns, self.lifs, self.pools):
            x = _time_distributed(conv, x)
            x = _time_distributed(bn, x)
            x = lif(x, beta_schedule=beta_schedule)
            x = _time_distributed(pool, x)
        identity = x
        x = _time_distributed(self.rpe_conv, x)
        x = _time_distributed(self.rpe_bn, x)
        x = self.rpe_lif(x, beta_schedule=beta_schedule)
        x = x + identity
        time_steps, batch_size, channels, height, width = x.shape
        return x.reshape(time_steps, batch_size, channels, height * width).contiguous()


class DirectFDESpikformer(nn.Module):
    def __init__(self, neuron_factory, in_channels=2, num_classes=11,
                 embed_dims=256, depths=2, num_heads=16, mlp_ratio=4,
                 time_steps=16, shared_beta_init=None, beta_mode='grid',
                 beta_min=0.02, beta_max=0.30, beta_anchor=0.1,
                 beta_residual_scale=0.05, beta_residual_init='zero',
                 beta_anchor_weight=0.0, beta_smooth_weight=0.0):
        super().__init__()
        self.time_steps = int(time_steps)
        self.beta_mode = beta_mode
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_anchor = float(beta_anchor)
        self.beta_residual_scale = float(beta_residual_scale)
        self.beta_anchor_weight = float(beta_anchor_weight)
        self.beta_smooth_weight = float(beta_smooth_weight)
        self._last_beta_schedule = None
        if shared_beta_init is None:
            self.beta_raw = None
        else:
            beta_init = min(max(float(shared_beta_init), self.beta_min + 1e-6), self.beta_max - 1e-6)
            normalized = (beta_init - self.beta_min) / (self.beta_max - self.beta_min)
            initial_raw = torch.logit(torch.tensor(normalized))
            if beta_mode == 'fixed_per_time':
                self.register_buffer('beta_raw', initial_raw.repeat(self.time_steps))
            elif beta_mode == 'shared_scalar':
                self.beta_raw = nn.Parameter(initial_raw.reshape(1))
            elif beta_mode == 'grid':
                self.beta_raw = nn.Parameter(initial_raw.repeat(self.time_steps))
            elif beta_mode == 'residual_grid4':
                patterns = {
                    'zero': [0.0, 0.0, 0.0, 0.0],
                    'increasing': [-0.75, -0.25, 0.25, 0.75],
                    'decreasing': [0.75, 0.25, -0.25, -0.75],
                    'middle_high': [0.0, 0.75, 0.75, 0.0],
                    'middle_low': [0.0, -0.75, -0.75, 0.0],
                    'zigzag': [-0.75, 0.75, -0.75, 0.75],
                }
                if beta_residual_init not in patterns:
                    raise ValueError(f'Unsupported beta_residual_init={beta_residual_init!r}')
                residual = torch.tensor(patterns[beta_residual_init]).clamp(-0.999, 0.999)
                self.beta_raw = nn.Parameter(torch.atanh(residual))
            elif beta_mode == 'residual_grid16':
                self.beta_raw = nn.Parameter(torch.zeros(self.time_steps))
            else:
                raise ValueError(f'Unsupported Spikformer beta_mode={beta_mode!r}')
        self.patch_embed = DirectFDESPS(in_channels, embed_dims, neuron_factory)
        self.blocks = nn.ModuleList([
            DirectFDETransformerBlock(embed_dims, num_heads, mlp_ratio, neuron_factory,
                drop_path=0.1 * index / max(1, depths - 1))
            for index in range(depths)
        ])
        self.head = ClassificationHead(embed_dims, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, x_time=None, output_time=None, method=None, options=None):
        if x.dim() != 5:
            raise ValueError(f'Expected [T, B, C, H, W], got {tuple(x.shape)}')
        if x.shape[0] != self.time_steps:
            raise ValueError(f'Expected T={self.time_steps}, got T={x.shape[0]}')
        for module in self.modules():
            if isinstance(module, LIFNeuronFDE):
                module.reset()
        beta_schedule = None
        if self.beta_mode in ('fixed_per_time', 'shared_scalar', 'grid') and self.beta_raw is not None:
            beta_schedule = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.beta_raw)
            if beta_schedule.numel() == 1:
                beta_schedule = beta_schedule.expand(self.time_steps)
        elif self.beta_mode in ('residual_grid4', 'residual_grid16'):
            residual = torch.tanh(self.beta_raw)
            if residual.numel() != self.time_steps:
                sample_pos = torch.linspace(
                    0.0, float(residual.numel() - 1), self.time_steps,
                    device=residual.device, dtype=residual.dtype,
                )
                control_pos = torch.arange(
                    residual.numel(), device=residual.device, dtype=residual.dtype,
                )
                weights = (
                    1.0 - (sample_pos[:, None] - control_pos[None, :]).abs()
                ).clamp_min(0.0)
                residual = weights @ residual
            beta_schedule = self.beta_anchor + self.beta_residual_scale * residual
        if beta_schedule is not None:
            if torch.any((beta_schedule <= 0.0) | (beta_schedule >= 1.0)):
                raise ValueError('VOGL1 beta schedule must remain strictly inside (0, 1)')
            self._last_beta_schedule = beta_schedule
            first_lif = self.patch_embed.lifs[0]
            if getattr(self, 'fde_backend', 'torch') == 'cupy':
                beta_schedule = LIFNeuronFDE.build_vogl1_schedule(beta_schedule, first_lif.h)
        x = self.patch_embed(x, beta_schedule=beta_schedule)
        for block in self.blocks:
            x = block(x, beta_schedule=beta_schedule)
        return self.head(x)

    def get_alpha_value(self):
        if self._last_beta_schedule is None:
            return None
        return self._last_beta_schedule.detach().cpu().tolist()

    def beta_regularization(self):
        beta = self._last_beta_schedule
        if beta is None:
            return next(self.parameters()).new_zeros(())
        anchor = (beta - self.beta_anchor).square().mean()
        smooth = (beta[1:] - beta[:-1]).square().mean()
        return self.beta_anchor_weight * anchor + self.beta_smooth_weight * smooth

    def get_attention_stats(self):
        entropies = []
        for module in self.modules():
            if isinstance(module, LIFNeuronFDE):
                stats = module.get_attention_stats()
                if stats is not None:
                    entropies.append(stats['entropy'])
        return None if not entropies else {'entropy': sum(entropies) / len(entropies)}


class Spikformer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=256, num_heads=16, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=2, sr_ratios=1, tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.patch_embed = SPS(img_size_h=img_size_h, img_size_w=img_size_w,
                               patch_size=patch_size, in_channels=in_channels,
                               embed_dims=embed_dims, tau=tau, threshold=threshold,
                               surrogate_grad_scale=surrogate_grad_scale)

        self.block = nn.ModuleList([
            Block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer,
                  sr_ratio=sr_ratios, tau=tau, threshold=threshold,
                  surrogate_grad_scale=surrogate_grad_scale)
            for j in range(depths)
        ])

        self.head = ClassificationHead(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Input: [B, C, H, W]
        x = self.patch_embed(x)  # [B, C, N]

        for blk in self.block:
            x = blk(x)

        # Global average pooling and classification
        x = self.head(x)
        return x


@register_model
def spikformer(pretrained=False, **kwargs):
    # 提取 SNNWrapper 需要的参数
    integrator = kwargs.pop('integrator', 'fdeint')
    alpha = kwargs.pop('alpha', None)
    beta = kwargs.pop('beta', 0.5)
    multi_coefficient = kwargs.pop('multi_coefficient', None)
    learn_coefficient = kwargs.pop('learn_coefficient', False)
    coefficient_constraint = kwargs.pop('coefficient_constraint', 'none')
    coefficient_update_interval = kwargs.pop('coefficient_update_interval', 1)
    learn_alpha = kwargs.pop('learn_alpha', False)
    alpha_mode = kwargs.pop('alpha_mode', 'global')
    alpha_constraint = kwargs.pop('alpha_constraint', 'none')
    device = kwargs.pop('device', 'cuda:0')
    num_classes = kwargs.pop('num_classes',101)
    in_channels = kwargs.pop('in_chans', kwargs.pop('in_channels', 2))
    embed_dims = kwargs.pop('embed_dims', 256)
    depths = kwargs.pop('depths', 2)
    kwargs.pop('batch_size', None)
    time_steps = kwargs.pop('T', 16)
    kwargs.pop('backend', None)
    # 过滤掉其他不被 Spikformer 接受的参数
    kwargs.pop('method', None)
    kwargs.pop('step_size', None) 
    kwargs.pop('time_interval', None)
    kwargs.pop('memory', None)
    kwargs.pop('connect_f', None)
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    
    model = Spikformer(
        patch_size=16, embed_dims=embed_dims, num_heads=16, mlp_ratios=4,
        in_channels=in_channels, num_classes=num_classes, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths, sr_ratios=1,
        tau=2.0, threshold=1.0, surrogate_grad_scale=5.0,
        **kwargs
    )
    model.default_cfg = _cfg()
    
    # 包装成 SNNWrapper
    if alpha is None:
        alpha = beta
    net = SNNWrapper(
        model,
        integrator=integrator,
        alpha=alpha,
        multi_coefficient=multi_coefficient,
        learn_coefficient=learn_coefficient,
        learn_alpha=learn_alpha,
        alpha_mode=alpha_mode,
        time_steps=time_steps,
        alpha_constraint=alpha_constraint,
    ).to(device)

    # 设置神经元形状，输入格式为 [N, C, H, W]
    net._set_neuron_shapes(input_shape=(1, in_channels, 128, 128))
    return net


def _create_sew_resnet(block, layers, pretrained=False, **kwargs):
    integrator = kwargs.pop('integrator', 'fdeint')
    alpha = kwargs.pop('alpha', None)
    beta = kwargs.pop('beta', 0.5)
    multi_coefficient = kwargs.pop('multi_coefficient', None)
    learn_coefficient = kwargs.pop('learn_coefficient', False)
    kwargs.pop('coefficient_constraint', None)
    kwargs.pop('coefficient_update_interval', None)
    learn_alpha = kwargs.pop('learn_alpha', False)
    alpha_mode = kwargs.pop('alpha_mode', 'global')
    alpha_constraint = kwargs.pop('alpha_constraint', 'none')
    device = kwargs.pop('device', 'cuda:0')
    num_classes = kwargs.pop('num_classes', 101)
    connect_f = kwargs.pop('connect_f', 'ADD')
    in_channels = kwargs.pop('in_chans', kwargs.pop('in_channels', 2))
    base_channels = kwargs.pop('base_channels', 64)
    kwargs.pop('batch_size', None)
    time_steps = kwargs.pop('T', 16)
    kwargs.pop('backend', None)

    kwargs.pop('drop_rate', None)
    kwargs.pop('drop_path_rate', None)
    kwargs.pop('drop_block_rate', None)
    kwargs.pop('method', None)
    kwargs.pop('step_size', None)
    kwargs.pop('time_interval', None)
    kwargs.pop('memory', None)
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    kwargs.pop('global_pool', None)
    kwargs.pop('scriptable', None)
    kwargs.pop('exportable', None)
    kwargs.pop('no_jit', None)

    model = SEWResNet(
        block,
        layers,
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        connect_f=connect_f,
        tau=2.0,
        threshold=1.0,
        surrogate_grad_scale=5.0,
    )
    model.default_cfg = _cfg()

    if alpha is None:
        alpha = beta
    net = SNNWrapper(
        model,
        integrator=integrator,
        alpha=alpha,
        multi_coefficient=multi_coefficient,
        learn_coefficient=learn_coefficient,
        learn_alpha=learn_alpha,
        alpha_mode=alpha_mode,
        time_steps=time_steps,
        alpha_constraint=alpha_constraint,
    ).to(device)

    net._set_neuron_shapes(input_shape=(1, in_channels, 128, 128))
    return net


@register_model
def sew_resnet18(pretrained=False, **kwargs):
    return _create_sew_resnet(SEWBasicBlock, [2, 2, 2, 2], pretrained=pretrained, **kwargs)


def _create_direct_fde_neuron_factory(kwargs):
    alpha = kwargs.pop('alpha', None)
    beta = kwargs.pop('beta', 0.5)
    method = _direct_fde_method(kwargs.pop('method', 'pred'))
    multi_coefficient = kwargs.pop('multi_coefficient', None)
    learn_coefficient = kwargs.pop('learn_coefficient', False)
    coefficient_constraint = kwargs.pop('coefficient_constraint', 'none')
    coefficient_update_interval = kwargs.pop('coefficient_update_interval', 1)
    glmulti_history_grad = kwargs.pop('glmulti_history_grad', 'truncated')
    history_grad_mode = kwargs.pop('history_grad_mode', 'approx')
    batch_size = kwargs.pop('batch_size', 32)
    time_steps = kwargs.pop('T', kwargs.pop('time_steps', 16))
    step_size = kwargs.pop('step_size', 0.1)
    memory = kwargs.pop('memory', -1)
    backend = kwargs.pop('backend', 'torch')
    vo_solver = kwargs.pop('vo_solver', 'abm')
    vo_reset_in_drift = kwargs.pop('vo_reset_in_drift', False)
    vo_residual_mix = kwargs.pop('vo_residual_mix', 0.0)
    attn_dim = kwargs.pop('attn_dim', 4)
    attn_temperature = kwargs.pop('attn_temperature', 1.0)
    attn_mode = kwargs.pop('attn_mode', 'per_sample')
    attn_init = kwargs.pop('attn_init', 'orthogonal_zero_score')
    attn_stats_interval = kwargs.pop('attn_stats_interval', 32)
    fagl_layers = int(kwargs.pop('fagl_layers', -1))
    kwargs.pop('learn_alpha', None)
    kwargs.pop('alpha_mode', None)
    kwargs.pop('alpha_constraint', None)
    kwargs.pop('beta_mode', None)
    kwargs.pop('beta_init', None)
    kwargs.pop('beta_lr', None)
    # Passed by the common training entry point but unused by fixed-order
    # multi-term neurons, including the direct Spikformer path.
    kwargs.pop('beta_min', None)
    kwargs.pop('beta_max', None)
    kwargs.pop('beta_anchor', None)
    kwargs.pop('beta_residual_scale', None)
    kwargs.pop('beta_residual_init', None)
    kwargs.pop('beta_anchor_weight', None)
    kwargs.pop('beta_smooth_weight', None)
    if alpha is None:
        alpha = beta
    is_multi = isinstance(alpha, (list, tuple))
    if is_multi and method != 'glmulti':
        raise ValueError('Multiple alpha values require method=glmulti.')
    if not is_multi and method == 'glmulti':
        raise ValueError('method=glmulti requires multiple alpha values.')

    shared_multi_coefficient = None
    if is_multi and learn_coefficient:
        initial = torch.as_tensor(multi_coefficient, dtype=torch.float32)
        if coefficient_constraint == 'softplus':
            if torch.any(initial <= 0.0):
                raise ValueError('Softplus coefficients must be initialized above zero.')
            initial = torch.log(torch.expm1(initial.clamp_min(1e-6)))
        elif coefficient_constraint == 'softmax':
            if torch.any(initial <= 0.0):
                raise ValueError('Softmax coefficients must be initialized above zero.')
            initial = torch.log(initial)
        elif coefficient_constraint != 'none':
            raise ValueError("coefficient_constraint must be 'none', 'softplus', or 'softmax'")
        shared_multi_coefficient = nn.Parameter(initial)

    neuron_index = 0

    def factory(name):
        nonlocal neuron_index
        neuron_method = method
        if method == 'fagl1' and fagl_layers >= 0 and neuron_index >= fagl_layers:
            neuron_method = 'gl1'
        neuron_index += 1
        return LIFNeuronFDE(
            tau=2.0,
            threshold=1.0,
            surrogate_alpha=5.0,
            surrogate_function='arctan',
            method=neuron_method,
            beta=float(alpha) if not is_multi else 0.5,
            batchsize=batch_size,
            T=time_steps,
            step_size=step_size,
            memory=memory,
            step_mode='m',
            backend=backend,
            name=name,
            v_reset=0.0,
            multi_beta=list(alpha) if is_multi else None,
            multi_coefficient=multi_coefficient if is_multi else None,
            learn_coefficient=learn_coefficient if is_multi else False,
            coefficient_constraint=coefficient_constraint,
            shared_multi_coefficient=shared_multi_coefficient,
            coefficient_update_interval=coefficient_update_interval,
            glmulti_history_grad=glmulti_history_grad,
            history_grad_mode=history_grad_mode,
            vo_solver=vo_solver,
            vo_reset_in_drift=vo_reset_in_drift,
            vo_residual_mix=vo_residual_mix,
            attn_dim=attn_dim,
            attn_temperature=attn_temperature,
            attn_mode=attn_mode,
            attn_init=attn_init,
            attn_stats_interval=attn_stats_interval,
        )
    return factory, backend


def _discard_timm_kwargs(kwargs):
    for key in (
        'integrator', 'time_interval', 'drop_rate', 'drop_path_rate',
        'drop_block_rate', 'pretrained_cfg', 'pretrained_cfg_overlay',
        'cache_dir', 'global_pool', 'scriptable', 'exportable', 'no_jit',
    ):
        kwargs.pop(key, None)


def _create_direct_fde_resnet(block, layers, pretrained=False, **kwargs):
    device = kwargs.pop('device', 'cuda:0')
    num_classes = kwargs.pop('num_classes', 11)
    connect_f = kwargs.pop('connect_f', 'ADD')
    in_channels = kwargs.pop('in_chans', kwargs.pop('in_channels', 2))
    base_channels = kwargs.pop('base_channels', 64)
    time_steps = int(kwargs.get('T', 16))
    requested_method = kwargs.get('method', 'pred')
    beta_init = kwargs.get('beta_init', None)
    if requested_method in ('vopred', 'vopredictor', 'vogl', 'vogl1'):
        beta_init = float(beta_init if beta_init is not None else 0.5)
    else:
        beta_init = None
    beta_mode = kwargs.pop('beta_mode', 'grid')
    if beta_mode == 'shared_per_time':
        beta_mode = 'grid'
    beta_min = kwargs.pop('beta_min', 0.02)
    beta_max = kwargs.pop('beta_max', 0.30)
    beta_anchor = kwargs.pop('beta_anchor', 0.1)
    beta_residual_scale = kwargs.pop('beta_residual_scale', 0.05)
    beta_residual_init = kwargs.pop('beta_residual_init', 'zero')
    beta_anchor_weight = kwargs.pop('beta_anchor_weight', 0.0)
    beta_smooth_weight = kwargs.pop('beta_smooth_weight', 0.0)
    vo_solver = kwargs.pop('vo_solver', 'abm')
    vo_reset_in_drift = kwargs.pop('vo_reset_in_drift', False)
    vo_residual_mix = kwargs.pop('vo_residual_mix', 0.0)
    kwargs['vo_solver'] = vo_solver
    kwargs['vo_reset_in_drift'] = vo_reset_in_drift
    kwargs['vo_residual_mix'] = vo_residual_mix
    kwargs.pop('beta_lr', None)
    kwargs.pop('embed_dims', None)
    kwargs.pop('depths', None)
    neuron_factory, backend = _create_direct_fde_neuron_factory(kwargs)
    _discard_timm_kwargs(kwargs)
    if kwargs:
        raise TypeError(f'Unexpected direct FDE ResNet arguments: {sorted(kwargs)}')

    model = DirectFDEResNet(
        block=block,
        layers=layers,
        neuron_factory=neuron_factory,
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        connect_f=connect_f,
        time_steps=time_steps,
        shared_beta_init=beta_init,
        beta_mode=beta_mode,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_anchor=beta_anchor,
        beta_residual_scale=beta_residual_scale,
        beta_residual_init=beta_residual_init,
        beta_anchor_weight=beta_anchor_weight,
        beta_smooth_weight=beta_smooth_weight,
    )
    model.default_cfg = _cfg()
    model.fde_backend = backend
    return model.to(device)


@register_model
def sew_resnet18_lif_fde(pretrained=False, **kwargs):
    return _create_direct_fde_resnet(
        DirectFDEBasicBlock, [2, 2, 2, 2], pretrained=pretrained, **kwargs
    )


@register_model
def sew_resnet18_lif_fde_vo(pretrained=False, **kwargs):
    kwargs['method'] = 'vopredictor'
    return _create_direct_fde_resnet(
        DirectFDEBasicBlock, [2, 2, 2, 2], pretrained=pretrained, **kwargs
    )


@register_model
def sew_resnet18_lif_fde_vogl(pretrained=False, **kwargs):
    kwargs['method'] = 'vogl1'
    return _create_direct_fde_resnet(
        DirectFDEBasicBlock, [2, 2, 2, 2], pretrained=pretrained, **kwargs
    )


@register_model
def sew_resnet18_lif_fde_fagl(pretrained=False, **kwargs):
    kwargs['method'] = 'fagl1'
    kwargs.setdefault('fagl_layers', -1)
    return _create_direct_fde_resnet(
        DirectFDEBasicBlock, [2, 2, 2, 2], pretrained=pretrained, **kwargs
    )


@register_model
def sew_resnet34_lif_fde(pretrained=False, **kwargs):
    return _create_direct_fde_resnet(
        DirectFDEBasicBlock, [3, 4, 6, 3], pretrained=pretrained, **kwargs
    )


@register_model
def sew_resnet50_lif_fde(pretrained=False, **kwargs):
    return _create_direct_fde_resnet(
        DirectFDEBottleneck, [3, 4, 6, 3], pretrained=pretrained, **kwargs
    )


def _create_direct_fde_spikformer(fixed_embed_dims, pretrained=False, **kwargs):
    device = kwargs.pop('device', 'cuda:0')
    num_classes = kwargs.pop('num_classes', 11)
    in_channels = kwargs.pop('in_chans', kwargs.pop('in_channels', 2))
    depths = kwargs.pop('depths', 2)
    time_steps = int(kwargs.get('T', 16))
    requested_method = kwargs.get('method', 'pred')
    beta_init = kwargs.get('beta_init', None)
    if requested_method in ('vopred', 'vopredictor', 'vogl', 'vogl1'):
        beta_init = float(beta_init if beta_init is not None else 0.5)
    else:
        beta_init = None
    beta_mode = kwargs.get('beta_mode', 'grid')
    if beta_mode == 'shared_per_time':
        beta_mode = 'grid'
    beta_min = kwargs.pop('beta_min', 0.02)
    beta_max = kwargs.pop('beta_max', 0.30)
    beta_anchor = kwargs.pop('beta_anchor', 0.1)
    beta_residual_scale = kwargs.pop('beta_residual_scale', 0.05)
    beta_residual_init = kwargs.pop('beta_residual_init', 'zero')
    beta_anchor_weight = kwargs.pop('beta_anchor_weight', 0.0)
    beta_smooth_weight = kwargs.pop('beta_smooth_weight', 0.0)
    kwargs.pop('embed_dims', None)
    kwargs.pop('connect_f', None)
    neuron_factory, backend = _create_direct_fde_neuron_factory(kwargs)
    _discard_timm_kwargs(kwargs)
    if kwargs:
        raise TypeError(f'Unexpected direct FDE Spikformer arguments: {sorted(kwargs)}')
    model = DirectFDESpikformer(
        neuron_factory=neuron_factory,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dims=fixed_embed_dims,
        depths=depths,
        time_steps=time_steps,
        shared_beta_init=beta_init,
        beta_mode=beta_mode,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_anchor=beta_anchor,
        beta_residual_scale=beta_residual_scale,
        beta_residual_init=beta_residual_init,
        beta_anchor_weight=beta_anchor_weight,
        beta_smooth_weight=beta_smooth_weight,
    )
    model.default_cfg = _cfg()
    model.fde_backend = backend
    return model.to(device)


@register_model
def spikformer_e256_lif_fde(pretrained=False, **kwargs):
    return _create_direct_fde_spikformer(256, pretrained=pretrained, **kwargs)


@register_model
def spikformer_e256_lif_fde_vogl(pretrained=False, **kwargs):
    kwargs['method'] = 'vogl1'
    return _create_direct_fde_spikformer(256, pretrained=pretrained, **kwargs)


@register_model
def spikformer_e384_lif_fde(pretrained=False, **kwargs):
    return _create_direct_fde_spikformer(384, pretrained=pretrained, **kwargs)


@register_model
def spikformer_e512_lif_fde(pretrained=False, **kwargs):
    return _create_direct_fde_spikformer(512, pretrained=pretrained, **kwargs)


@register_model
def sew_resnet34(pretrained=False, **kwargs):
    return _create_sew_resnet(SEWBasicBlock, [3, 4, 6, 3], pretrained=pretrained, **kwargs)


@register_model
def sew_resnet50(pretrained=False, **kwargs):
    return _create_sew_resnet(SEWBottleneck, [3, 4, 6, 3], pretrained=pretrained, **kwargs)
