import torch
import torch.nn as nn
import torch.nn.functional as F


def seq_to_ann_forward(x_seq, module):
    y_shape = [x_seq.shape[0], x_seq.shape[1]]
    y = module(x_seq.flatten(0, 1).contiguous())
    y_shape.extend(y.shape[1:])
    return y.view(y_shape)


class Conv2d(nn.Conv2d):
    def __init__(self, *args, step_mode='s', **kwargs):
        super().__init__(*args, **kwargs)
        self.step_mode = step_mode

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)
        if x.dim() != 5:
            raise ValueError(f'expected [T, N, C, H, W], got {tuple(x.shape)}')
        return seq_to_ann_forward(x, lambda y: nn.Conv2d.forward(self, y))


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, step_mode='s', **kwargs):
        super().__init__(*args, **kwargs)
        self.step_mode = step_mode

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)
        if x.dim() != 5:
            raise ValueError(f'expected [T, N, C, H, W], got {tuple(x.shape)}')
        return seq_to_ann_forward(x, lambda y: nn.BatchNorm2d.forward(self, y))


class MaxPool2d(nn.MaxPool2d):
    def __init__(self, *args, step_mode='s', **kwargs):
        super().__init__(*args, **kwargs)
        self.step_mode = step_mode

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)
        if x.dim() != 5:
            raise ValueError(f'expected [T, N, C, H, W], got {tuple(x.shape)}')
        return seq_to_ann_forward(x, lambda y: nn.MaxPool2d.forward(self, y))


class Flatten(nn.Flatten):
    def __init__(self, start_dim=1, end_dim=-1, step_mode='s'):
        super().__init__(start_dim=start_dim, end_dim=end_dim)
        self.step_mode = step_mode

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)
        return seq_to_ann_forward(x, lambda y: nn.Flatten.forward(self, y))


class Linear(nn.Linear):
    def __init__(self, *args, step_mode='s', **kwargs):
        super().__init__(*args, **kwargs)
        self.step_mode = step_mode

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)
        return seq_to_ann_forward(x, lambda y: nn.Linear.forward(self, y))


class Dropout(nn.Module):
    def __init__(self, p=0.5, step_mode='s'):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f'dropout probability has to be in [0, 1), got {p}')
        self.p = p
        self.step_mode = step_mode
        self.mask = None

    def reset(self):
        self.mask = None

    def _mask_like(self, x):
        self.mask = F.dropout(torch.ones_like(x), self.p, training=True)

    def forward(self, x):
        if not self.training:
            return x
        base = x if self.step_mode == 's' else x[0]
        if self.mask is None or self.mask.shape != base.shape or self.mask.device != base.device:
            self._mask_like(base)
        return x * self.mask
