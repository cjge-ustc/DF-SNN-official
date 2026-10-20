import torch
import torch.nn as nn
import torch.nn.functional as F

# class VotingLayer(nn.Module):
#     def __init__(self, voting_size: int = 10):
#         """
#         * :ref:`API in English <VotingLayer-en>`
#
#         .. _VotingLayer-cn:
#
#         :param voting_size: 决定一个类别的投票数量
#         :type voting_size: int
#
#         投票层，对 ``shape = [..., C * voting_size]`` 的输入在最后一维上做 ``kernel_size = voting_size, stride = voting_size`` 的平均池化
#
#         """
#         super().__init__()
#         self.voting_size = voting_size
#
#
#     def forward(self, x: torch.Tensor):
#         return F.avg_pool1d(x.unsqueeze(1), self.voting_size, self.voting_size).squeeze(1)

class VotingLayer(nn.Module):
    def __init__(self, voting_size: int = 10, strict: bool = True):
        super().__init__()
        self.voting_size = voting_size
        self.strict = strict  # True 时要求最后一维能整除

    def forward(self, x: torch.Tensor):
        v = self.voting_size
        L = x.size(-1)
        if self.strict and L % v != 0:
            raise ValueError(f"last dim ({L}) must be divisible by voting_size ({v})")
        # 若不严格，可先裁掉尾巴：x = x[..., : (L // v) * v]
        L = (L // v) * v
        x = x[..., :L]                      # 防止不整除时越界
        return x.reshape(*x.shape[:-1], L // v, v).mean(dim=-1)

class ClassificationHead(nn.Module):
    def __init__(self, embed_dims, num_classes):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        """
        Input: [T, B, embed_dims, num_patches]
        Output: [B, num_classes]
        """
        # Average over patches (dim=3)
        x = x.mean(-1)  # [T, B, embed_dims]
        # Average over time (dim=0)
        # Linear classification
        x = self.head(x)  # [B, num_classes]
        x = x.mean(0)  # [B, embed_dims]
        return x