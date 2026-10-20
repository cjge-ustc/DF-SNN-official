import torch
import math

# ========== 1. Sigmoid Surrogate ==========
class SigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        ctx.scale = scale
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        scale = ctx.scale
        sigmoid = torch.sigmoid(scale * input)
        surrogate_grad = scale * sigmoid * (1 - sigmoid)
        return grad_output * surrogate_grad, None

def sigmoid_surrogate(input, scale=5.0):
    return SigmoidSurrogate.apply(input, scale)


# ========== 2. Arctan Surrogate ==========
class ArctanSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        ctx.scale = scale
        return (input >= 0).float()

    # @staticmethod
    # def backward(ctx, grad_output):
    #     input, = ctx.saved_tensors
    #     scale = ctx.scale
    #     surrogate_grad = scale / (1 + (scale * input) ** 2)
    #     return grad_output * surrogate_grad, None

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        scale = ctx.scale
        return scale / 2 / (1 + (math.pi / 2 * scale * input).pow_(2)) * grad_output, None

def arctan_surrogate(input, scale=2.0):
    return ArctanSurrogate.apply(input, scale)


# class ArctanSurrogate(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, scale):
#         ctx.save_for_backward(input)
#         ctx.scale = scale
#         # 前向也返回连续值！
#         output = torch.sigmoid(scale * input)  # 或其他连续函数
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         scale = ctx.scale
#         surrogate_grad = scale / (1 + (scale * input) ** 2) / 2
#         return grad_output * surrogate_grad, None

# ========== 3. Piecewise Linear Surrogate ==========
class PiecewiseLinearSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        ctx.save_for_backward(input)
        ctx.gamma = gamma
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        gamma = ctx.gamma
        surrogate_grad = ((input >= -gamma) & (input <= gamma)).float() / (2 * gamma)
        return grad_output * surrogate_grad, None

def piecewise_linear_surrogate(input, gamma=1.0):
    return PiecewiseLinearSurrogate.apply(input, gamma)


# ========== 4. Gaussian Surrogate ==========
class GaussianSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sigma):
        ctx.save_for_backward(input)
        ctx.sigma = sigma
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigma = ctx.sigma
        surrogate_grad = torch.exp(- (input ** 2) / (2 * sigma ** 2)) / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
        return grad_output * surrogate_grad, None

def gaussian_surrogate(input, sigma=1.0):
    return GaussianSurrogate.apply(input, sigma)
