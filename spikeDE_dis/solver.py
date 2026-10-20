import torch
import math

from h5py.h5z import FLAG_SKIP_EDC

# from .utils import _flatten, ReversedListView, _flatten_convert_none_to_zeros
# from . import fdeint
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


from torch.utils import dlpack
try:  # Optional CuPy dependency
    import cupy  # type: ignore
    from cupy import cuda  # type: ignore
except Exception:  # pragma: no cover - maintain torch-only fallback
    cupy = None


def _as_alpha_component(alpha, index, reference):
    if isinstance(alpha, (tuple, list)):
        alpha_i = alpha[index]
    else:
        alpha_i = alpha
    if not isinstance(alpha_i, torch.Tensor):
        alpha_i = torch.tensor(alpha_i, dtype=reference.dtype, device=reference.device)
    else:
        alpha_i = alpha_i.to(device=reference.device, dtype=reference.dtype)
    return alpha_i


def _gamma_tensor(x):
    return torch.exp(torch.lgamma(x))


_CUPY_SUPPORTED_DTYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
)


def _assert_cupy_backend_ready(device, dtype):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if device.type != "cuda":
        raise RuntimeError("CuPy backend requires CUDA tensors.")
    cupy.cuda.Device(device.index or 0).use()
    if dtype not in _CUPY_SUPPORTED_DTYPES:
        raise TypeError(f"CuPy backend only supports float16/32/64, got {dtype}.")


def _cupy_weighted_history_sum(
    flat_history,
    coeff_cp,
    start_idx,
    history_len,
    state_shape,
    dtype,
    device,
):
    if history_len <= 0:
        return torch.zeros(state_shape, device=device, dtype=dtype)

    history_slice = flat_history.narrow(0, start_idx, history_len)
    history_slice = history_slice.reshape(history_len, -1)
    history_slice_cp = cupy.asarray(history_slice)
    result_cp = coeff_cp.dot(history_slice_cp)
    result = dlpack.from_dlpack(result_cp)
    return result.view(state_shape)


_WEIGHTED_SUM_KERNELS = {}


def _get_weighted_sum_kernel(dtype):
    if cupy is None:
        return None, None
    if dtype == torch.float32:
        type_name = "float"
        cp_dtype = cupy.float32
    elif dtype == torch.float64:
        type_name = "double"
        cp_dtype = cupy.float64
    else:
        return None, None
    kernel = _WEIGHTED_SUM_KERNELS.get(type_name)
    if kernel is None:
        code = f"""
        extern "C" __global__
        void weighted_sum(const {type_name}* coeff, const {type_name}* history, {type_name}* out,
                          int history_len, int state_size) {{
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < state_size) {{
                {type_name} acc = 0;
                for (int j = 0; j < history_len; ++j) {{
                    acc += coeff[j] * history[j * state_size + idx];
                }}
                out[idx] = acc;
            }}
        }}
        """
        kernel = cupy.RawKernel(code, "weighted_sum")
        _WEIGHTED_SUM_KERNELS[type_name] = kernel
    return kernel, cp_dtype


class _CupyWeightedHistorySum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, history):
        if coeff.numel() == 0:
            return torch.zeros(history.shape[1:], device=history.device, dtype=history.dtype)
        ctx.save_for_backward(coeff, history)
        coeff_flat = coeff.detach().contiguous()
        history_flat = history.detach().reshape(history.shape[0], -1).contiguous()
        kernel, cp_dtype = _get_weighted_sum_kernel(history.dtype)
        device_index = coeff.device.index if coeff.device.index is not None else 0
        with cupy.cuda.Device(device_index):
            if kernel is None:
                coeff_cp = cupy.from_dlpack(dlpack.to_dlpack(coeff_flat))
                history_cp = cupy.from_dlpack(dlpack.to_dlpack(history_flat))
                result_cp = coeff_cp.dot(history_cp)
            else:
                coeff_cp = cupy.from_dlpack(dlpack.to_dlpack(coeff_flat)).astype(cp_dtype)
                history_cp = cupy.from_dlpack(dlpack.to_dlpack(history_flat)).astype(cp_dtype)
                state_size = history_flat.shape[1]
                result_cp = cupy.empty(state_size, dtype=cp_dtype)
                threads = 256
                blocks = (state_size + threads - 1) // threads
                kernel((blocks,), (threads,), (coeff_cp, history_cp, result_cp, history.shape[0], state_size))
        result = dlpack.from_dlpack(result_cp)
        if result.dtype != history.dtype:
            result = result.to(dtype=history.dtype)
        return result.view(history.shape[1:])

    @staticmethod
    def backward(ctx, grad_out):
        coeff, history = ctx.saved_tensors
        if coeff.numel() == 0:
            return None, None
        history_flat = history.reshape(history.shape[0], -1).to(dtype=torch.float64)
        grad_out_flat = grad_out.reshape(-1).to(dtype=torch.float64)
        grad_coeff = (history_flat * grad_out_flat).sum(dim=1).to(dtype=coeff.dtype)
        grad_history = coeff.to(dtype=torch.float64).view(-1, *([1] * grad_out.dim())) * grad_out.to(dtype=torch.float64)
        grad_history = grad_history.to(dtype=history.dtype)
        return grad_coeff, grad_history


def _torch_tensor_to_cupy_flat(tensor: torch.Tensor):
    """Convert a CUDA torch tensor to a 1D CuPy view without copying."""
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if tensor.device.type != "cuda":
        raise TypeError("CuPy backend expects CUDA tensors.")
    return cupy.asarray(tensor.detach().reshape(-1).contiguous())


def _cupy_flat_to_torch(cp_array, shape, device, dtype):
    """Convert a 1D CuPy array back to a torch tensor and clone the contents."""
    tensor = torch.as_tensor(cp_array.reshape(-1), device=device, dtype=dtype)
    return tensor.view(shape).clone()


def _cupy_add_scaled_inplace(dst: torch.Tensor, src: torch.Tensor, scale) -> None:
    """Compute dst += scale * src on CUDA through CuPy, preserving dst storage."""
    _assert_cupy_backend_ready(dst.device, dst.dtype)
    if src.device != dst.device:
        src = src.to(device=dst.device)
    if src.dtype != dst.dtype:
        src = src.to(dtype=dst.dtype)
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch for CuPy axpy: dst={dst.shape}, src={src.shape}")

    dst_needs_copy_back = not dst.is_contiguous()
    dst_work = dst.detach().contiguous() if dst_needs_copy_back else dst.detach()
    src_work = src.detach().contiguous()

    device_index = dst.device.index if dst.device.index is not None else 0
    with cupy.cuda.Device(device_index):
        dst_cp = cupy.from_dlpack(dlpack.to_dlpack(dst_work.reshape(-1)))
        src_cp = cupy.from_dlpack(dlpack.to_dlpack(src_work.reshape(-1)))
        dst_cp += scale * src_cp

    if dst_needs_copy_back:
        dst.copy_(dst_work.view_as(dst))


def euler_integrate_tuple(ode_func, y0_tuple, t_grid: torch.Tensor):
    """
    手写 Euler 积分（显式），支持 tuple 状态。
    - ode_func: callable(t, y_tuple) -> tuple(dy_i)
    - y0_tuple: tuple(tensor, ...) 初始状态 == 0
    - t_grid: 1D Tensor，[t0, t1, ..., tN]，步长可不等（会逐段用 dt = t_{k+1}-t_k）
    返回：tuple( Y_i )，其中 Y_i 的形状是 (N, *y0_i.shape)
    """
    assert isinstance(y0_tuple, tuple) 
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    # Initialize history lists for each component
    final_spike_history = []

    # Clone initial values
    y_current = list(y0_tuple)

    # Euler integration: y_{k+1} = y_k + dt * f(t_k, y_k)
    for k in range(N - 1):
        tk = t_grid[k]
        dt = t_grid[k + 1] - t_grid[k]  # Scalar tensor, will broadcast automatically
        dy = ode_func(tk, tuple(y_current))  # Expect tuple return, consistent with y structure
        # assert isinstance(dy, tuple) and len(dy) == len(y)

        # Update all integrated components except the last one
        for i in range(len(y_current) - 1):
            y_current[i] = y_current[i] + dt * dy[i]
        # Final element is the output spike (pass-through)
        y_current[-1] = dy[-1]
        final_spike_history.append(dy[-1])

    return final_spike_history

@torch.compile(mode="reduce-overhead")
def gl_integrate_tuple(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Integrate a Riemann-Liouville fractional differential equation using the
    Grünwald-Letnikov (GL) discretization method.

    Solves equations of the form: D^α y(t) = f(t, y)

    Reference:
        Jahanshahi et al., "Determining the chaotic behavior in a fractional-order
        finance system with negative parameters." Nonlinear Dynamics 94.2 (2018): 1303-1317.

    Args:
        ode_func: Callable (t, y_tuple) -> tuple.
                  Right-hand side function returning derivatives with the same structure as y_tuple.
        y0_tuple: Tuple of initial state tensors for each component.
        alpha: Fractional derivative order, must be in the range (0, 1).
        t_grid: 1D tensor of uniformly spaced time points. t_grid[0] is the initial time.
        memory: Number of past steps to retain for the convolution sum.
                If None or -1, the full history is used (short-memory principle disabled).

    Returns:
        list of lists: y_history where y_history[i][k] is the state of component i at time step k+1.
                       - Length of each y_history[i] is len(t_grid) - 1 (excludes initial condition).
                       - y_history[:-1] contains membrane potential histories for each layer.
                       - y_history[-1] contains spike output history (final SNN layer, not integrated).
    """
    assert isinstance(y0_tuple, tuple) 
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[-2]) # uniform step size
    h_alpha = torch.pow(h, alpha)

    # GL coefficients: need up to c[N]. Build as a list to keep alpha differentiable.
    c_values = [torch.ones((), dtype=dtype, device=device)]
    for j in range(1, N + 1):
        c_values.append((1 - (1 + alpha) / j) * c_values[-1])
    c = torch.stack(c_values)

    # Initialize with y_0 (clone to avoid modifying input)
    y_current = list(y0_tuple) #tuple(y.clone() for y in y0_tuple)
    # History: y_history[i][j] stores y_j for component i
    # Initialize with y_0
    y_history = [[] for y in y0_tuple]
    # y_history = [[y.clone()] for y in y0_tuple]

    for k in range(N - 1):
        t_k = t_grid[k]
        # Evaluate f(t_k, y_k)

        dy = ode_func(t_k, tuple(y_current))
        # assert isinstance(dy, tuple) and len(dy) == len(y_current)

        # Determine memory range
        if memory is None or memory == -1:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)

        start_idx = max(0, k + 1 - memory_length)
        history_len = k + 1 - start_idx

        # Update all integrated components except the last one (spike)
        for i in range(len(y_current) - 1):
            # Accumulate: Σ c_{k+1-j} * y_j for j from start_idx to k
            if k > 0:
                convolution_sum = 0#torch.zeros_like(y[i])
                for j in range(start_idx, k):
                    # GL coefficient for this lag
                    convolution_sum = convolution_sum + c[k+1-j] * y_history[i][j]
                # here we assume at time k, we have k elements (without y0=0)
                # the most restrict formulation should be convolution_sum + c[k-j] * y_history[i][j]
                # which however seems do have have good numerical stability when h=1.0
            else:
                convolution_sum = 0

            # y_{k+1} = h^alpha * f_k - convolution_sum
            y_current[i] = h_alpha * dy[i] - convolution_sum
            y_history[i].append(y_current[i])

        # Pass-through output (spike) - last component is not integrated
        y_current[-1] = dy[-1]
        y_history[-1].append(y_current[-1])

    return y_history

@torch.compile(mode="reduce-overhead")
def trap_integrate_tuple(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Integrate a Riemann-Liouville fractional differential equation using the
     Product Trapezoidal method.

    Solves equations of the form: D^α y(t) = f(t, y)

    Reference:
        Book: Fractional calculus: models and numerical methods, Page 56

    Args:
        ode_func: Callable (t, y_tuple) -> tuple.
                  Right-hand side function returning derivatives with the same structure as y_tuple.
        y0_tuple: Tuple of initial state tensors for each component.
        alpha: Fractional derivative order, must be in the range (0, 1).
        t_grid: 1D tensor of uniformly spaced time points. t_grid[0] is the initial time.
        memory: Number of past steps to retain for the convolution sum.
                If None or -1, the full history is used (short-memory principle disabled).

    Returns:
        list of lists: y_history where y_history[i][k] is the state of component i at time step k+1.
                       - Length of each y_history[i] is len(t_grid) - 1 (excludes initial condition).
                       - y_history[:-1] contains membrane potential histories for each layer.
                       - y_history[-1] contains spike output history (final SNN layer, not integrated).
    """
    assert isinstance(y0_tuple, tuple) 

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = (t_grid[-1] - t_grid[-2]) # uniform step size
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    # Initialize with y_0 (clone to avoid modifying input)
    y_current = list(y0_tuple)  # tuple(y.clone() for y in y0_tuple)
    # History: y_history[i][j] stores y_j for component i
    # Initialize with y_0
    y_history = [[] for y in y0_tuple]
    # y_history = [[y.clone()] for y in y0_tuple]

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)
        history_len = k + 1 - start_idx

        # Compute A_{j,k+1} weights for indices from start_idx to k
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

        # General formula for j >= 1:
        # A_{j,k+1} = (k+2-j)^{1-α} + (k-j)^{1-α} - 2(k+1-j)^{1-α}
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        A_j_kp1 = kjp2 + kj - 2 * kjp1

        # Special handling for j=0 if it's in the range:
        # A_{0,k+1} = k^{1-α} - (k+α)(k+1)^{-α}
        if start_idx == 0:
            k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
            kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -alpha)
            first_value = k_power - (k + alpha) * kp1_neg_alpha
            A_j_kp1 = torch.cat([first_value.reshape(1), A_j_kp1[1:]])

        # Update first (num_components - 1) state components using fractional integration
        for i in range(len(y0_tuple) - 1):
            # Compute convolution sum: sum_{j=start_idx}^{k} A_{j,k+1} * y_j[i]
            if k > 0:
                convolution_sum = 0#torch.zeros_like(y[i])
                for j in range(start_idx, k): #range(start_idx, k) because we have k (not k+1) elements at time k
                    # GL coefficient for this lag
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + A_j_kp1[local_idx] * y_history[i][j]
                # here we assume at time k, we have k elements (without y0=0)
                # the most restrict formulation should be local_idx = j - start_idx + 1
                # which however seems do have have good numerical stability
            else:
                convolution_sum = 0

            # y_{k+1} = Γ(2-α) * h^α * f_k - convolution_sum
            y_current[i] = h_alpha_gamma * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

        # Pass-through output (spike) - last component is not integrated
        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    return y_history



@torch.compile(mode="reduce-overhead")
def l1_integrate_tuple(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Integrate a Caputo fractional differential equation using the L1 method.

    Solves equations of the form: D^α y(t) = f(t, y)

    Reference:
        A compact finite difference scheme for the fractional sub-diffusion equations,
        Journal of Computational Physics, vol. 230, no. 3, pp. 586–595, 2011.

    Args:
        ode_func: Callable (t, y_tuple) -> tuple.
                  Right-hand side function returning derivatives with the same structure as y_tuple.
        y0_tuple: Tuple of initial state tensors for each component.
        alpha: Fractional derivative order, must be in the range (0, 1).
        t_grid: 1D tensor of uniformly spaced time points. t_grid[0] is the initial time.
        memory: Number of past steps to retain for the convolution sum.
                If None or -1, the full history is used (short-memory principle disabled).

    Returns:
        list of lists: y_history where y_history[i][k] is the state of component i at time step k+1.
                       - Length of each y_history[i] is len(t_grid) - 1 (excludes initial condition).
                       - y_history[:-1] contains membrane potential histories for each layer.
                       - y_history[-1] contains spike output history (final SNN layer, not integrated).
    """
    assert isinstance(y0_tuple, tuple) 

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = (t_grid[-1] - t_grid[-2]) # uniform step size
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha


    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]

    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute c_j^(k) weights for indices from start_idx to k
        # General formula for j >= 1: c_j^(k) = (k-j+2)^{1-α} - 2(k-j+1)^{1-α} + (k-j)^{1-α}
        # Special case for j = 0: c_0^(k) = -[(k+1)^{1-α} - k^{1-α}]
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        c_j_k = kjp2 - 2 * kjp1 + kj

        # Special handling for j=0 if it's in the range
        if start_idx == 0:
            kp1_power = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), one_minus_alpha)
            k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
            c_j_k[0] = -(kp1_power - k_power)

        # Update integrated states
        for i in range(len(y0_tuple) - 1):
            # Compute convolution sum: sum_{j=start_idx}^{k} c_j^(k) * y_j[i]
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):  # range(start_idx, k) because we have k elements (without y0=0)
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + c_j_k[local_idx] * y_history[i][j]
                    # the most restrict formulation should be local_idx = j - start_idx + 1
            else:
                convolution_sum = 0

            # y_{k+1} = h^α * Γ(2-α) * f_k - convolution_sum
            y_current[i] = h_alpha_gamma * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

        # Pass-through output (spike) - last component is not integrated
        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    return y_history

@torch.compile(mode="reduce-overhead")
def pred_integrate_tuple(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Integrate a Caputo fractional differential equation using
    one-step Adams-Bashforth (Euler) predictor method,

    Solves equations of the form: D^α y(t) = f(t, y)

    Reference:
        Detailed error analysis for a fractional Adams method.
        Numerical algorithms 36.1 (2004): 31-52.

    Args:
        ode_func: Callable (t, y_tuple) -> tuple.
                  Right-hand side function returning derivatives with the same structure as y_tuple.
        y0_tuple: Tuple of initial state tensors for each component.
        alpha: Fractional derivative order, must be in the range (0, 1).
        t_grid: 1D tensor of uniformly spaced time points. t_grid[0] is the initial time.
        memory: Number of past steps to retain for the convolution sum.
                If None or -1, the full history is used (short-memory principle disabled).

    Returns:
        list of lists: y_history where y_history[i][k] is the state of component i at time step k+1.
                       - Length of each y_history[i] is len(t_grid) - 1 (excludes initial condition).
                       - y_history[:-1] contains membrane potential histories for each layer.
                       - y_history[-1] contains spike output history (final SNN layer, not integrated).
    """
    assert isinstance(y0_tuple, tuple) 

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = (t_grid[-1] - t_grid[-2]) # uniform step size
    # gamma_alpha = 1 / math.gamma(alpha)
    # h_alpha_over_alpha = torch.pow(h, alpha) / alpha *
    h_alpha_over_alpha = torch.pow(h, alpha) / (alpha * math.gamma(alpha))

    # Number of components to integrate (all except last spike component)
    n_integrate = len(y0_tuple) - 1

    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]

    # History for function evaluations (only for integrated components)
    fhistory = [[] for _ in range(n_integrate)]

    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Store function evaluations for integrated components
        for i in range(n_integrate):
            fhistory[i].append(f_k[i])

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute weights b_{j,k+1} for indices from start_idx to k
        # b_{j,k+1} = (h^α / α) * [(k+1-j)^α - (k-j)^α]
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
        # the most restrict formulation should be
        b_j_kp1 = h_alpha_over_alpha * (
                torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
        )
        # b_j_kp1 = h_alpha_over_alpha * (
        #         torch.pow(k + 2 - j_vals, alpha) - torch.pow(k + 1 - j_vals, alpha)
        # )

        # Update integrated states
        for i in range(n_integrate):
            convolution_sum = 0
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx
                convolution_sum = convolution_sum + b_j_kp1[local_idx] * fhistory[i][j]
            y_current[i] = convolution_sum
            y_history[i].append(y_current[i])

        # Pass-through output (spike) - last component is not integrated
        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])
    del fhistory
    return y_history


def gl_integrate_tuple_per_state(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """Torch-only GL solver that allows one alpha tensor per tuple state."""
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[-2])

    n_integrate = len(y0_tuple) - 1
    alpha_tuple = [_as_alpha_component(alpha, i, y0_tuple[i]) for i in range(n_integrate)]
    h_alpha = [torch.pow(h, alpha_i) for alpha_i in alpha_tuple]

    c_values = []
    for alpha_i in alpha_tuple:
        c_i = [torch.ones_like(alpha_i)]
        for j in range(1, N + 1):
            c_i.append((1 - (1 + alpha_i) / j) * c_i[-1])
        c_values.append(c_i)

    y_current = list(y0_tuple)
    y_history = [[] for _ in y0_tuple]

    for k in range(N - 1):
        t_k = t_grid[k]
        dy = ode_func(t_k, tuple(y_current))
        if memory is None or memory == -1:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"
        start_idx = max(0, k + 1 - memory_length)

        for i in range(n_integrate):
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):
                    convolution_sum = convolution_sum + c_values[i][k + 1 - j] * y_history[i][j]
            else:
                convolution_sum = 0
            y_current[i] = h_alpha[i] * dy[i] - convolution_sum
            y_history[i].append(y_current[i])

        y_current[-1] = dy[-1]
        y_history[-1].append(y_current[-1])

    return y_history


def trap_integrate_tuple_per_state(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """Torch-only product trapezoidal solver with one alpha tensor per tuple state."""
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[-2])

    n_integrate = len(y0_tuple) - 1
    alpha_tuple = [_as_alpha_component(alpha, i, y0_tuple[i]) for i in range(n_integrate)]
    h_alpha_gamma = [torch.pow(h, alpha_i) * _gamma_tensor(2 - alpha_i) for alpha_i in alpha_tuple]
    one_minus_alpha = [1 - alpha_i for alpha_i in alpha_tuple]

    y_current = list(y0_tuple)
    y_history = [[] for _ in y0_tuple]

    for k in range(N - 1):
        t_k = t_grid[k]
        f_k = ode_func(t_k, tuple(y_current))
        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"
        start_idx = max(0, k + 1 - memory_length)

        for i in range(n_integrate):
            alpha_i = alpha_tuple[i]
            oma_i = one_minus_alpha[i]
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):
                    kjp2 = torch.pow(torch.tensor(k + 2 - j, dtype=dtype, device=device), oma_i)
                    kj = torch.pow(torch.tensor(k - j, dtype=dtype, device=device), oma_i)
                    kjp1 = torch.pow(torch.tensor(k + 1 - j, dtype=dtype, device=device), oma_i)
                    coeff = kjp2 + kj - 2 * kjp1
                    if j == 0:
                        k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), oma_i)
                        kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -alpha_i)
                        coeff = k_power - (k + alpha_i) * kp1_neg_alpha
                    convolution_sum = convolution_sum + coeff * y_history[i][j]
            else:
                convolution_sum = 0
            y_current[i] = h_alpha_gamma[i] * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    return y_history


def pred_integrate_tuple_per_state(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """Torch-only predictor solver with one alpha tensor per tuple state."""
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[0]) / (N - 1)

    n_integrate = len(y0_tuple) - 1
    alpha_tuple = [_as_alpha_component(alpha, i, y0_tuple[i]) for i in range(n_integrate)]
    h_alpha_over_alpha = [
        torch.pow(h, alpha_i) / (alpha_i * _gamma_tensor(alpha_i))
        for alpha_i in alpha_tuple
    ]

    y_history = [[] for _ in y0_tuple]
    fhistory = [[] for _ in range(n_integrate)]
    y_current = list(y0_tuple)

    for k in range(N - 1):
        t_k = t_grid[k]
        f_k = ode_func(t_k, tuple(y_current))
        for i in range(n_integrate):
            fhistory[i].append(f_k[i])

        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"
        start_idx = max(0, k + 1 - memory_length)

        for i in range(n_integrate):
            alpha_i = alpha_tuple[i]
            convolution_sum = 0
            for j in range(start_idx, k + 1):
                coeff = h_alpha_over_alpha[i] * (
                    torch.pow(torch.tensor(k + 1 - j, dtype=dtype, device=device), alpha_i)
                    - torch.pow(torch.tensor(k - j, dtype=dtype, device=device), alpha_i)
                )
                convolution_sum = convolution_sum + coeff * fhistory[i][j]
            y_current[i] = convolution_sum
            y_history[i].append(y_current[i])

        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    del fhistory
    return y_history


def step_dynamics(ode_func, y0_tuple, t_grid: torch.Tensor):
    """
    Step through a discrete-time dynamical system, collecting spike outputs.

    Repeatedly applies the update function to evolve the state over time.
    Unlike numerical integrators (Euler, GL), no dt scaling is applied.

    The update function directly computes the next state!

    Args:
        ode_func: Callable (t, y_tuple) -> tuple. State update function.
        y0_tuple: Tuple of initial state tensors.
        t_grid: 1D tensor of time points (length T+1).

    Returns:
        list: Spike outputs (last component of state) at each time step.
    """
    assert isinstance(y0_tuple, tuple) 
    N = len(t_grid)
    assert N >= 2

    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    y = list(y0_tuple)
    spike_history = []

    for k in range(N - 1):
        t_k = t_grid[k]
        y = ode_func(t_k, tuple(y))
        spike_history.append(y[-1])

    return spike_history


# def euler_integrate_tuple(ode_func, y0_tuple, t_grid: torch.Tensor):
#     """
#     修改版：对LIF神经元使用半隐式方法
#     """
#     assert isinstance(y0_tuple, tuple) 
#     T = t_grid.numel()
#     assert T >= 1
#     device = y0_tuple[0].device
#     dtype = y0_tuple[0].dtype
#     t_grid = t_grid.to(device=device, dtype=dtype)
#
#     outs = [torch.empty((T,) + y0.shape, device=device, dtype=dtype) for y0 in y0_tuple]
#     y = [y0.clone() for y0 in y0_tuple]
#
#     for i, y0 in enumerate(y):
#         outs[i][0].copy_(y0)
#
#     for k in range(T - 1):
#         tk = t_grid[k]
#         dt = t_grid[k + 1] - tk
#
#         # 第一次调用：获取基本导数
#         dy = ode_func(tk, tuple(y))
#         assert isinstance(dy, tuple) and len(dy) == len(y)
#
#         # 预测步（用于计算spike）
#         y_predict = [y[i] + dt * dy[i] for i in range(len(y))]
#
#         # 第二次调用：基于预测值重新计算（关键！）
#         # 这模拟了隐式方法
#         dy_corrected = ode_func(tk, tuple(y_predict))
#
#         # 使用修正后的导数
#         for i in range(len(y)):
#             if True:  # 假设第一个是膜电位
#                 # 对膜电位使用修正的导数
#                 y[i] = y[i] + dt * dy_corrected[i]
#             else:
#                 # 其他变量使用原始导数
#                 y[i] = y[i] + dt * dy[i]
#             outs[i][k + 1].copy_(y[i])
#     return tuple(outs)



@torch.compile(mode="reduce-overhead")
def trap_integrate_tuple(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Integrate a Riemann-Liouville fractional differential equation using the
     Product Trapezoidal method.

    Solves equations of the form: D^α y(t) = f(t, y)

    Reference:
        Book: Fractional calculus: models and numerical methods, Page 56

    Args:
        ode_func: Callable (t, y_tuple) -> tuple.
                  Right-hand side function returning derivatives with the same structure as y_tuple.
        y0_tuple: Tuple of initial state tensors for each component.
        alpha: Fractional derivative order, must be in the range (0, 1).
        t_grid: 1D tensor of uniformly spaced time points. t_grid[0] is the initial time.
        memory: Number of past steps to retain for the convolution sum.
                If None or -1, the full history is used (short-memory principle disabled).

    Returns:
        list of lists: y_history where y_history[i][k] is the state of component i at time step k+1.
                       - Length of each y_history[i] is len(t_grid) - 1 (excludes initial condition).
                       - y_history[:-1] contains membrane potential histories for each layer.
                       - y_history[-1] contains spike output history (final SNN layer, not integrated).
    """
    assert isinstance(y0_tuple, tuple)

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = (t_grid[-1] - t_grid[-2]) # uniform step size
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    # Initialize with y_0 (clone to avoid modifying input)
    y_current = list(y0_tuple)  # tuple(y.clone() for y in y0_tuple)
    # History: y_history[i][j] stores y_j for component i
    # Initialize with y_0
    y_history = [[] for y in y0_tuple]
    # y_history = [[y.clone()] for y in y0_tuple]

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute A_{j,k+1} weights for indices from start_idx to k
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

        # General formula for j >= 1:
        # A_{j,k+1} = (k+2-j)^{1-α} + (k-j)^{1-α} - 2(k+1-j)^{1-α}
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        A_j_kp1 = kjp2 + kj - 2 * kjp1

        # Special handling for j=0 if it's in the range:
        # A_{0,k+1} = k^{1-α} - (k+α)(k+1)^{-α}
        if start_idx == 0:
            k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
            kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -alpha)
            first_value = k_power - (k + alpha) * kp1_neg_alpha
            A_j_kp1 = torch.cat([first_value.reshape(1), A_j_kp1[1:]])

        # Update first (num_components - 1) state components using fractional integration
        for i in range(len(y0_tuple) - 1):
            # Compute convolution sum: sum_{j=start_idx}^{k} A_{j,k+1} * y_j[i]
            if k > 0:
                convolution_sum = 0#torch.zeros_like(y[i])
                for j in range(start_idx, k): #range(start_idx, k) because we have k (not k+1) elements at time k
                    # GL coefficient for this lag
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + A_j_kp1[local_idx] * y_history[i][j]
                # here we assume at time k, we have k elements (without y0=0)
                # the most restrict formulation should be local_idx = j - start_idx + 1
                # which however seems do have have good numerical stability
            else:
                convolution_sum = 0

            # y_{k+1} = Γ(2-α) * h^α * f_k - convolution_sum
            y_current[i] = h_alpha_gamma * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

        # Pass-through output (spike) - last component is not integrated
        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    return y_history



#------------------------------------ the following is distributed-order ------------------------------------#

@torch.compile(mode="reduce-overhead")
def glmethod_multiterm_integrate_tuple(ode_func, y0_tuple, beta, t_grid: torch.Tensor, memory=None,
                                       multi_coefficient=None):
    """
    Grünwald-Letnikov method for multi-term RL fractional differential equations with tuple state.
    Σ w_j D^{β_j} y(t) = f(t,y)

    Based on the distributed GL scheme where the aggregated weights encode
    the multi-term structure through: c̃_m = Σ w_j h^{-α_j} c_m^{(α_j)}

    Args:
        ode_func: callable(t, y_tuple) returns tuple with same structure as y_tuple
        y0_tuple: tuple of initial states
        beta: Tensor of fractional orders, each in the range (0,1), shape (n_terms,)
        t_grid: time points sequence (must be equally spaced)
        memory: history window size (None means use full history)
        multi_coefficient: Tensor of weights w_j for each term, shape (n_terms,)

    Returns:
        tuple of lists: history of all state components at all time points
    """
    assert isinstance(y0_tuple, tuple) 
    assert multi_coefficient is not None, "multi_coefficient must be provided"

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = (t_grid[-1] - t_grid[-2]) # uniform step size

    # Ensure beta and coefficient are tensors on the correct device
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=dtype, device=device)
    else:
        beta = beta.to(device=device, dtype=dtype)

    if not isinstance(multi_coefficient, torch.Tensor):
        coefficient = torch.tensor(multi_coefficient, dtype=dtype, device=device)
    else:
        coefficient = multi_coefficient.to(device=device, dtype=dtype)

    n_terms = len(beta)
    assert len(coefficient) == n_terms, "beta and coefficient must have the same length"

    # Compute GL coefficients c_m^{(α_j)} for each fractional order
    # c[j, m] = c_m^{(α_j)} where c_0 = 1, c_m = (1 - (1+α)/m) * c_{m-1}
    c = torch.zeros(n_terms, N + 1, dtype=dtype, device=device)
    c[:, 0] = 1.0
    for m in range(1, N + 1):
        # Vectorized over all terms: c[j,m] = (1 - (1+β_j)/m) * c[j,m-1]
        c[:, m] = (1 - (1 + beta) / m) * c[:, m - 1]

    # Compute h^{-α_j} for each term
    h_neg_power = torch.pow(h, -beta)  # Shape: (n_terms,)

    # Compute distributed GL weights: c̃_m = Σ w_j * h^{-α_j} * c_m^{(α_j)}
    # Vectorized: (n_terms,) * (n_terms,) * (n_terms, N+1) -> sum over dim 0 -> (N+1,)
    weighted_h = coefficient * h_neg_power  # Shape: (n_terms,)
    c_tilde = torch.sum(weighted_h.unsqueeze(1) * c, dim=0)  # Shape: (N+1,)

    # c̃_0 = Σ w_j * h^{-α_j} (since c_0^{(α_j)} = 1 for all j)
    c_tilde_0 = c_tilde[0]
    c_tilde_0_inv = 1.0 / c_tilde_0

    # Number of components to integrate (all except last spike component)
    n_integrate = len(y0_tuple) - 1

    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]

    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Update integrated states (all except last spike component)
        for i in range(n_integrate):
            # Compute convolution sum: Σ c̃_{k+1-j} * y_j[i] for j from start_idx to k
            # This corresponds to Σ_{m=1}^{min(k+1,M)} c̃_m * y_{k+1-m}
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):  # range(start_idx, k) because we have k elements (without y0=0)
                    coeff_idx = k - j  # history[j] is y_{j+1}; nearest state uses c_1.
                    convolution_sum = convolution_sum + c_tilde[coeff_idx] * y_history[i][j]
            else:
                convolution_sum = 0

            # y_{k+1} = (1/c̃_0) * (f_k - convolution_sum)
            y_current[i] = c_tilde_0_inv * (f_k[i] - convolution_sum)
            y_history[i].append(y_current[i])

        # Pass-through output (spike) - last component is not integrated
        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    return y_history



#------------------------------------ the following is adjoint training code ------------------------------------#


def fdeint_adjoint(func, y0_tuple, alpha, t_grid, method, memory=None, backend="torch",
                   multi_coefficient=None):
    # params = tuple(p for p in func.parameters())  # 或只取 requires_grad=True 的
    params = find_parameters(func)
    n_state = len(y0_tuple)
    n_params = len(params)
    # 注意：apply 不接受关键字参数；把计数放在前两位最简单
    return FDEAdjointMethod.apply(
        func, n_state, n_params, *y0_tuple, alpha, t_grid, method, backend, *params, memory, multi_coefficient
    )


class FDEAdjointMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ode_func, n_state, n_params, *args):
        # n_state, n_params 是 Python int（非 Tensor）；其梯度应返回 None
        """
        Forward pass
        Args:
            ode_func: ODE 函数
            y0_tuple: 初始状态 (tuple of tensors)
            alpha: 参数
            t_grid: 时间网格
            method: 积分方法
            func_params: 模型参数 (tuple)
            memory: 额外内存参数
        """
        n_state = int(n_state); n_params = int(n_params)

        # 解析位置参数： y0_1,...,y0_n, alpha, t_grid, method, p1,...,pm, memory, multi_coefficient
        y0_tuple   = tuple(args[:n_state])                               # Tensors
        alpha       = args[n_state]                                      # Tensor 或 float（若要学习，必须是 Tensor）
        t_grid     = args[n_state + 1]                                  # Tensor
        method     = args[n_state + 2]                                  # str / enum（非 Tensor）
        backend    = args[n_state + 3]
        func_params= tuple(args[n_state + 4 : n_state + 4 + n_params])   # Tensors (Parameters)
        memory     = args[n_state + 4 + n_params]                       # 任意对象（非 Tensor）
        multi_coefficient = args[n_state + 5 + n_params]

        with torch.no_grad():
            solver = _get_solver(backend, method, "forward")
            if method == "glmulti":
                yhistory = solver(
                    ode_func=ode_func,
                    y0_tuple=y0_tuple,
                    alpha=alpha,
                    t_grid=t_grid,
                    memory=memory,
                    multi_coefficient=multi_coefficient
                )
            else:
                yhistory = solver(
                    ode_func=ode_func,
                    y0_tuple=y0_tuple,
                    alpha=alpha,
                    t_grid=t_grid,
                    memory=memory
                )

        # 检查是否需要梯度
        y0_needs_grad = any(t.requires_grad for t in y0_tuple)
        params_need_grad = any(p.requires_grad for p in func_params) if func_params else False
        multi_coeff_needs_grad = (
            torch.is_tensor(multi_coefficient) and multi_coefficient.requires_grad
        )

        ctx.n_state = n_state
        ctx.n_params = n_params
        ctx.multi_coefficient = multi_coefficient
        if y0_needs_grad or params_need_grad or multi_coeff_needs_grad:
            # 保存必要的张量用于反向传播
            # 注意：保存整个 tuple 对象，而不是展开的张量
            ctx.save_for_backward(t_grid)
            ctx.func_params = func_params  # 同样保存为属性
            ctx.yhistory = yhistory  # 保存最终状态
            ctx.ode_func = ode_func
            ctx.alpha = alpha
            ctx.method = method
            ctx.memory = memory
            ctx.backend = backend
            ctx.multi_coefficient = multi_coefficient
            ctx.y0_tuple = y0_tuple

        # 返回结果
        # if method == 'euler':
        #     outs = tuple(yhistory)
        # else:
        #     # 如果 yhistory 是嵌套列表，取最后一个
        outs = tuple([y[-1] for y in yhistory])

        return outs

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Backward pass - 计算梯度
        """
        # 早退：不需要反传时，返回正确数量的 None
        if not hasattr(ctx, 'yhistory'):
            n_state = ctx.n_state
            n_params = ctx.n_params
            grads = []
            grads.append(None)  # ode_func
            grads.append(None)  # n_state
            grads.append(None)  # n_params
            grads.extend([None] * n_state)  # y0_1,...,y0_n
            grads.append(None)  # alpha
            grads.append(None)  # t_grid
            grads.append(None)  # method
            grads.append(None)  # backend
            grads.extend([None] * n_params)  # p1,...,pm
            grads.append(None)  # memory
            grads.append(None)  # multi_coefficient
            return tuple(grads)

        # 恢复保存的张量和属性
        t_grid = ctx.saved_tensors[0]
        func_params = ctx.func_params
        yhistory = ctx.yhistory
        # yhistory is the last states for euler and the full history for other methods

        func = ctx.ode_func
        alpha = ctx.alpha
        method = ctx.method
        memory = ctx.memory
        n_tensors = len(yhistory)

        # 创建 augmented dynamics
        class AugDynamics:
            def __init__(self, func, n_tensors, func_params):
                self.func = func
                self.n_tensors = n_tensors
                self.f_params = func_params  # 使用传入的参数

            def __call__(self, t, y_aug):
                y, adj_y, adj_params = y_aug

                with torch.set_grad_enabled(True):
                    # detach 并设置 requires_grad
                    y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                    func_eval = self.func(t, y)

                    # 计算 VJP
                    vjp_y_and_params = torch.autograd.grad(
                        func_eval,
                        y + self.f_params,
                        tuple(adj_y),
                        allow_unused=True,
                        retain_graph=False,  # 不保留图
                        create_graph=False
                    )

                vjp_y = vjp_y_and_params[:self.n_tensors]
                vjp_params = vjp_y_and_params[self.n_tensors:]

                # 处理 None 梯度
                vjp_y = tuple(
                    torch.zeros_like(y_) if vjp_y_ is None else vjp_y_
                    for vjp_y_, y_ in zip(vjp_y, y)
                )


                vjp_params = tuple(
                        torch.zeros_like(p) if vp is None else vp
                        for vp, p in zip(vjp_params, self.f_params)
                    )


                return (func_eval, vjp_y, vjp_params)

        # 创建 augmented dynamics 实例
        augmented_dynamics = AugDynamics(func, n_tensors, func_params)
        t_grid_flip = t_grid.flip(0)

        with torch.no_grad():
            adj_y = grad_output

            # 初始化参数梯度
            if func_params:
                adj_params = tuple(torch.zeros_like(p) for p in func_params)
            else:
                adj_params = ()

            # 设置初始增广状态
            # 注意：yhistory 作为初始 y 状态
            aug_y0 = ([], adj_y, adj_params)

            # 调用反向求解器
            solver = _get_solver(ctx.backend, method, "backward")
            if method == "glmulti":
                adj_y, adj_params = solver(
                    augmented_dynamics,
                    aug_y0,
                    alpha,
                    t_grid_flip,
                    yhistory,  # 传递最终状态
                    memory,
                    ctx.multi_coefficient,
                )
            else:
                adj_y, adj_params = solver(
                    augmented_dynamics,
                    aug_y0,
                    alpha,
                    t_grid_flip,
                    yhistory,  # 传递最终状态
                    memory
                )
        grad_multi_coeff = None
        if (
            ctx.multi_coefficient is not None
            and torch.is_tensor(ctx.multi_coefficient)
            and ctx.multi_coefficient.requires_grad
        ):
            with torch.set_grad_enabled(True):
                y0_tuple = tuple(y_.detach() for y_ in ctx.y0_tuple)
                alpha = ctx.alpha
                if torch.is_tensor(alpha):
                    alpha = alpha.detach()
                yhistory_fwd = forward_glmulti(
                    ode_func=ctx.ode_func,
                    y0_tuple=y0_tuple,
                    alpha=alpha,
                    t_grid=t_grid,
                    memory=memory,
                    multi_coefficient=ctx.multi_coefficient,
                )
                outs = tuple([y[-1] for y in yhistory_fwd])
                grad_multi_coeff = torch.autograd.grad(
                    outs,
                    ctx.multi_coefficient,
                    grad_outputs=grad_output,
                    allow_unused=True,
                )[0]

        # 清理
        # del augmented_dynamics
        # del ctx.yhistory

        # 在最后，确保清理所有局部变量
        del augmented_dynamics
        del yhistory  # 也要删除局部变量
        del func
        del func_params
        del ctx.yhistory
        del ctx.ode_func
        del ctx.func_params
        del ctx.alpha
        del ctx.method
        if hasattr(ctx, 'y0_tuple'):
            del ctx.y0_tuple
        del ctx.backend
        del ctx.multi_coefficient

        # 准备返回值
        # 返回格式：(grad_func, grad_y0_tuple, grad_alpha, grad_t_grid, grad_method, grad_func_params, grad_memory)
        grad_y0 = adj_y
        grad_params = adj_params

        return None, None, None, *grad_y0, None, None, None, None, *grad_params, None, grad_multi_coeff


def forward_euler_wo_history(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    手写 Euler 积分（显式），支持 tuple 状态。
    - ode_func: callable(t, y_tuple) -> tuple(dy_i)
    - y0_tuple: tuple(tensor, ...) 初始状态 == 0
    - t_grid: 1D Tensor，[t0, t1, ..., tN]，步长可不等（会逐段用 dt = t_{k+1}-t_k）
    返回：tuple( Y_i )，其中 Y_i 的形状是 (N, *y0_i.shape)
    """
    assert isinstance(y0_tuple, tuple) 
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    # Initialize history lists for each component

    # Clone initial values
    y = list(y0_tuple)

    # Euler integration: y_{k+1} = y_k + dt * f(t_k, y_k)
    for k in range(N - 1):
        tk = t_grid[k]
        dt = t_grid[k + 1] - t_grid[k]  # Scalar tensor, will broadcast automatically
        dy = ode_func(tk, tuple(y))  # Expect tuple return, consistent with y structure
        # assert isinstance(dy, tuple) and len(dy) == len(y)
        # Update all integrated components except the last one
        # for i in range(len(y)):
        #     y[i] = y[i] + dt * dy[i]
        for i in range(len(y)):
            y[i].add_(dy[i], alpha=dt)
    return y


def backward_euler_wo_history(ode_func, y_aug, alpha, t_grid, y_finalstate, memory=None):
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs((t_grid[-1] - t_grid[-2]))  # uniform step size
        h = float(h)

        _, adj_y0, adj_params0 = y_aug
        # 初始化
        # y_state = [x.detach().clone() for x in y_finalstate]
        # adj_y = [x.detach().clone() for x in adj_y0]
        # adj_params = tuple(p.detach().clone() for p in adj_params0) if adj_params0 else ()
        y_state = list(y_finalstate)
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

    # return [x.detach().clone() for x in adj_y0], [x.detach().clone() for x in adj_params0]

        for k in range(N - 1):
            tk = t_grid[k]

            # 调用 augmented dynamics
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # 更新状态
            for i in range(len(adj_y)):
                # adj_y[i] = adj_y[i] + h * vjp_y[i]
                # y_state[i] = y_state[i] - h * func_eval[i]
                adj_y[i].add_(vjp_y[i], alpha=h)
                y_state[i].add_(func_eval[i], alpha=-h)  # 注意这里是 -h（减法）

            # 更新参数梯度
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    return adj_y, adj_params


def forward_euler_w_history(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    手写 Euler 积分（显式），支持 tuple 状态。
    - ode_func: callable(t, y_tuple) -> tuple(dy_i)
    - y0_tuple: tuple(tensor, ...) 初始状态 == 0
    - t_grid: 1D Tensor，[t0, t1, ..., tN]，步长可不等（会逐段用 dt = t_{k+1}-t_k）
    返回：tuple( Y_i )，其中 Y_i 的形状是 (N, *y0_i.shape)
    """
    assert isinstance(y0_tuple, tuple)
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    # Initialize history lists for each component

    # Clone initial values
    y_current = list(y0_tuple)
    y_history = [[] for _ in y0_tuple]

    # Euler integration: y_{k+1} = y_k + dt * f(t_k, y_k)
    for k in range(N - 1):
        tk = t_grid[k]
        dt = t_grid[k + 1] - t_grid[k]  # Scalar tensor, will broadcast automatically
        dy = ode_func(tk, tuple(y_current))  # Expect tuple return, consistent with y structure
        # assert isinstance(dy, tuple) and len(dy) == len(y)

        # Update all integrated components except the last one
        for i in range(len(y_current)):
            y_current[i] = y_current[i] + dt * dy[i]
        # Final element is the output spike (pass-through)
            y_history[i].append(y_current[i])
    return y_history


def backward_euler_w_history(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs((t_grid[-1] - t_grid[-2]))  # uniform step size
        h = float(h)

        _, adj_y0, adj_params0 = y_aug

        adj_y = [y.clone() for y in adj_y0]  # [y0.clone() for y0 in y0_tuple]
        adj_params = list(adj_params0)

        # return tuple(y_i.clone() for y_i in adj_y0), tuple(y_i.clone() for y_i in adj_params0)

        # for k in range(N - 1):
        #     tk = t_grid[k]
        #     y_state = list([y[-1-k] for y in yhistory])
        for k in range(N - 2):
            tk = t_grid[k+1]
            #t_grid_flip = t_grid.flip(0) recal that t has been flipped already
            # y_state = list([y[-k - 1] for y in yhistory])
            y_state = list([y[-k - 2] for y in yhistory])
            # 调用 augmented dynamics
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))
            # vjp_y = tuple(torch.zeros_like(y_i) for y_i in adj_y)
            # vjp_params = tuple(torch.zeros_like(y_i) for y_i in adj_params)
            # 更新状态
            # for i in range(len(adj_y)):
            #     adj_y[i] = adj_y[i] + h * vjp_y[i]
            for i in range(len(adj_y)):
                adj_y[i].add_(vjp_y[i], alpha=h)

            # 更新参数梯度
            # if adj_params and vjp_params:
            #     adj_params = tuple(
            #         ap + h * vp for ap, vp in zip(adj_params, vjp_params)
            #     )
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    del yhistory, vjp_params, func_eval, vjp_y

    return adj_y, adj_params


def forward_gl(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    assert isinstance(y0_tuple, tuple) 
    N = len(t_grid)
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[-2]) # uniform step size
    h_alpha = torch.pow(h, alpha)

    # GL coefficients: need up to c[N]
    c = torch.zeros(N + 1, dtype=dtype, device=device)
    c[0] = 1
    for j in range(1, N + 1):
        c[j] = (1 - (1 + alpha) / j) * c[j - 1]

    # Initialize with y_0 (clone to avoid modifying input)
    y_current = list(y0_tuple)  # tuple(y.clone() for y in y0_tuple)
    # History: y_history[i][j] stores y_j for component i
    # Initialize with y_0
    # y_history = [[y, ] for y in y0_tuple]
    y_history = [[] for y in y0_tuple]
    # y_history = [[y.clone()] for y in y0_tuple]

    for k in range(N - 1):
        t_k = t_grid[k]
        # Evaluate f(t_k, y_k)

        dy = ode_func(t_k, tuple(y_current))
        # assert isinstance(dy, tuple) and len(dy) == len(y_current)

        # Determine memory range
        if memory is None or memory == -1:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)

        start_idx = max(0, k + 1 - memory_length)

        # Update all integrated components
        for i in range(len(y_current)):
            # Accumulate: Σ c_{k+1-j} * y_j for j from start_idx to k
            if k > 0:
                convolution_sum = 0#torch.zeros_like(y[i])
                for j in range(start_idx, k):
                    # GL coefficient for this lag
                    convolution_sum = convolution_sum + c[k+1-j] * y_history[i][j]
                # here we assume at time k, we have k elements (without y0=0)
                # the most restrict formulation should be convolution_sum + c[k-j] * y_history[i][j]
                # which however seems do have have good numerical stability
            else:
                convolution_sum = 0

            # convolution_sum = None
            # for j in range(start_idx, k+1):
            # # # GL coefficient for this lag
            #     if convolution_sum is None:
            #         convolution_sum = c[k+1-j] * y_history[i][j]
            #     else:
            #         convolution_sum = convolution_sum + c[k+1-j] * y_history[i][j]


            # y_{k+1} = h^alpha * f_k - convolution_sum
            y_current[i] = h_alpha * dy[i] - convolution_sum
            y_history[i].append(y_current[i])

    return y_history


def backward_gl(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha = torch.pow(h, alpha)

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype

        # GL coefficients
        c = torch.zeros(N + 1, dtype=dtype, device=device)
        c[0] = 1
        for j in range(1, N + 1):
            c[j] = (1 - (1 + alpha) / j) * c[j - 1]

        # Initialize adjoint history lists for each component
        adjy_history = [[xx, ] for xx in adj_y0]

        # Clone initial adjoint values
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        for k in range(N - 2):
            tk = t_grid[k+1]
            #t_grid_flip = t_grid.flip(0) recal that t has been flipped already
            # y_state = list([y[-k - 1] for y in yhistory])
            y_state = list([y[-k - 2] for y in yhistory])

            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # Determine memory range
            if memory is None or memory == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(memory, k + 1)

            start_idx = max(0, k + 1 - memory_length)

            # Update all adjoint components
            for i in range(len(adj_y)):
                # Calculate history sum

                if True:#k > 0:
                    convolution_sum = 0  # torch.zeros_like(y[i])
                    for j in range(start_idx, k+1):
                        # GL coefficient for this lag
                        convolution_sum = convolution_sum + c[k+1-j] * adjy_history[i][j]
                    # here we assume at time k, we have k elements (without y0=0)
                    # the most restrict formulation should be convolution_sum + c[k-j] * y_history[i][j]
                    # which however seems do have have good numerical stability
                else:
                    convolution_sum = 0

                # convolution_sum = None
                # for j in range(start_idx, k+1):
                #     # # GL coefficient for this lag
                #     if convolution_sum is None:
                #         convolution_sum = c[k + 1 - j] * adjy_history[i][j]
                #     else:
                #         convolution_sum = convolution_sum + c[k + 1  - j] * adjy_history[i][j]

                # Update adjoint state
                adj_y[i] = h_alpha * vjp_y[i] - convolution_sum
                adjy_history[i].append(adj_y[i])

            # 更新参数梯度
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    del adjy_history, yhistory

    return adj_y, adj_params
def forward_trap(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Forward pass of Product Trapezoidal method for adjoint training.
    """
    assert isinstance(y0_tuple, tuple) 

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = (t_grid[-1] - t_grid[-2]) # uniform step size
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    # Initialize with y_0
    y_current = list(y0_tuple)
    y_history = [[] for y in y0_tuple]

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Determine memory range
        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute A_{j,k+1} weights for indices from start_idx to k
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

        # General formula for j >= 1:
        # A_{j,k+1} = (k+2-j)^{1-α} + (k-j)^{1-α} - 2(k+1-j)^{1-α}
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        A_j_kp1 = kjp2 + kj - 2 * kjp1

        # Special handling for j=0 if it's in the range:
        # A_{0,k+1} = k^{1-α} - (k+α)(k+1)^{-α}
        if start_idx == 0:
            k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
            kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -alpha)
            A_j_kp1[0] = k_power - (k + alpha) * kp1_neg_alpha

        # Update ALL state components (forward integrates all, not len-1)
        for i in range(len(y_current)):
            # Compute convolution sum: sum_{j=start_idx}^{k-1} A_{j,k+1} * y_j[i]
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + A_j_kp1[local_idx] * y_history[i][j]
            else:
                convolution_sum = 0

            # y_{k+1} = Γ(2-α) * h^α * f_k - convolution_sum
            y_current[i] = h_alpha_gamma * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

    return y_history


def backward_trap(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    """
    Backward pass of Product Trapezoidal method for adjoint training.
    """
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
        one_minus_alpha = 1 - alpha

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype

        # Initialize adjoint history lists for each component (with initial values)
        adjy_history = [[xx, ] for xx in adj_y0]

        # Clone initial adjoint values
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        for k in range(N - 2):
            tk = t_grid[k+1]
            # y_state = list([y[-k - 1] for y in yhistory])
            y_state = list([y[-k - 2] for y in yhistory])

            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # Determine memory range
            if memory is None:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)

            start_idx = max(0, k + 1 - memory_length)

            # Compute A_{j,k+1} weights for indices from start_idx to k
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

            # General formula for j >= 1:
            # A_{j,k+1} = (k+2-j)^{1-α} + (k-j)^{1-α} - 2(k+1-j)^{1-α}
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
            kj = torch.pow(k - j_vals, one_minus_alpha)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
            A_j_kp1 = kjp2 + kj - 2 * kjp1

            # Special handling for j=0 if it's in the range:
            # A_{0,k+1} = k^{1-α} - (k+α)(k+1)^{-α}
            if start_idx == 0:
                k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
                kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -alpha)
                A_j_kp1[0] = k_power - (k + alpha) * kp1_neg_alpha

            # Update all adjoint components
            for i in range(len(adj_y)):
                # Calculate history sum - note: range goes to k+1 (one more than forward)
                convolution_sum = 0
                for j in range(start_idx, k + 1):
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + A_j_kp1[local_idx] * adjy_history[i][j]

                # Update adjoint state
                adj_y[i] = h_alpha_gamma * vjp_y[i] - convolution_sum
                adjy_history[i].append(adj_y[i])

            # Update parameter gradients
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

    del adjy_history, yhistory

    return adj_y, adj_params


def forward_l1(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Forward pass of L1 method for adjoint training.
    """
    assert isinstance(y0_tuple, tuple) 
    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[-2]) # uniform step size
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]
    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]
        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, y_current)

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute c_j^(k) weights for indices from start_idx to k
        # General formula for j >= 1: c_j^(k) = (k-j+2)^{1-α} - 2(k-j+1)^{1-α} + (k-j)^{1-α}
        # Special case for j = 0: c_0^(k) = -[(k+1)^{1-α} - k^{1-α}]
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        c_j_k = kjp2 - 2 * kjp1 + kj

        # Special handling for j=0 if it's in the range
        if start_idx == 0:
            kp1_power = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), one_minus_alpha)
            k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
            c_j_k[0] = -(kp1_power - k_power)

        # Update ALL state components (forward integrates all, not len-1)
        for i in range(len(y_current)):
            # Compute convolution sum: sum_{j=start_idx}^{k-1} c_j^(k) * y_j[i]
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + c_j_k[local_idx] * y_history[i][j]
            else:
                convolution_sum = 0

            # y_{k+1} = h^α * Γ(2-α) * f_k - convolution_sum
            y_current[i] = h_alpha_gamma * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

    return y_history


def backward_l1(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    """
    Backward pass of L1 method for adjoint training.
    """
    with torch.no_grad():
        N = len(t_grid)
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
        one_minus_alpha = 1 - alpha

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype

        # Initialize adjoint history lists for each component (with initial values)
        adjy_history = [[xx, ] for xx in adj_y0]

        # Clone initial adjoint values
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        for k in range(N - 2):
            tk = t_grid[k+1]
            y_state = list([y[-k - 2] for y in yhistory])

            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # Determine memory range
            if memory is None:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)

            start_idx = max(0, k + 1 - memory_length)

            # Compute c_j^(k) weights for indices from start_idx to k
            # General formula for j >= 1: c_j^(k) = (k-j+2)^{1-α} - 2(k-j+1)^{1-α} + (k-j)^{1-α}
            # Special case for j = 0: c_0^(k) = -[(k+1)^{1-α} - k^{1-α}]
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
            kj = torch.pow(k - j_vals, one_minus_alpha)
            c_j_k = kjp2 - 2 * kjp1 + kj

            # Special handling for j=0 if it's in the range
            if start_idx == 0:
                kp1_power = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), one_minus_alpha)
                k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
                c_j_k[0] = -(kp1_power - k_power)

            # Update all adjoint components
            for i in range(len(adj_y)):
                # Calculate history sum - note: range goes to k+1 (one more than forward)
                convolution_sum = 0
                for j in range(start_idx, k + 1):
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + c_j_k[local_idx] * adjy_history[i][j]

                # Update adjoint state
                adj_y[i] = h_alpha_gamma * vjp_y[i] - convolution_sum
                adjy_history[i].append(adj_y[i])

            # Update parameter gradients
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

    del adjy_history, yhistory

    return adj_y, adj_params


def forward_pred(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Forward pass of Adams-Bashforth (Euler) predictor method for adjoint training.
    """
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[0]) / (N - 1)
    # gamma_alpha = 1 / math.gamma(alpha)
    h_alpha_over_alpha = torch.pow(h, alpha) / (alpha * math.gamma(alpha))

    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]
    # History for function evaluations (for ALL components in forward)
    fhistory = [[] for _ in y0_tuple]
    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]
        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, y_current)

        # Store function evaluations for ALL components
        for i in range(len(y_current)):
            fhistory[i].append(f_k[i])

        # Determine memory range
        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute weights b_{j,k+1} for indices from start_idx to k
        # b_{j,k+1} = (h^α / α) * [(k+1-j)^α - (k-j)^α]
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
        b_j_kp1 = h_alpha_over_alpha * (
                torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
        )

        # Update ALL state components
        for i in range(len(y_current)):
            # Compute convolution sum: sum_{j=start_idx}^{k} b_{j,k+1} * f_j[i]
            convolution_sum = 0
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx
                convolution_sum = convolution_sum +  b_j_kp1[local_idx] * fhistory[i][j]

            # y_{k+1} = (1/Γ(α)) * convolution_sum
            y_current[i] = convolution_sum
            y_history[i].append(y_current[i])

    del fhistory
    return y_history


def backward_pred(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    """
    Backward pass of Adams-Bashforth (Euler) predictor method for adjoint training.
    """
    with torch.no_grad():
        if not yhistory:
            raise ValueError('yhistory must not be empty for backward predictor.')
        N = t_grid.numel()
        if N < 2:
            raise ValueError('t_grid must contain at least two points.')
        expected_steps = N - 1
        if any(len(y) != expected_steps for y in yhistory):
            raise ValueError('Forward history length mismatch between forward/backward passes.')
        h = torch.abs(t_grid[-1] - t_grid[-2])
        # gamma_alpha = 1 / math.gamma(alpha)
        h_alpha_over_alpha = torch.pow(h, alpha) / (alpha * math.gamma(alpha))

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype

        # Initialize adjoint history lists with initial values
        adjf_history = [[xx, ] for xx in adj_y0]

        # Clone initial adjoint values
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        for k in range(N - 2):
            tk = t_grid[k]
            y_state = list([y[-k - 2] for y in yhistory])

            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # Store adjoint of function evaluation
            for i in range(len(adj_y)):
                adjf_history[i].append(vjp_y[i])

            # Determine memory range
            if memory is None:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)

            start_idx = max(0, k + 1 - memory_length)

            # Compute weights b_{j,k+1}
            # b_{j,k+1} = (h^α / α) * [(k+1-j)^α - (k-j)^α]
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
            b_j_kp1 = h_alpha_over_alpha * (
                    torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
            )

            # Update all adjoint components
            for i in range(len(adj_y)):
                # Compute convolution sum over adjoint history
                convolution_sum = 0
                for j in range(start_idx, k + 1):
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + b_j_kp1[local_idx] * adjf_history[i][j]

                # Update adjoint state
                adj_y[i] = adj_y0[i] +  convolution_sum

            # Update parameter gradients
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

    del adjf_history, yhistory

    return adj_y, adj_params


def forward_pred_cupy(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if not isinstance(y0_tuple, tuple) or len(y0_tuple) == 0:
        raise TypeError("y0_tuple must be a non-empty tuple for CuPy predictor.")
    N = len(t_grid)
    if N < 2:
        raise ValueError("t_grid must contain at least two elements for predictor integration.")
    T = N - 1
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    _assert_cupy_backend_ready(device, dtype)

    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[0]) / (N - 1)
    h_alpha_over_alpha = torch.pow(h, alpha) / (alpha * math.gamma(alpha))

    num_components = len(y0_tuple)
    y_current = list(y0_tuple)
    state_shapes = [state.shape for state in y0_tuple]
    state_sizes = [state.numel() for state in y0_tuple]
    history_buffers = [
        torch.empty((N - 1, state_sizes[i]), device=device, dtype=dtype)
        for i in range(num_components)
    ]
    output_buffers = [
        torch.empty((T, state_sizes[i]), device=device, dtype=dtype)
        for i in range(num_components)
    ]
    output_views = [
        output_buffers[i].view(T, *state_shapes[i])
        for i in range(num_components)
    ]

    for k in range(T):
        t_k = t_grid[k]
        f_k = ode_func(t_k, tuple(y_current))
        if not isinstance(f_k, tuple) or len(f_k) != num_components:
            raise TypeError("ODE function must return a tuple matching y0_tuple length.")

        for i in range(num_components):
            history_buffers[i][k].copy_(f_k[i].detach().reshape(-1))

        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            if memory_length <= 0:
                raise ValueError("memory must be greater than 0 when provided.")
        start_idx = max(0, k + 1 - memory_length)
        history_len = k + 1 - start_idx

        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
        coeff_slice = h_alpha_over_alpha * (
            torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
        )
        coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())

        for i in range(num_components):
            y_val = _cupy_weighted_history_sum(
                history_buffers[i],
                coeff_cp,
                start_idx,
                history_len,
                state_shapes[i],
                dtype,
                device,
            )
            output_views[i][k].copy_(y_val)
            y_current[i] = output_views[i][k]

    y_history = [list(output_views[i]) for i in range(num_components)]
    del history_buffers

    return y_history



def backward_pred_cupy(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    with torch.no_grad():
        if not yhistory:
            raise ValueError('yhistory must not be empty for backward predictor.')
        N = t_grid.numel()
        if N < 2:
            raise ValueError('t_grid must contain at least two points for backward predictor.')
        expected_steps = N - 1
        if any(len(y) != expected_steps for y in yhistory):
            raise ValueError('Forward history length mismatch between forward/backward passes.')

        h = torch.abs(t_grid[-1] - t_grid[-2])
        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype
        _assert_cupy_backend_ready(device, dtype)
        h_alpha_over_alpha = torch.pow(h, alpha) / (alpha * math.gamma(alpha))

        adj_y = list(adj_y0)
        adj_y0_clone = tuple(t.clone() for t in adj_y0)
        adj_params = list(adj_params0)
        state_shapes = [adj.shape for adj in adj_y0]
        state_sizes = [adj.numel() for adj in adj_y0]
        adj_history_buffers = [
            torch.empty((expected_steps, state_sizes[i]), device=device, dtype=dtype)
            for i in range(len(adj_y0))
        ]
        for i in range(len(adj_y0)):
            adj_history_buffers[i][0].copy_(adj_y0_clone[i].detach().reshape(-1))

        for k in range(N - 2):
            tk = t_grid[k]
            y_state = list(y[-k - 2] for y in yhistory)
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            for i in range(len(adj_y)):
                write_idx = k + 1
                adj_history_buffers[i][write_idx].copy_(vjp_y[i].detach().reshape(-1))

            if memory is None:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)
                if memory_length <= 0:
                    raise ValueError('memory must be greater than 0 when provided.')
            start_idx = max(0, k + 1 - memory_length)
            history_len = k + 1 - start_idx

            j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
            coeff_slice = h_alpha_over_alpha * (
                torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
            )
            coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())

            for i in range(len(adj_y)):
                adj_tensor = _cupy_weighted_history_sum(
                    adj_history_buffers[i],
                    coeff_cp,
                    start_idx,
                    history_len,
                    state_shapes[i],
                    dtype,
                    device,
                )
                adj_tensor.add_(adj_y0_clone[i])
                adj_y[i] = adj_tensor

            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        del adj_history_buffers, yhistory

        return adj_y, adj_params





def forward_gl_cupy(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if not isinstance(y0_tuple, tuple) or len(y0_tuple) == 0:
        raise TypeError("y0_tuple must be a non-empty tuple for CuPy GL solver.")
    N = t_grid.numel()
    if N < 2:
        raise ValueError("t_grid must contain at least two elements for GL integration.")
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    _assert_cupy_backend_ready(device, dtype)

    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[0]) / (N - 1)
    h_alpha = torch.pow(h, alpha)

    c = torch.zeros(N + 1, dtype=dtype, device=device)
    c[0] = 1
    for j in range(1, N + 1):
        c[j] = (1 - (1 + alpha) / j) * c[j - 1]

    num_components = len(y0_tuple)
    y_current = list(y0_tuple)
    state_shapes = [state.shape for state in y0_tuple]
    state_sizes = [state.numel() for state in y0_tuple]
    history_buffers = [
        torch.empty((N - 1, state_sizes[i]), device=device, dtype=dtype)
        for i in range(num_components)
    ]
    history_views = [
        history_buffers[i].view(N - 1, *state_shapes[i])
        for i in range(num_components)
    ]

    for k in range(N - 1):
        t_k = t_grid[k]
        dy = ode_func(t_k, tuple(y_current))

        if memory is None or memory == -1:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
        start_idx = max(0, k + 1 - memory_length)
        history_len = max(0, k - start_idx)

        if history_len > 0:
            j_vals = torch.arange(start_idx, k, dtype=torch.long, device=device)
            coeff_indices = (k + 1 - j_vals).to(torch.long)
            coeff_slice = c.index_select(0, coeff_indices)
            coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())
        else:
            coeff_cp = None

        for i in range(num_components):
            if history_len > 0:
                conv = _cupy_weighted_history_sum(
                    history_buffers[i],
                    coeff_cp,
                    start_idx,
                    history_len,
                    state_shapes[i],
                    dtype,
                    device,
                )
            else:
                conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)
            history_views[i][k].copy_(h_alpha * dy[i] - conv)
            y_current[i] = history_views[i][k]

    y_history = [list(history_views[i]) for i in range(num_components)]
    del history_buffers
    return y_history


def backward_gl_cupy(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    with torch.no_grad():
        if not yhistory:
            raise ValueError("yhistory must not be empty for GL adjoint.")
        N = t_grid.numel()
        if N < 2:
            raise ValueError("t_grid must contain at least two points for GL adjoint.")
        expected_steps = N - 1
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha = torch.pow(h, alpha)

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype
        _assert_cupy_backend_ready(device, dtype)

        c = torch.zeros(N + 1, dtype=dtype, device=device)
        c[0] = 1
        for j in range(1, N + 1):
            c[j] = (1 - (1 + alpha) / j) * c[j - 1]

        adj_y = list(adj_y0)
        adj_params = list(adj_params0)
        state_shapes = [adj.shape for adj in adj_y0]
        state_sizes = [adj.numel() for adj in adj_y0]
        history_buffers = [
            torch.empty((expected_steps, state_sizes[i]), device=device, dtype=dtype)
            for i in range(len(adj_y0))
        ]
        for i in range(len(adj_y0)):
            history_buffers[i][0].copy_(adj_y0[i].detach().reshape(-1))

        for k in range(N - 2):
            tk = t_grid[k]
            y_state = list(y[-k - 2] for y in yhistory)
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            if memory is None or memory == -1:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)
            start_idx = max(0, k + 1 - memory_length)
            history_len = max(0, k + 1 - start_idx)

            if history_len > 0:
                j_vals = torch.arange(start_idx, k + 1, dtype=torch.long, device=device)
                coeff_indices = (k + 1 - j_vals).to(torch.long)
                coeff_slice = c.index_select(0, coeff_indices)
                coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())
            else:
                coeff_cp = None

            for i in range(len(adj_y)):
                if history_len > 0:
                    conv = _cupy_weighted_history_sum(
                        history_buffers[i],
                        coeff_cp,
                        start_idx,
                        history_len,
                        state_shapes[i],
                        dtype,
                        device,
                    )
                else:
                    conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

                adj_tensor = h_alpha * vjp_y[i] - conv
                history_buffers[i][k + 1].copy_(adj_tensor.detach().reshape(-1))
                adj_y[i] = adj_tensor

            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        del history_buffers, yhistory
        return adj_y, adj_params


def backward_glmulti_cupy(ode_func, y_aug, alpha, t_grid, yhistory, memory=None, multi_coefficient=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    with torch.no_grad():
        if not yhistory:
            raise ValueError("yhistory must not be empty for GL multi-term adjoint.")
        if multi_coefficient is None:
            raise ValueError("multi_coefficient must be provided for GL multi-term adjoint.")
        N = t_grid.numel()
        if N < 2:
            raise ValueError("t_grid must contain at least two points for GL multi-term adjoint.")
        expected_steps = N - 1
        h = torch.abs(t_grid[-1] - t_grid[-2])

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype
        _assert_cupy_backend_ready(device, dtype)

        if not isinstance(alpha, torch.Tensor):
            beta = torch.tensor(alpha, dtype=dtype, device=device)
        else:
            beta = alpha.to(device=device, dtype=dtype)

        if not isinstance(multi_coefficient, torch.Tensor):
            coefficient = torch.tensor(multi_coefficient, dtype=dtype, device=device)
        else:
            coefficient = multi_coefficient.to(device=device, dtype=dtype)

        n_terms = len(beta)
        if len(coefficient) != n_terms:
            raise ValueError("beta and multi_coefficient must have the same length.")

        c = torch.zeros(n_terms, N + 1, dtype=dtype, device=device)
        c[:, 0] = 1.0
        for m in range(1, N + 1):
            c[:, m] = (1 - (1 + beta) / m) * c[:, m - 1]

        h_neg_power = torch.pow(h, -beta)
        weighted_h = coefficient * h_neg_power
        c_tilde = torch.sum(weighted_h.unsqueeze(1) * c, dim=0)
        c_tilde_0_inv = 1.0 / c_tilde[0]

        adj_y = list(adj_y0)
        adj_params = list(adj_params0)
        state_shapes = [adj.shape for adj in adj_y0]
        state_sizes = [adj.numel() for adj in adj_y0]
        history_buffers = [
            torch.empty((expected_steps, state_sizes[i]), device=device, dtype=dtype)
            for i in range(len(adj_y0))
        ]
        for i in range(len(adj_y0)):
            history_buffers[i][0].copy_(adj_y0[i].detach().reshape(-1))

        for k in range(N - 2):
            tk = t_grid[k]
            y_state = list(y[-k - 2] for y in yhistory)
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            if memory is None or memory == -1:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)
            start_idx = max(0, k + 1 - memory_length)
            history_len = max(0, k + 1 - start_idx)

            if history_len > 0:
                j_vals = torch.arange(start_idx, k + 1, dtype=torch.long, device=device)
                coeff_indices = (k + 1 - j_vals).to(torch.long)
                coeff_slice = c_tilde.index_select(0, coeff_indices)
                coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())
            else:
                coeff_cp = None

            for i in range(len(adj_y)):
                if history_len > 0:
                    conv = _cupy_weighted_history_sum(
                        history_buffers[i],
                        coeff_cp,
                        start_idx,
                        history_len,
                        state_shapes[i],
                        dtype,
                        device,
                    )
                else:
                    conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

                adj_tensor = c_tilde_0_inv * (vjp_y[i] - conv)
                history_buffers[i][k + 1].copy_(adj_tensor.detach().reshape(-1))
                adj_y[i] = adj_tensor

            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        del history_buffers, yhistory
        return adj_y, adj_params


def forward_trap_cupy(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if not isinstance(y0_tuple, tuple) or len(y0_tuple) == 0:
        raise TypeError("y0_tuple must be a non-empty tuple for CuPy trap solver.")
    N = len(t_grid)
    if N < 2:
        raise ValueError("t_grid must contain at least two elements for trap integration.")
    T = N - 1
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    _assert_cupy_backend_ready(device, dtype)

    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[0]) / (N - 1)
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    num_components = len(y0_tuple)
    y_current = list(y0_tuple)
    state_shapes = [state.shape for state in y0_tuple]
    state_sizes = [state.numel() for state in y0_tuple]
    history_buffers = [
        torch.empty((T, state_sizes[i]), device=device, dtype=dtype)
        for i in range(num_components)
    ]
    history_views = [
        history_buffers[i].view(T, *state_shapes[i])
        for i in range(num_components)
    ]

    for k in range(T):
        t_k = t_grid[k]
        f_k = ode_func(t_k, tuple(y_current))
        if not isinstance(f_k, tuple) or len(f_k) != num_components:
            raise TypeError("ODE function must return a tuple matching y0_tuple length.")

        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            if memory_length <= 0:
                raise ValueError("memory must be greater than 0 when provided.")
        start_idx = max(0, k + 1 - memory_length)
        history_len = max(0, k - start_idx)

        if history_len > 0:
            j_vals = torch.arange(start_idx, k, dtype=dtype, device=device)
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
            kj = torch.pow(k - j_vals, one_minus_alpha)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
            coeff_slice = kjp2 + kj - 2 * kjp1

            if start_idx == 0:
                k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
                kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -alpha)
                coeff_slice[0] = k_power - (k + alpha) * kp1_neg_alpha

            coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())
        else:
            coeff_cp = None

        for i in range(num_components):
            if history_len > 0:
                conv = _cupy_weighted_history_sum(
                    history_buffers[i],
                    coeff_cp,
                    start_idx,
                    history_len,
                    state_shapes[i],
                    dtype,
                    device,
                )
            else:
                conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

            history_views[i][k].copy_(h_alpha_gamma * f_k[i] - conv)
            y_current[i] = history_views[i][k]

    y_history = [list(history_views[i]) for i in range(num_components)]
    del history_buffers
    return y_history


def backward_trap_cupy(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    with torch.no_grad():
        if not yhistory:
            raise ValueError("yhistory must not be empty for trap adjoint.")
        N = t_grid.numel()
        if N < 2:
            raise ValueError("t_grid must contain at least two points for trap adjoint.")
        expected_steps = N - 1
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
        one_minus_alpha = 1 - alpha

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype
        _assert_cupy_backend_ready(device, dtype)

        adj_y = list(adj_y0)
        adj_params = list(adj_params0)
        state_shapes = [adj.shape for adj in adj_y0]
        state_sizes = [adj.numel() for adj in adj_y0]
        history_buffers = [
            torch.empty((expected_steps, state_sizes[i]), device=device, dtype=dtype)
            for i in range(len(adj_y0))
        ]
        for i in range(len(adj_y0)):
            history_buffers[i][0].copy_(adj_y0[i].detach().reshape(-1))

        for k in range(N - 2):
            tk = t_grid[k]
            y_state = list(y[-k - 2] for y in yhistory)
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            if memory is None:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)
            start_idx = max(0, k + 1 - memory_length)
            history_len = max(0, k + 1 - start_idx)

            if history_len > 0:
                j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
                kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
                kj = torch.pow(k - j_vals, one_minus_alpha)
                kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
                coeff_slice = kjp2 + kj - 2 * kjp1

                if start_idx == 0:
                    k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
                    kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -alpha)
                    coeff_slice[0] = k_power - (k + alpha) * kp1_neg_alpha

                coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())
            else:
                coeff_cp = None

            for i in range(len(adj_y)):
                if history_len > 0:
                    conv = _cupy_weighted_history_sum(
                        history_buffers[i],
                        coeff_cp,
                        start_idx,
                        history_len,
                        state_shapes[i],
                        dtype,
                        device,
                    )
                else:
                    conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

                adj_tensor = h_alpha_gamma * vjp_y[i] - conv
                history_buffers[i][k + 1].copy_(adj_tensor.detach().reshape(-1))
                adj_y[i] = adj_tensor

            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        del history_buffers, yhistory
        return adj_y, adj_params


def backward_glmulti_cupy(ode_func, y_aug, alpha, t_grid, yhistory, memory=None, multi_coefficient=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    with torch.no_grad():
        if not yhistory:
            raise ValueError("yhistory must not be empty for GL multi-term adjoint.")
        if multi_coefficient is None:
            raise ValueError("multi_coefficient must be provided for GL multi-term adjoint.")
        N = t_grid.numel()
        if N < 2:
            raise ValueError("t_grid must contain at least two points for GL multi-term adjoint.")
        expected_steps = N - 1
        h = torch.abs(t_grid[-1] - t_grid[-2])

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype
        _assert_cupy_backend_ready(device, dtype)

        if not isinstance(alpha, torch.Tensor):
            beta = torch.tensor(alpha, dtype=dtype, device=device)
        else:
            beta = alpha.to(device=device, dtype=dtype)

        if not isinstance(multi_coefficient, torch.Tensor):
            coefficient = torch.tensor(multi_coefficient, dtype=dtype, device=device)
        else:
            coefficient = multi_coefficient.to(device=device, dtype=dtype)

        n_terms = len(beta)
        if len(coefficient) != n_terms:
            raise ValueError("beta and multi_coefficient must have the same length.")

        c = torch.zeros(n_terms, N + 1, dtype=dtype, device=device)
        c[:, 0] = 1.0
        for m in range(1, N + 1):
            c[:, m] = (1 - (1 + beta) / m) * c[:, m - 1]

        h_neg_power = torch.pow(h, -beta)
        weighted_h = coefficient * h_neg_power
        c_tilde = torch.sum(weighted_h.unsqueeze(1) * c, dim=0)
        c_tilde_0_inv = 1.0 / c_tilde[0]

        adj_y = list(adj_y0)
        adj_params = list(adj_params0)
        state_shapes = [adj.shape for adj in adj_y0]
        state_sizes = [adj.numel() for adj in adj_y0]
        history_buffers = [
            torch.empty((expected_steps, state_sizes[i]), device=device, dtype=dtype)
            for i in range(len(adj_y0))
        ]
        for i in range(len(adj_y0)):
            history_buffers[i][0].copy_(adj_y0[i].detach().reshape(-1))

        for k in range(N - 2):
            tk = t_grid[k]
            y_state = list(y[-k - 2] for y in yhistory)
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            if memory is None or memory == -1:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)
            start_idx = max(0, k + 1 - memory_length)
            history_len = max(0, k + 1 - start_idx)

            if history_len > 0:
                j_vals = torch.arange(start_idx, k + 1, dtype=torch.long, device=device)
                coeff_indices = (k + 1 - j_vals).to(torch.long)
                coeff_slice = c_tilde.index_select(0, coeff_indices)
                coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())
            else:
                coeff_cp = None

            for i in range(len(adj_y)):
                if history_len > 0:
                    conv = _cupy_weighted_history_sum(
                        history_buffers[i],
                        coeff_cp,
                        start_idx,
                        history_len,
                        state_shapes[i],
                        dtype,
                        device,
                    )
                else:
                    conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

                adj_tensor = c_tilde_0_inv * (vjp_y[i] - conv)
                history_buffers[i][k + 1].copy_(adj_tensor.detach().reshape(-1))
                adj_y[i] = adj_tensor

            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        del history_buffers, yhistory
        return adj_y, adj_params


def forward_l1_cupy(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if not isinstance(y0_tuple, tuple) or len(y0_tuple) == 0:
        raise TypeError("y0_tuple must be a non-empty tuple for CuPy L1 solver.")
    N = len(t_grid)
    if N < 2:
        raise ValueError("t_grid must contain at least two elements for L1 integration.")
    T = N - 1
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    _assert_cupy_backend_ready(device, dtype)

    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[0]) / (N - 1)
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    num_components = len(y0_tuple)
    y_current = list(y0_tuple)
    state_shapes = [state.shape for state in y0_tuple]
    state_sizes = [state.numel() for state in y0_tuple]
    history_buffers = [
        torch.empty((T, state_sizes[i]), device=device, dtype=dtype)
        for i in range(num_components)
    ]
    history_views = [
        history_buffers[i].view(T, *state_shapes[i])
        for i in range(num_components)
    ]

    for k in range(T):
        t_k = t_grid[k]
        f_k = ode_func(t_k, y_current)
        if not isinstance(f_k, tuple) or len(f_k) != num_components:
            raise TypeError("ODE function must return a tuple matching y0_tuple length.")

        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            if memory_length <= 0:
                raise ValueError("memory must be greater than 0 when provided.")
        start_idx = max(0, k + 1 - memory_length)
        history_len = max(0, k - start_idx)

        if history_len > 0:
            j_vals = torch.arange(start_idx, k, dtype=dtype, device=device)
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
            kj = torch.pow(k - j_vals, one_minus_alpha)
            coeff_slice = kjp2 - 2 * kjp1 + kj

            if start_idx == 0:
                kp1_power = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), one_minus_alpha)
                k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
                coeff_slice[0] = -(kp1_power - k_power)

            coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())
        else:
            coeff_cp = None

        for i in range(num_components):
            if history_len > 0:
                conv = _cupy_weighted_history_sum(
                    history_buffers[i],
                    coeff_cp,
                    start_idx,
                    history_len,
                    state_shapes[i],
                    dtype,
                    device,
                )
            else:
                conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

            history_views[i][k].copy_(h_alpha_gamma * f_k[i] - conv)
            y_current[i] = history_views[i][k]

    y_history = [list(history_views[i]) for i in range(num_components)]
    del history_buffers
    return y_history


def backward_l1_cupy(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    with torch.no_grad():
        if not yhistory:
            raise ValueError("yhistory must not be empty for L1 adjoint.")
        N = len(t_grid)
        if N < 2:
            raise ValueError("t_grid must contain at least two points for L1 adjoint.")
        expected_steps = N - 1
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
        one_minus_alpha = 1 - alpha

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype
        _assert_cupy_backend_ready(device, dtype)

        adj_y = list(adj_y0)
        adj_params = list(adj_params0)
        state_shapes = [adj.shape for adj in adj_y0]
        state_sizes = [adj.numel() for adj in adj_y0]
        history_buffers = [
            torch.empty((expected_steps, state_sizes[i]), device=device, dtype=dtype)
            for i in range(len(adj_y0))
        ]
        for i in range(len(adj_y0)):
            history_buffers[i][0].copy_(adj_y0[i].detach().reshape(-1))

        for k in range(N - 2):
            tk = t_grid[k + 1]
            y_state = list(y[-k - 2] for y in yhistory)
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            if memory is None:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)
            start_idx = max(0, k + 1 - memory_length)
            history_len = max(0, k + 1 - start_idx)

            if history_len > 0:
                j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
                kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
                kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
                kj = torch.pow(k - j_vals, one_minus_alpha)
                coeff_slice = kjp2 - 2 * kjp1 + kj

                if start_idx == 0:
                    kp1_power = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), one_minus_alpha)
                    k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
                    coeff_slice[0] = -(kp1_power - k_power)

                coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())
            else:
                coeff_cp = None

            for i in range(len(adj_y)):
                if history_len > 0:
                    conv = _cupy_weighted_history_sum(
                        history_buffers[i],
                        coeff_cp,
                        start_idx,
                        history_len,
                        state_shapes[i],
                        dtype,
                        device,
                    )
                else:
                    conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

                adj_tensor = h_alpha_gamma * vjp_y[i] - conv
                history_buffers[i][k + 1].copy_(adj_tensor.detach().reshape(-1))
                adj_y[i] = adj_tensor

            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        del history_buffers, yhistory
        return adj_y, adj_params


def backward_glmulti_cupy(ode_func, y_aug, alpha, t_grid, yhistory, memory=None, multi_coefficient=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    with torch.no_grad():
        if not yhistory:
            raise ValueError("yhistory must not be empty for GL multi-term adjoint.")
        if multi_coefficient is None:
            raise ValueError("multi_coefficient must be provided for GL multi-term adjoint.")
        N = t_grid.numel()
        if N < 2:
            raise ValueError("t_grid must contain at least two points for GL multi-term adjoint.")
        expected_steps = N - 1
        h = torch.abs(t_grid[-1] - t_grid[-2])

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype
        _assert_cupy_backend_ready(device, dtype)

        if not isinstance(alpha, torch.Tensor):
            beta = torch.tensor(alpha, dtype=dtype, device=device)
        else:
            beta = alpha.to(device=device, dtype=dtype)

        if not isinstance(multi_coefficient, torch.Tensor):
            coefficient = torch.tensor(multi_coefficient, dtype=dtype, device=device)
        else:
            coefficient = multi_coefficient.to(device=device, dtype=dtype)

        n_terms = len(beta)
        if len(coefficient) != n_terms:
            raise ValueError("beta and multi_coefficient must have the same length.")

        c = torch.zeros(n_terms, N + 1, dtype=dtype, device=device)
        c[:, 0] = 1.0
        for m in range(1, N + 1):
            c[:, m] = (1 - (1 + beta) / m) * c[:, m - 1]

        h_neg_power = torch.pow(h, -beta)
        weighted_h = coefficient * h_neg_power
        c_tilde = torch.sum(weighted_h.unsqueeze(1) * c, dim=0)
        c_tilde_0_inv = 1.0 / c_tilde[0]

        adj_y = list(adj_y0)
        adj_params = list(adj_params0)
        state_shapes = [adj.shape for adj in adj_y0]
        state_sizes = [adj.numel() for adj in adj_y0]
        history_buffers = [
            torch.empty((expected_steps, state_sizes[i]), device=device, dtype=dtype)
            for i in range(len(adj_y0))
        ]
        for i in range(len(adj_y0)):
            history_buffers[i][0].copy_(adj_y0[i].detach().reshape(-1))

        for k in range(N - 2):
            tk = t_grid[k]
            y_state = list(y[-k - 2] for y in yhistory)
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            if memory is None or memory == -1:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)
            start_idx = max(0, k + 1 - memory_length)
            history_len = max(0, k + 1 - start_idx)

            if history_len > 0:
                j_vals = torch.arange(start_idx, k + 1, dtype=torch.long, device=device)
                coeff_indices = (k + 1 - j_vals).to(torch.long)
                coeff_slice = c_tilde.index_select(0, coeff_indices)
                coeff_cp = cupy.asarray(coeff_slice.detach().contiguous())
            else:
                coeff_cp = None

            for i in range(len(adj_y)):
                if history_len > 0:
                    conv = _cupy_weighted_history_sum(
                        history_buffers[i],
                        coeff_cp,
                        start_idx,
                        history_len,
                        state_shapes[i],
                        dtype,
                        device,
                    )
                else:
                    conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

                adj_tensor = c_tilde_0_inv * (vjp_y[i] - conv)
                history_buffers[i][k + 1].copy_(adj_tensor.detach().reshape(-1))
                adj_y[i] = adj_tensor

            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        del history_buffers, yhistory
        return adj_y, adj_params


def gl_integrate_tuple_cupy(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if not isinstance(y0_tuple, tuple) or len(y0_tuple) == 0:
        raise TypeError("y0_tuple must be a non-empty tuple for CuPy GL integrator.")
    N = len(t_grid)
    if N < 2:
        raise ValueError("t_grid must contain at least two elements for GL integration.")
    T = N - 1
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    _assert_cupy_backend_ready(device, dtype)

    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[-2])
    h_alpha = torch.pow(h, alpha)

    c = torch.zeros(N + 1, dtype=dtype, device=device)
    c[0] = 1
    for j in range(1, N + 1):
        c[j] = (1 - (1 + alpha) / j) * c[j - 1]

    n_integrate = max(0, len(y0_tuple) - 1)
    y_history = [[] for _ in y0_tuple]
    y_current = list(y0_tuple)
    state_shapes = [state.shape for state in y0_tuple[:n_integrate]]

    for k in range(T):
        t_k = t_grid[k]
        dy = ode_func(t_k, tuple(y_current))
        if not isinstance(dy, tuple) or len(dy) != len(y0_tuple):
            raise TypeError("ODE function must return a tuple matching y0_tuple length.")

        if memory is None or memory == -1:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            if memory_length <= 0:
                raise ValueError("memory must be greater than 0 when provided.")
        start_idx = max(0, k + 1 - memory_length)
        history_len = max(0, k - start_idx)

        if history_len > 0:
            j_vals = torch.arange(start_idx, k, dtype=torch.long, device=device)
            coeff_indices = (k + 1 - j_vals).to(torch.long)
            coeff_slice = c.index_select(0, coeff_indices)

        for i in range(n_integrate):
            if history_len > 0:
                conv = 0
                for j in range(start_idx, k):
                    local_idx = j - start_idx
                    conv = conv + coeff_slice[local_idx] * y_history[i][j]
            else:
                conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

            y_val = h_alpha * dy[i] - conv
            y_current[i] = y_val
            y_history[i].append(y_val)

        y_current[-1] = dy[-1]
        y_history[-1].append(y_current[-1])

    return y_history


def trap_integrate_tuple_cupy(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if not isinstance(y0_tuple, tuple) or len(y0_tuple) == 0:
        raise TypeError("y0_tuple must be a non-empty tuple for CuPy trap integrator.")
    N = len(t_grid)
    if N < 2:
        raise ValueError("t_grid must contain at least two elements for trap integration.")
    T = N - 1
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    _assert_cupy_backend_ready(device, dtype)

    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[-2])
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    n_integrate = max(0, len(y0_tuple) - 1)
    y_history = [[] for _ in y0_tuple]
    y_current = list(y0_tuple)
    state_shapes = [state.shape for state in y0_tuple[:n_integrate]]

    for k in range(T):
        t_k = t_grid[k]
        f_k = ode_func(t_k, tuple(y_current))
        if not isinstance(f_k, tuple) or len(f_k) != len(y0_tuple):
            raise TypeError("ODE function must return a tuple matching y0_tuple length.")

        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            if memory_length <= 0:
                raise ValueError("memory must be greater than 0 when provided.")
        start_idx = max(0, k + 1 - memory_length)
        history_len = max(0, k - start_idx)

        if history_len > 0:
            j_vals = torch.arange(start_idx, k, dtype=dtype, device=device)
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
            kj = torch.pow(k - j_vals, one_minus_alpha)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
            coeff_slice = kjp2 + kj - 2 * kjp1

            if start_idx == 0:
                k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
                kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -alpha)
                coeff_slice[0] = k_power - (k + alpha) * kp1_neg_alpha

        for i in range(n_integrate):
            if history_len > 0:
                conv = 0
                for j in range(start_idx, k):
                    local_idx = j - start_idx
                    conv = conv + coeff_slice[local_idx] * y_history[i][j]
            else:
                conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

            y_val = h_alpha_gamma * f_k[i] - conv
            y_current[i] = y_val
            y_history[i].append(y_val)

        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    return y_history


def l1_integrate_tuple_cupy(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if not isinstance(y0_tuple, tuple) or len(y0_tuple) == 0:
        raise TypeError("y0_tuple must be a non-empty tuple for CuPy L1 integrator.")
    N = len(t_grid)
    if N < 2:
        raise ValueError("t_grid must contain at least two elements for L1 integration.")
    T = N - 1
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    _assert_cupy_backend_ready(device, dtype)

    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[-2])
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    n_integrate = max(0, len(y0_tuple) - 1)
    y_history = [[] for _ in y0_tuple]
    y_current = list(y0_tuple)
    state_shapes = [state.shape for state in y0_tuple[:n_integrate]]

    for k in range(T):
        t_k = t_grid[k]
        f_k = ode_func(t_k, tuple(y_current))
        if not isinstance(f_k, tuple) or len(f_k) != len(y0_tuple):
            raise TypeError("ODE function must return a tuple matching y0_tuple length.")

        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            if memory_length <= 0:
                raise ValueError("memory must be greater than 0 when provided.")
        start_idx = max(0, k + 1 - memory_length)
        history_len = max(0, k - start_idx)

        if history_len > 0:
            j_vals = torch.arange(start_idx, k, dtype=dtype, device=device)
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
            kj = torch.pow(k - j_vals, one_minus_alpha)
            coeff_slice = kjp2 - 2 * kjp1 + kj

            if start_idx == 0:
                kp1_power = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), one_minus_alpha)
                k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
                coeff_slice[0] = -(kp1_power - k_power)

        for i in range(n_integrate):
            if history_len > 0:
                history_tensor = torch.stack(y_history[i][start_idx:k], dim=0)
                conv = _CupyWeightedHistorySum.apply(coeff_slice, history_tensor)
            else:
                conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

            y_val = h_alpha_gamma * f_k[i] - conv
            y_current[i] = y_val
            y_history[i].append(y_val)

        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    return y_history


def pred_integrate_tuple_cupy(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if not isinstance(y0_tuple, tuple) or len(y0_tuple) == 0:
        raise TypeError("y0_tuple must be a non-empty tuple for CuPy predictor integrator.")
    N = len(t_grid)
    if N < 2:
        raise ValueError("t_grid must contain at least two elements for predictor integration.")
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    _assert_cupy_backend_ready(device, dtype)

    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[-2])
    h_alpha_over_alpha = torch.pow(h, alpha) / (alpha * math.gamma(alpha))

    n_integrate = len(y0_tuple) - 1
    y_history = [[] for _ in y0_tuple]
    fhistory = [[] for _ in range(n_integrate)]
    y_current = list(y0_tuple)

    for k in range(N - 1):
        t_k = t_grid[k]
        f_k = ode_func(t_k, tuple(y_current))
        if not isinstance(f_k, tuple) or len(f_k) != len(y0_tuple):
            raise TypeError("ODE function must return a tuple matching y0_tuple length.")

        for i in range(n_integrate):
            fhistory[i].append(f_k[i])

        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            if memory_length <= 0:
                raise ValueError("memory must be greater than 0 when provided.")
        start_idx = max(0, k + 1 - memory_length)
        history_len = k + 1 - start_idx

        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
        coeff_slice = h_alpha_over_alpha * (
            torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
        )

        for i in range(n_integrate):
            if history_len > 0:
                convolution_sum = 0
                for j in range(start_idx, k + 1):
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + coeff_slice[local_idx] * fhistory[i][j]
                y_val = convolution_sum
            else:
                y_val = torch.zeros_like(y_current[i])
            y_current[i] = y_val
            y_history[i].append(y_val)

        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    del fhistory
    return y_history


def forward_glmulti_cupy(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None,
                         multi_coefficient=None, history_chunk_size=None,
                         detach_conv=True):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    if not isinstance(y0_tuple, tuple) or len(y0_tuple) == 0:
        raise TypeError("y0_tuple must be a non-empty tuple for CuPy multi-term GL solver.")
    if multi_coefficient is None:
        raise ValueError("multi_coefficient must be provided for CuPy multi-term GL solver.")
    N = len(t_grid)
    if N < 2:
        raise ValueError("t_grid must contain at least two elements for multi-term GL integration.")
    T = N - 1
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    _assert_cupy_backend_ready(device, dtype)

    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[-2])

    if not isinstance(alpha, torch.Tensor):
        beta = torch.tensor(alpha, dtype=dtype, device=device)
    else:
        beta = alpha.to(device=device, dtype=dtype)

    if not isinstance(multi_coefficient, torch.Tensor):
        coefficient = torch.tensor(multi_coefficient, dtype=dtype, device=device)
    else:
        coefficient = multi_coefficient.to(device=device, dtype=dtype)

    n_terms = len(beta)
    if len(coefficient) != n_terms:
        raise ValueError("beta and multi_coefficient must have the same length.")

    c = torch.zeros(n_terms, N + 1, dtype=dtype, device=device)
    c[:, 0] = 1.0
    for m in range(1, N + 1):
        c[:, m] = (1 - (1 + beta) / m) * c[:, m - 1]

    h_neg_power = torch.pow(h, -beta)
    weighted_h = coefficient * h_neg_power
    c_tilde = torch.sum(weighted_h.unsqueeze(1) * c, dim=0)
    c_tilde_0_inv = 1.0 / c_tilde[0]

    n_integrate = max(0, len(y0_tuple) - 1)
    y_history = [[] for _ in y0_tuple]
    y_current = list(y0_tuple)

    state_shapes = [state.shape for state in y0_tuple[:n_integrate]]
    if history_chunk_size is None:
        history_chunk_size = 96
    for k in range(T):
        t_k = t_grid[k]
        f_k = ode_func(t_k, tuple(y_current))
        if not isinstance(f_k, tuple) or len(f_k) != len(y0_tuple):
            raise TypeError("ODE function must return a tuple matching y0_tuple length.")

        if memory is None or memory == -1:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            if memory_length <= 0:
                raise ValueError("memory must be greater than 0 when provided.")

        start_idx = max(0, k + 1 - memory_length)
        history_len = max(0, k - start_idx)

        if history_len > 0:
            j_vals = torch.arange(start_idx, k, dtype=torch.long, device=device)
            coeff_indices = (k + 1 - j_vals).to(torch.long)
            coeff_slice = c_tilde.index_select(0, coeff_indices)

        for i in range(n_integrate):
            if history_len > 0:
                if history_chunk_size <= 0:
                    conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)
                    for offset, idx in enumerate(range(start_idx, k)):
                        coeff_term = coeff_slice[offset]
                        hist_term = y_history[i][idx]
                        if detach_conv:
                            coeff_term = coeff_term.detach()
                            hist_term = hist_term.detach()
                        conv = conv + coeff_term * hist_term
                else:
                    conv = None
                    offset = 0
                    chunk_start = start_idx
                    while chunk_start < k:
                        chunk_end = min(chunk_start + history_chunk_size, k)
                        history_tensor = torch.stack(y_history[i][chunk_start:chunk_end], dim=0)
                        coeff_chunk = coeff_slice[offset:offset + (chunk_end - chunk_start)]
                        if detach_conv:
                            history_tensor = history_tensor.detach()
                            coeff_chunk = coeff_chunk.detach()
                        chunk_conv = _CupyWeightedHistorySum.apply(coeff_chunk, history_tensor)
                        conv = chunk_conv if conv is None else conv + chunk_conv
                        offset += (chunk_end - chunk_start)
                        chunk_start = chunk_end
            else:
                conv = torch.zeros(state_shapes[i], device=device, dtype=dtype)

            y_val = c_tilde_0_inv * (f_k[i] - conv)
            y_current[i] = y_val
            y_history[i].append(y_val)

        if len(y0_tuple) > 0:
            y_current[-1] = f_k[-1]
            y_history[-1].append(y_current[-1])

    return y_history


@torch.compile(mode="reduce-overhead")
def l1_integrate_tuple(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Integrate a Caputo fractional differential equation using the L1 method.

    Solves equations of the form: D^α y(t) = f(t, y)

    Reference:
        A compact finite difference scheme for the fractional sub-diffusion equations,
        Journal of Computational Physics, vol. 230, no. 3, pp. 586–595, 2011.

    Args:
        ode_func: Callable (t, y_tuple) -> tuple.
                  Right-hand side function returning derivatives with the same structure as y_tuple.
        y0_tuple: Tuple of initial state tensors for each component.
        alpha: Fractional derivative order, must be in the range (0, 1).
        t_grid: 1D tensor of uniformly spaced time points. t_grid[0] is the initial time.
        memory: Number of past steps to retain for the convolution sum.
                If None or -1, the full history is used (short-memory principle disabled).

    Returns:
        list of lists: y_history where y_history[i][k] is the state of component i at time step k+1.
                       - Length of each y_history[i] is len(t_grid) - 1 (excludes initial condition).
                       - y_history[:-1] contains membrane potential histories for each layer.
                       - y_history[-1] contains spike output history (final SNN layer, not integrated).
    """
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = (t_grid[-1] - t_grid[0]) / (N - 1)
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha


    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]

    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute c_j^(k) weights for indices from start_idx to k
        # General formula for j >= 1: c_j^(k) = (k-j+2)^{1-α} - 2(k-j+1)^{1-α} + (k-j)^{1-α}
        # Special case for j = 0: c_0^(k) = -[(k+1)^{1-α} - k^{1-α}]
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        c_j_k = kjp2 - 2 * kjp1 + kj

        # Special handling for j=0 if it's in the range
        if start_idx == 0:
            kp1_power = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), one_minus_alpha)
            k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
            c_j_k[0] = -(kp1_power - k_power)

        # Update integrated states
        for i in range(len(y0_tuple) - 1):
            # Compute convolution sum: sum_{j=start_idx}^{k} c_j^(k) * y_j[i]
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):  # range(start_idx, k) because we have k elements (without y0=0)
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + c_j_k[local_idx] * y_history[i][j]
                    # the most restrict formulation should be local_idx = j - start_idx + 1
            else:
                convolution_sum = 0

            # y_{k+1} = h^α * Γ(2-α) * f_k - convolution_sum
            y_current[i] = h_alpha_gamma * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

        # Pass-through output (spike) - last component is not integrated
        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    return y_history


@torch.compile(mode="reduce-overhead")
def pred_integrate_tuple1(ode_func, y0_tuple, beta, t_grid: torch.Tensor, memory=None):
    """
    From the paper "Detailed error analysis for a fractional
        Adams method."  Numerical algorithms 36.1 (2004): 31-52.

    Args:
        func: callable(t, y_tuple) 返回与y_tuple相同结构的元组
        y0_tuple: 初始状态的元组
        beta: 分数阶指数 (0, 1)
        t_grid: 时间点序列
        memory: 历史窗口大小（None表示使用全部历史）

    Returns:
        tuple: 所有时间点的状态输出
    """
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    N = len(t_grid) - 1
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype

    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[0]) / (N - 1)

    gamma_beta = 1 / math.gamma(beta)
    h_beta_over_beta = torch.pow(h, beta) / beta

    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]

    # Determine which components need integration
    n_integrate = len(y0_tuple) - 1

    # History for function evaluations (only for integrated components)
    fhistory = [[] for i in range(n_integrate)]

    # Clone initial values
    y = list(y0_tuple)

    for k in range(N):
        tn = t_grid[k]

        # Calculate derivatives
        f_k = ode_func(tn, tuple(y))

        # Store history for integrated components
        for i in range(n_integrate):
            fhistory[i].append(f_k[i])

        memory_k = 0 if memory is None else max(0, k - memory)

        # Calculate coefficients
        j_vals = torch.arange(memory_k, k + 1, dtype=dtype, device=device)
        b_j = gamma_beta * h_beta_over_beta * (
                torch.pow(k + 1 - j_vals, beta) - torch.pow(k - j_vals, beta)
        )

        # Update integrated states
        for i in range(n_integrate):
            # Replace einsum with for loop
            weighted_sum = 0#torch.zeros_like(y[i])
            for j_idx, j in enumerate(range(memory_k, k + 1)):
                weighted_sum = weighted_sum + b_j[j_idx] * fhistory[i][j]

            y[i] = weighted_sum  # because y0=0
            y_history[i].append(y[i])

        # Pass-through output (spike) - last component
        y[-1] = f_k[-1]
        y_history[-1].append(y[-1])

    return y_history


@torch.compile(mode="reduce-overhead")
def pred_integrate_tuple(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Integrate a Caputo fractional differential equation using
    one-step Adams-Bashforth (Euler) predictor method,

    Solves equations of the form: D^α y(t) = f(t, y)

    Reference:
        Detailed error analysis for a fractional Adams method.
        Numerical algorithms 36.1 (2004): 31-52.

    Args:
        ode_func: Callable (t, y_tuple) -> tuple.
                  Right-hand side function returning derivatives with the same structure as y_tuple.
        y0_tuple: Tuple of initial state tensors for each component.
        alpha: Fractional derivative order, must be in the range (0, 1).
        t_grid: 1D tensor of uniformly spaced time points. t_grid[0] is the initial time.
        memory: Number of past steps to retain for the convolution sum.
                If None or -1, the full history is used (short-memory principle disabled).

    Returns:
        list of lists: y_history where y_history[i][k] is the state of component i at time step k+1.
                       - Length of each y_history[i] is len(t_grid) - 1 (excludes initial condition).
                       - y_history[:-1] contains membrane potential histories for each layer.
                       - y_history[-1] contains spike output history (final SNN layer, not integrated).
    """
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = (t_grid[-1] - t_grid[0]) / (N - 1)
    # gamma_alpha = 1 / math.gamma(alpha)
    # h_alpha_over_alpha = torch.pow(h, alpha) / alpha *
    h_alpha_over_alpha = torch.pow(h, alpha) / (alpha * math.gamma(alpha))

    # Number of components to integrate (all except last spike component)
    n_integrate = len(y0_tuple) - 1

    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]

    # History for function evaluations (only for integrated components)
    fhistory = [[] for _ in range(n_integrate)]

    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Store function evaluations for integrated components
        for i in range(n_integrate):
            fhistory[i].append(f_k[i])

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute weights b_{j,k+1} for indices from start_idx to k
        # b_{j,k+1} = (h^α / α) * [(k+1-j)^α - (k-j)^α]
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
        # the most restrict formulation should be
        b_j_kp1 = h_alpha_over_alpha * (
                torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
        )
        # b_j_kp1 = h_alpha_over_alpha * (
        #         torch.pow(k + 2 - j_vals, alpha) - torch.pow(k + 1 - j_vals, alpha)
        # )

        # Update integrated states
        for i in range(n_integrate):
            convolution_sum = 0
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx
                convolution_sum = convolution_sum + b_j_kp1[local_idx] * fhistory[i][j]
            y_current[i] = convolution_sum
            y_history[i].append(y_current[i])

        # Pass-through output (spike) - last component is not integrated
        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])
    del fhistory
    return y_history


def vopred_time_integrate_tuple(ode_func, y0_tuple, alpha_time, t_grid: torch.Tensor, memory=None):
    """Predictor method with one network-wide learnable order per integration step."""
    if not isinstance(y0_tuple, tuple) or not y0_tuple:
        raise ValueError("y0_tuple must be a non-empty tuple")

    n_points = len(t_grid)
    n_steps = n_points - 1
    if alpha_time.ndim != 1 or alpha_time.numel() != n_steps:
        raise ValueError(
            f"alpha_time must have {n_steps} values for this grid, got shape {tuple(alpha_time.shape)}"
        )

    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    alpha_time = alpha_time.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[0]) / n_steps

    n_integrate = len(y0_tuple) - 1
    y_history = [[] for _ in y0_tuple]
    f_history = [[] for _ in range(n_integrate)]
    y_current = list(y0_tuple)

    for k in range(n_steps):
        f_k = ode_func(t_grid[k], tuple(y_current))
        for i in range(n_integrate):
            f_history[i].append(f_k[i])

        memory_length = k + 1 if memory is None else min(memory, k + 1)
        if memory_length <= 0:
            raise ValueError("memory must be greater than 0")
        start_idx = k + 1 - memory_length

        alpha_k = alpha_time[k]
        factor = torch.pow(h, alpha_k) / torch.exp(torch.lgamma(alpha_k + 1.0))
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
        weights = factor * (
            torch.pow(k + 1 - j_vals, alpha_k) - torch.pow(k - j_vals, alpha_k)
        )

        for i in range(n_integrate):
            convolution_sum = torch.zeros_like(y_current[i])
            for local_idx, j in enumerate(range(start_idx, k + 1)):
                convolution_sum = convolution_sum + weights[local_idx] * f_history[i][j]
            y_current[i] = convolution_sum
            y_history[i].append(y_current[i])

        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    return y_history




def step_dynamics(ode_func, y0_tuple, t_grid: torch.Tensor):
    """
    Step through a discrete-time dynamical system, collecting spike outputs.

    Repeatedly applies the update function to evolve the state over time.
    Unlike numerical integrators (Euler, GL), no dt scaling is applied.

    The update function directly computes the next state!

    Args:
        ode_func: Callable (t, y_tuple) -> tuple. State update function.
        y0_tuple: Tuple of initial state tensors.
        t_grid: 1D tensor of time points (length T+1).

    Returns:
        list: Spike outputs (last component of state) at each time step.
    """
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    T = t_grid.numel() - 1
    assert T >= 1

    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    y = list(y0_tuple)
    spike_history = []

    for k in range(T):
        t_k = t_grid[k]
        y = ode_func(t_k, tuple(y))
        spike_history.append(y[-1])

    return spike_history


# def euler_integrate_tuple(ode_func, y0_tuple, t_grid: torch.Tensor):
#     """
#     修改版：对LIF神经元使用半隐式方法
#     """
#     assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
#     T = t_grid.numel()
#     assert T >= 1
#     device = y0_tuple[0].device
#     dtype = y0_tuple[0].dtype
#     t_grid = t_grid.to(device=device, dtype=dtype)
#
#     outs = [torch.empty((T,) + y0.shape, device=device, dtype=dtype) for y0 in y0_tuple]
#     y = [y0.clone() for y0 in y0_tuple]
#
#     for i, y0 in enumerate(y):
#         outs[i][0].copy_(y0)
#
#     for k in range(T - 1):
#         tk = t_grid[k]
#         dt = t_grid[k + 1] - tk
#
#         # 第一次调用：获取基本导数
#         dy = ode_func(tk, tuple(y))
#         assert isinstance(dy, tuple) and len(dy) == len(y)
#
#         # 预测步（用于计算spike）
#         y_predict = [y[i] + dt * dy[i] for i in range(len(y))]
#
#         # 第二次调用：基于预测值重新计算（关键！）
#         # 这模拟了隐式方法
#         dy_corrected = ode_func(tk, tuple(y_predict))
#
#         # 使用修正后的导数
#         for i in range(len(y)):
#             if True:  # 假设第一个是膜电位
#                 # 对膜电位使用修正的导数
#                 y[i] = y[i] + dt * dy_corrected[i]
#             else:
#                 # 其他变量使用原始导数
#                 y[i] = y[i] + dt * dy[i]
#             outs[i][k + 1].copy_(y[i])
#     return tuple(outs)

#------------------------------------ the following is distributed-order ------------------------------------#

@torch.compile(mode="reduce-overhead")
def glmethod_multiterm_integrate_tuple(ode_func, y0_tuple, beta, t_grid: torch.Tensor, memory=None,
                                       multi_coefficient=None):
    """
    Grünwald-Letnikov method for multi-term RL fractional differential equations with tuple state.
    Σ w_j D^{β_j} y(t) = f(t,y)

    Based on the distributed GL scheme where the aggregated weights encode
    the multi-term structure through: c̃_m = Σ w_j h^{-α_j} c_m^{(α_j)}

    Args:
        ode_func: callable(t, y_tuple) returns tuple with same structure as y_tuple
        y0_tuple: tuple of initial states
        beta: Tensor of fractional orders, each in the range (0,1), shape (n_terms,)
        t_grid: time points sequence (must be equally spaced)
        memory: history window size (None means use full history)
        multi_coefficient: Tensor of weights w_j for each term, shape (n_terms,)

    Returns:
        tuple of lists: history of all state components at all time points
    """
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    assert multi_coefficient is not None, "multi_coefficient must be provided"

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = (t_grid[-1] - t_grid[0]) / (N - 1)

    # Ensure beta and coefficient are tensors on the correct device
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=dtype, device=device)
    else:
        beta = beta.to(device=device, dtype=dtype)

    if not isinstance(multi_coefficient, torch.Tensor):
        coefficient = torch.tensor(multi_coefficient, dtype=dtype, device=device)
    else:
        coefficient = multi_coefficient.to(device=device, dtype=dtype)

    n_terms = len(beta)
    assert len(coefficient) == n_terms, "beta and coefficient must have the same length"

    # Compute GL coefficients c_m^{(α_j)} for each fractional order
    # c[j, m] = c_m^{(α_j)} where c_0 = 1, c_m = (1 - (1+α)/m) * c_{m-1}
    c = torch.zeros(n_terms, N + 1, dtype=dtype, device=device)
    c[:, 0] = 1.0
    for m in range(1, N + 1):
        # Vectorized over all terms: c[j,m] = (1 - (1+β_j)/m) * c[j,m-1]
        c[:, m] = (1 - (1 + beta) / m) * c[:, m - 1]

    # Compute h^{-α_j} for each term
    h_neg_power = torch.pow(h, -beta)  # Shape: (n_terms,)

    # Compute distributed GL weights: c̃_m = Σ w_j * h^{-α_j} * c_m^{(α_j)}
    # Vectorized: (n_terms,) * (n_terms,) * (n_terms, N+1) -> sum over dim 0 -> (N+1,)
    weighted_h = coefficient * h_neg_power  # Shape: (n_terms,)
    c_tilde = torch.sum(weighted_h.unsqueeze(1) * c, dim=0)  # Shape: (N+1,)

    # c̃_0 = Σ w_j * h^{-α_j} (since c_0^{(α_j)} = 1 for all j)
    c_tilde_0 = c_tilde[0]
    c_tilde_0_inv = 1.0 / c_tilde_0

    # Number of components to integrate (all except last spike component)
    n_integrate = len(y0_tuple) - 1

    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]

    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Update integrated states (all except last spike component)
        for i in range(n_integrate):
            # Compute convolution sum: Σ c̃_{k+1-j} * y_j[i] for j from start_idx to k
            # This corresponds to Σ_{m=1}^{min(k+1,M)} c̃_m * y_{k+1-m}
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):  # range(start_idx, k) because we have k elements (without y0=0)
                    coeff_idx = k - j  # history[j] is y_{j+1}; nearest state uses c_1.
                    convolution_sum = convolution_sum + c_tilde[coeff_idx] * y_history[i][j]
            else:
                convolution_sum = 0

            # y_{k+1} = (1/c̃_0) * (f_k - convolution_sum)
            y_current[i] = c_tilde_0_inv * (f_k[i] - convolution_sum)
            y_history[i].append(y_current[i])

        # Pass-through output (spike) - last component is not integrated
        y_current[-1] = f_k[-1]
        y_history[-1].append(y_current[-1])

    return y_history



#------------------------------------ the following is adjoint training code ------------------------------------#

def fdeint_adjoint(func, y0_tuple, alpha, t_grid, method, memory=None, backend="torch",
                   multi_coefficient=None):
    # params = tuple(p for p in func.parameters())  # 或只取 requires_grad=True 的
    params = find_parameters(func)
    n_state = len(y0_tuple)
    n_params = len(params)
    # 注意：apply 不接受关键字参数；把计数放在前两位最简单
    return FDEAdjointMethod.apply(
        func, n_state, n_params, *y0_tuple, alpha, t_grid, method, backend, *params, memory, multi_coefficient
    )


class FDEAdjointMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ode_func, n_state, n_params, *args):
        # n_state, n_params 是 Python int（非 Tensor）；其梯度应返回 None
        """
        Forward pass
        Args:
            ode_func: ODE 函数
            y0_tuple: 初始状态 (tuple of tensors)
            alpha: 参数
            t_grid: 时间网格
            method: 积分方法
            func_params: 模型参数 (tuple)
            memory: 额外内存参数
        """
        n_state = int(n_state); n_params = int(n_params)

        # 解析位置参数： y0_1,...,y0_n, alpha, t_grid, method, p1,...,pm, memory, multi_coefficient
        y0_tuple   = tuple(args[:n_state])                               # Tensors
        alpha       = args[n_state]                                      # Tensor 或 float（若要学习，必须是 Tensor）
        t_grid     = args[n_state + 1]                                  # Tensor
        method     = args[n_state + 2]                                  # str / enum（非 Tensor）
        backend    = args[n_state + 3]
        func_params= tuple(args[n_state + 4 : n_state + 4 + n_params])   # Tensors (Parameters)
        memory     = args[n_state + 4 + n_params]                       # 任意对象（非 Tensor）
        multi_coefficient = args[n_state + 5 + n_params]

        with torch.no_grad():
            solver = _get_solver(backend, method, "forward")
            if method == "glmulti":
                yhistory = solver(
                    ode_func=ode_func,
                    y0_tuple=y0_tuple,
                    alpha=alpha,
                    t_grid=t_grid,
                    memory=memory,
                    multi_coefficient=multi_coefficient
                )
            else:
                yhistory = solver(
                    ode_func=ode_func,
                    y0_tuple=y0_tuple,
                    alpha=alpha,
                    t_grid=t_grid,
                    memory=memory
                )

        # 检查是否需要梯度
        y0_needs_grad = any(t.requires_grad for t in y0_tuple)
        params_need_grad = any(p.requires_grad for p in func_params) if func_params else False
        multi_coeff_needs_grad = (
            torch.is_tensor(multi_coefficient) and multi_coefficient.requires_grad
        )

        ctx.n_state = n_state
        ctx.n_params = n_params
        ctx.multi_coefficient = multi_coefficient
        if y0_needs_grad or params_need_grad or multi_coeff_needs_grad:
            # 保存必要的张量用于反向传播
            # 注意：保存整个 tuple 对象，而不是展开的张量
            ctx.save_for_backward(t_grid)
            ctx.func_params = func_params  # 同样保存为属性
            ctx.yhistory = yhistory  # 保存最终状态
            ctx.ode_func = ode_func
            ctx.alpha = alpha
            ctx.method = method
            ctx.memory = memory
            ctx.backend = backend
            ctx.multi_coefficient = multi_coefficient
            ctx.y0_tuple = y0_tuple

        # 返回结果
        # if method == 'euler':
        #     outs = tuple(yhistory)
        # else:
        #     # 如果 yhistory 是嵌套列表，取最后一个
        outs = tuple([y[-1] for y in yhistory])

        return outs

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Backward pass - 计算梯度
        """
        # 早退：不需要反传时，返回正确数量的 None
        if not hasattr(ctx, 'yhistory'):
            n_state = ctx.n_state
            n_params = ctx.n_params
            grads = []
            grads.append(None)  # ode_func
            grads.append(None)  # n_state
            grads.append(None)  # n_params
            grads.extend([None] * n_state)  # y0_1,...,y0_n
            grads.append(None)  # alpha
            grads.append(None)  # t_grid
            grads.append(None)  # method
            grads.append(None)  # backend
            grads.extend([None] * n_params)  # p1,...,pm
            grads.append(None)  # memory
            grads.append(None)  # multi_coefficient
            return tuple(grads)

        # 恢复保存的张量和属性
        t_grid = ctx.saved_tensors[0]
        func_params = ctx.func_params
        yhistory = ctx.yhistory
        # yhistory is the last states for euler and the full history for other methods

        func = ctx.ode_func
        alpha = ctx.alpha
        method = ctx.method
        memory = ctx.memory
        n_tensors = len(yhistory)

        # 创建 augmented dynamics
        class AugDynamics:
            def __init__(self, func, n_tensors, func_params):
                self.func = func
                self.n_tensors = n_tensors
                self.f_params = func_params  # 使用传入的参数

            def __call__(self, t, y_aug):
                y, adj_y, adj_params = y_aug

                with torch.set_grad_enabled(True):
                    # detach 并设置 requires_grad
                    y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                    func_eval = self.func(t, y)

                    # 计算 VJP
                    vjp_y_and_params = torch.autograd.grad(
                        func_eval,
                        y + self.f_params,
                        tuple(adj_y),
                        allow_unused=True,
                        retain_graph=False,  # 不保留图
                        create_graph=False
                    )

                vjp_y = vjp_y_and_params[:self.n_tensors]
                vjp_params = vjp_y_and_params[self.n_tensors:]

                # 处理 None 梯度
                vjp_y = tuple(
                    torch.zeros_like(y_) if vjp_y_ is None else vjp_y_
                    for vjp_y_, y_ in zip(vjp_y, y)
                )


                vjp_params = tuple(
                        torch.zeros_like(p) if vp is None else vp
                        for vp, p in zip(vjp_params, self.f_params)
                    )


                return (func_eval, vjp_y, vjp_params)

        # 创建 augmented dynamics 实例
        augmented_dynamics = AugDynamics(func, n_tensors, func_params)
        t_grid_flip = t_grid.flip(0)

        with torch.no_grad():
            adj_y = grad_output

            # 初始化参数梯度
            if func_params:
                adj_params = tuple(torch.zeros_like(p) for p in func_params)
            else:
                adj_params = ()

            # 设置初始增广状态
            # 注意：yhistory 作为初始 y 状态
            aug_y0 = ([], adj_y, adj_params)

            # 调用反向求解器
            solver = _get_solver(ctx.backend, method, "backward")
            if method == "glmulti":
                adj_y, adj_params = solver(
                    augmented_dynamics,
                    aug_y0,
                    alpha,
                    t_grid_flip,
                    yhistory,  # 传递最终状态
                    memory,
                    ctx.multi_coefficient,
                )
            else:
                adj_y, adj_params = solver(
                    augmented_dynamics,
                    aug_y0,
                    alpha,
                    t_grid_flip,
                    yhistory,  # 传递最终状态
                    memory
                )

        grad_multi_coeff = None
        if (
            ctx.multi_coefficient is not None
            and torch.is_tensor(ctx.multi_coefficient)
            and ctx.multi_coefficient.requires_grad
        ):
            with torch.set_grad_enabled(True):
                y0_tuple = tuple(y_.detach() for y_ in ctx.y0_tuple)
                alpha = ctx.alpha
                if torch.is_tensor(alpha):
                    alpha = alpha.detach()
                yhistory_fwd = forward_glmulti(
                    ode_func=ctx.ode_func,
                    y0_tuple=y0_tuple,
                    alpha=alpha,
                    t_grid=t_grid,
                    memory=memory,
                    multi_coefficient=ctx.multi_coefficient,
                )
                outs = tuple([y[-1] for y in yhistory_fwd])
                grad_multi_coeff = torch.autograd.grad(
                    outs,
                    ctx.multi_coefficient,
                    grad_outputs=grad_output,
                    allow_unused=True,
                )[0]

        # 清理
        # del augmented_dynamics
        # del ctx.yhistory

        # 在最后，确保清理所有局部变量
        del augmented_dynamics
        del yhistory  # 也要删除局部变量
        del func
        del func_params
        del ctx.yhistory
        del ctx.ode_func
        del ctx.func_params
        del ctx.alpha
        del ctx.method
        if hasattr(ctx, 'y0_tuple'):
            del ctx.y0_tuple

        # 准备返回值
        # 返回格式：(grad_func, grad_y0_tuple, grad_alpha, grad_t_grid, grad_method, grad_func_params, grad_memory)
        grad_y0 = adj_y
        grad_params = adj_params

        return None, None, None, *grad_y0, None, None, None, None, *grad_params, None, grad_multi_coeff


def forward_euler_wo_history(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    手写 Euler 积分（显式），支持 tuple 状态。
    - ode_func: callable(t, y_tuple) -> tuple(dy_i)
    - y0_tuple: tuple(tensor, ...) 初始状态 == 0
    - t_grid: 1D Tensor，[t0, t1, ..., tN]，步长可不等（会逐段用 dt = t_{k+1}-t_k）
    返回：tuple( Y_i )，其中 Y_i 的形状是 (N, *y0_i.shape)
    """
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    T = t_grid.numel() - 1
    assert T >= 1
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    # Initialize history lists for each component

    # Clone initial values
    y = list(y0_tuple)

    # Euler integration: y_{k+1} = y_k + dt * f(t_k, y_k)
    for k in range(T):
        tk = t_grid[k]
        dt = t_grid[k + 1] - t_grid[k]  # Scalar tensor, will broadcast automatically
        dy = ode_func(tk, tuple(y))  # Expect tuple return, consistent with y structure
        assert isinstance(dy, tuple) and len(dy) == len(y)
        # Update all integrated components except the last one
        # for i in range(len(y)):
        #     y[i] = y[i] + dt * dy[i]
        for i in range(len(y)):
            y[i].add_(dy[i], alpha=dt)
    return y

def backward_euler_wo_history(ode_func, y_aug, alpha, t_grid, y_finalstate, memory=None):
    with torch.no_grad():
        T = t_grid.numel() - 1
        h = torch.abs((t_grid[-1] - t_grid[0]) / (T - 1))
        h = float(h)

        _, adj_y0, adj_params0 = y_aug
        # 初始化
        # y_state = [x.detach().clone() for x in y_finalstate]
        # adj_y = [x.detach().clone() for x in adj_y0]
        # adj_params = tuple(p.detach().clone() for p in adj_params0) if adj_params0 else ()
        y_state = list(y_finalstate)
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

    # return [x.detach().clone() for x in adj_y0], [x.detach().clone() for x in adj_params0]

        for k in range(T):
            tk = t_grid[k]

            # 调用 augmented dynamics
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # 更新状态
            for i in range(len(adj_y)):
                # adj_y[i] = adj_y[i] + h * vjp_y[i]
                # y_state[i] = y_state[i] - h * func_eval[i]
                adj_y[i].add_(vjp_y[i], alpha=h)
                y_state[i].add_(func_eval[i], alpha=-h)  # 注意这里是 -h（减法）

            # 更新参数梯度
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    return adj_y, adj_params


def forward_euler_w_history(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    手写 Euler 积分（显式），支持 tuple 状态。
    - ode_func: callable(t, y_tuple) -> tuple(dy_i)
    - y0_tuple: tuple(tensor, ...) 初始状态 == 0
    - t_grid: 1D Tensor，[t0, t1, ..., tN]，步长可不等（会逐段用 dt = t_{k+1}-t_k）
    返回：tuple( Y_i )，其中 Y_i 的形状是 (N, *y0_i.shape)
    """
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    T = t_grid.numel() - 1
    assert T >= 1
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    # Initialize history lists for each component

    # Clone initial values
    y = list(y0_tuple)
    y_history = [[] for _ in y0_tuple]

    # Euler integration: y_{k+1} = y_k + dt * f(t_k, y_k)
    for k in range(T):
        tk = t_grid[k]
        dt = t_grid[k + 1] - t_grid[k]  # Scalar tensor, will broadcast automatically
        dy = ode_func(tk, tuple(y))  # Expect tuple return, consistent with y structure
        assert isinstance(dy, tuple) and len(dy) == len(y)

        # Update all integrated components except the last one
        for i in range(len(y)):
            y[i] = y[i] + dt * dy[i]
        # Final element is the output spike (pass-through)
            y_history[i].append(y[i])
    return y_history


def backward_euler_w_history(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    with torch.no_grad():
        T = t_grid.numel() - 1
        h = torch.abs((t_grid[-1] - t_grid[0]) / (T - 1))
        h = float(h)

        _, adj_y0, adj_params0 = y_aug

        adj_y = [y.clone() for y in adj_y0]  # [y0.clone() for y0 in y0_tuple]
        adj_params = list(adj_params0)

        # return tuple(y_i.clone() for y_i in adj_y0), tuple(y_i.clone() for y_i in adj_params0)

        for k in range(T):
            tk = t_grid[k]
            y_state = list([y[-1-k] for y in yhistory])

            # 调用 augmented dynamics
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))
            # vjp_y = tuple(torch.zeros_like(y_i) for y_i in adj_y)
            # vjp_params = tuple(torch.zeros_like(y_i) for y_i in adj_params)
            # 更新状态
            # for i in range(len(adj_y)):
            #     adj_y[i] = adj_y[i] + h * vjp_y[i]
            for i in range(len(adj_y)):
                adj_y[i].add_(vjp_y[i], alpha=h)

            # 更新参数梯度
            # if adj_params and vjp_params:
            #     adj_params = tuple(
            #         ap + h * vp for ap, vp in zip(adj_params, vjp_params)
            #     )
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    del yhistory, vjp_params, func_eval, vjp_y

    return adj_y, adj_params


def backward_euler_w_history_cupy(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    if cupy is None:
        raise RuntimeError("CuPy backend requested but CuPy is unavailable.")
    with torch.no_grad():
        if not yhistory:
            raise ValueError("yhistory must not be empty for CuPy Euler adjoint.")
        T = t_grid.numel() - 1
        h = torch.abs((t_grid[-1] - t_grid[0]) / (T - 1))
        h = float(h)

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype
        _assert_cupy_backend_ready(device, dtype)

        adj_y = [y.clone() for y in adj_y0]
        adj_params = list(adj_params0)

        for k in range(T):
            tk = t_grid[k]
            y_state = [y[-1 - k] for y in yhistory]
            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            for adj, vjp in zip(adj_y, vjp_y):
                _cupy_add_scaled_inplace(adj, vjp, h)

            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    _cupy_add_scaled_inplace(ap, vp, h)

    del yhistory, vjp_params, func_eval, vjp_y

    return adj_y, adj_params


def forward_gl(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    N = t_grid.numel()
    assert N >= 2
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[0]) / (N - 1)
    h_alpha = torch.pow(h, alpha)

    # GL coefficients: need up to c[N]
    c = torch.zeros(N + 1, dtype=dtype, device=device)
    c[0] = 1
    for j in range(1, N + 1):
        c[j] = (1 - (1 + alpha) / j) * c[j - 1]

    # Initialize with y_0 (clone to avoid modifying input)
    y_current = list(y0_tuple)  # tuple(y.clone() for y in y0_tuple)
    # History: y_history[i][j] stores y_j for component i
    # Initialize with y_0
    # y_history = [[y, ] for y in y0_tuple]
    y_history = [[] for y in y0_tuple]
    # y_history = [[y.clone()] for y in y0_tuple]

    for k in range(N - 1):
        t_k = t_grid[k]
        # Evaluate f(t_k, y_k)

        dy = ode_func(t_k, tuple(y_current))
        # assert isinstance(dy, tuple) and len(dy) == len(y_current)

        # Determine memory range
        if memory is None or memory == -1:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)

        start_idx = max(0, k + 1 - memory_length)

        # Update all integrated components
        for i in range(len(y_current)):
            # Accumulate: Σ c_{k+1-j} * y_j for j from start_idx to k
            if k > 0:
                convolution_sum = 0#torch.zeros_like(y[i])
                for j in range(start_idx, k):
                    # GL coefficient for this lag
                    convolution_sum = convolution_sum + c[k+1-j] * y_history[i][j]
                # here we assume at time k, we have k elements (without y0=0)
                # the most restrict formulation should be convolution_sum + c[k-j] * y_history[i][j]
                # which however seems do have have good numerical stability
            else:
                convolution_sum = 0

            # convolution_sum = None
            # for j in range(start_idx, k+1):
            # # # GL coefficient for this lag
            #     if convolution_sum is None:
            #         convolution_sum = c[k+1-j] * y_history[i][j]
            #     else:
            #         convolution_sum = convolution_sum + c[k+1-j] * y_history[i][j]


            # y_{k+1} = h^alpha * f_k - convolution_sum
            y_current[i] = h_alpha * dy[i] - convolution_sum
            y_history[i].append(y_current[i])

    return y_history

def backward_gl(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    with torch.no_grad():
        N = t_grid.numel()
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha = torch.pow(h, alpha)

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype

        # GL coefficients
        c = torch.zeros(N + 1, dtype=dtype, device=device)
        c[0] = 1
        for j in range(1, N + 1):
            c[j] = (1 - (1 + alpha) / j) * c[j - 1]

        # Initialize adjoint history lists for each component
        adjy_history = [[xx, ] for xx in adj_y0]

        # Clone initial adjoint values
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        for k in range(N - 2):
            tk = t_grid[k]
            # y_state = list([y[-k - 1] for y in yhistory])
            y_state = list([y[-k - 2] for y in yhistory])

            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # Determine memory range
            if memory is None or memory == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(memory, k + 1)

            start_idx = max(0, k + 1 - memory_length)

            # Update all adjoint components
            for i in range(len(adj_y)):
                # Calculate history sum

                if True:#k > 0:
                    convolution_sum = 0  # torch.zeros_like(y[i])
                    for j in range(start_idx, k+1):
                        # GL coefficient for this lag
                        convolution_sum = convolution_sum + c[k+1-j] * adjy_history[i][j]
                    # here we assume at time k, we have k elements (without y0=0)
                    # the most restrict formulation should be convolution_sum + c[k-j] * y_history[i][j]
                    # which however seems do have have good numerical stability
                else:
                    convolution_sum = 0

                # convolution_sum = None
                # for j in range(start_idx, k+1):
                #     # # GL coefficient for this lag
                #     if convolution_sum is None:
                #         convolution_sum = c[k + 1 - j] * adjy_history[i][j]
                #     else:
                #         convolution_sum = convolution_sum + c[k + 1  - j] * adjy_history[i][j]

                # Update adjoint state
                adj_y[i] = h_alpha * vjp_y[i] - convolution_sum
                adjy_history[i].append(adj_y[i])

            # 更新参数梯度
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    del adjy_history, yhistory

    return adj_y, adj_params


def forward_trap(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Forward pass of Product Trapezoidal method for adjoint training.
    """
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0

    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)

    h = (t_grid[-1] - t_grid[0]) / (N - 1)
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    # Initialize with y_0
    y_current = list(y0_tuple)
    y_history = [[] for y in y0_tuple]

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]

        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, tuple(y_current))

        # Determine memory range
        if memory is None:
            memory_length = k + 1
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute A_{j,k+1} weights for indices from start_idx to k
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

        # General formula for j >= 1:
        # A_{j,k+1} = (k+2-j)^{1-α} + (k-j)^{1-α} - 2(k+1-j)^{1-α}
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        A_j_kp1 = kjp2 + kj - 2 * kjp1

        # Special handling for j=0 if it's in the range:
        # A_{0,k+1} = k^{1-α} - (k+α)(k+1)^{-α}
        if start_idx == 0:
            k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
            kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -alpha)
            A_j_kp1[0] = k_power - (k + alpha) * kp1_neg_alpha

        # Update ALL state components (forward integrates all, not len-1)
        for i in range(len(y_current)):
            # Compute convolution sum: sum_{j=start_idx}^{k-1} A_{j,k+1} * y_j[i]
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + A_j_kp1[local_idx] * y_history[i][j]
            else:
                convolution_sum = 0

            # y_{k+1} = Γ(2-α) * h^α * f_k - convolution_sum
            y_current[i] = h_alpha_gamma * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

    return y_history


def backward_trap(ode_func, y_aug, alpha, t_grid, yhistory, memory=None):
    """
    Backward pass of Product Trapezoidal method for adjoint training.
    """
    with torch.no_grad():
        N = t_grid.numel()
        h = torch.abs(t_grid[-1] - t_grid[-2])
        h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
        one_minus_alpha = 1 - alpha

        _, adj_y0, adj_params0 = y_aug
        device = adj_y0[0].device
        dtype = adj_y0[0].dtype

        # Initialize adjoint history lists for each component (with initial values)
        adjy_history = [[xx, ] for xx in adj_y0]

        # Clone initial adjoint values
        adj_y = list(adj_y0)
        adj_params = list(adj_params0)

        for k in range(N - 2):
            tk = t_grid[k]
            # y_state = list([y[-k - 1] for y in yhistory])
            y_state = list([y[-k - 2] for y in yhistory])

            func_eval, vjp_y, vjp_params = ode_func(tk, (y_state, adj_y, adj_params))

            # Determine memory range
            if memory is None:
                memory_length = k + 1
            else:
                memory_length = min(memory, k + 1)

            start_idx = max(0, k + 1 - memory_length)

            # Compute A_{j,k+1} weights for indices from start_idx to k
            j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

            # General formula for j >= 1:
            # A_{j,k+1} = (k+2-j)^{1-α} + (k-j)^{1-α} - 2(k+1-j)^{1-α}
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
            kj = torch.pow(k - j_vals, one_minus_alpha)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
            A_j_kp1 = kjp2 + kj - 2 * kjp1

            # Special handling for j=0 if it's in the range:
            # A_{0,k+1} = k^{1-α} - (k+α)(k+1)^{-α}
            if start_idx == 0:
                k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
                kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -alpha)
                A_j_kp1[0] = k_power - (k + alpha) * kp1_neg_alpha

            # Update all adjoint components
            for i in range(len(adj_y)):
                # Calculate history sum - note: range goes to k+1 (one more than forward)
                convolution_sum = 0
                for j in range(start_idx, k + 1):
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + A_j_kp1[local_idx] * adjy_history[i][j]

                # Update adjoint state
                adj_y[i] = h_alpha_gamma * vjp_y[i] - convolution_sum
                adjy_history[i].append(adj_y[i])

            # Update parameter gradients
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

    del adjy_history, yhistory

    return adj_y, adj_params


def forward_l1(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None):
    """
    Forward pass of L1 method for adjoint training.
    """
    assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
    N = len(t_grid)
    device = y0_tuple[0].device
    dtype = y0_tuple[0].dtype
    t_grid = t_grid.to(device=device, dtype=dtype)
    h = (t_grid[-1] - t_grid[0]) / (N - 1)
    h_alpha_gamma = torch.pow(h, alpha) * math.gamma(2 - alpha)
    one_minus_alpha = 1 - alpha

    # Initialize history lists for each component
    y_history = [[] for _ in y0_tuple]
    # Current state
    y_current = list(y0_tuple)

    # Main loop: compute y_{k+1} for k = 0, 1, ..., N-2
    for k in range(N - 1):
        t_k = t_grid[k]
        # Evaluate f(t_k, y_k)
        f_k = ode_func(t_k, y_current)

        # Determine memory range
        if memory is None:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(memory, k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute c_j^(k) weights for indices from start_idx to k
        # General formula for j >= 1: c_j^(k) = (k-j+2)^{1-α} - 2(k-j+1)^{1-α} + (k-j)^{1-α}
        # Special case for j = 0: c_0^(k) = -[(k+1)^{1-α} - k^{1-α}]
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_alpha)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_alpha)
        kj = torch.pow(k - j_vals, one_minus_alpha)
        c_j_k = kjp2 - 2 * kjp1 + kj

        # Special handling for j=0 if it's in the range
        if start_idx == 0:
            kp1_power = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), one_minus_alpha)
            k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_alpha)
            c_j_k[0] = -(kp1_power - k_power)

        # Update ALL state components (forward integrates all, not len-1)
        for i in range(len(y_current)):
            # Compute convolution sum: sum_{j=start_idx}^{k-1} c_j^(k) * y_j[i]
            if k > 0:
                convolution_sum = 0
                for j in range(start_idx, k):
                    local_idx = j - start_idx
                    convolution_sum = convolution_sum + c_j_k[local_idx] * y_history[i][j]
            else:
                convolution_sum = 0

            # y_{k+1} = h^α * Γ(2-α) * f_k - convolution_sum
            y_current[i] = h_alpha_gamma * f_k[i] - convolution_sum
            y_history[i].append(y_current[i])

    return y_history


def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


#### it seems torch compile cannot speed up the methods


def forward_glmulti(ode_func, y0_tuple, alpha, t_grid: torch.Tensor, memory=None, multi_coefficient=None):
    return glmethod_multiterm_integrate_tuple(
        ode_func,
        y0_tuple,
        alpha,
        t_grid,
        memory=memory,
        multi_coefficient=multi_coefficient
    )


forward_gl_compiled = torch.compile(forward_gl)
backward_gl_compiled = torch.compile(backward_gl)
forward_trap_compiled = torch.compile(forward_trap)
backward_trap_compiled = torch.compile(backward_trap)
forward_l1_compiled = torch.compile(forward_l1)
backward_l1_compiled = torch.compile(backward_l1)
forward_pred_compiled = torch.compile(forward_pred)
backward_pred_compiled = torch.compile(backward_pred)
forward_glmulti_compiled = torch.compile(forward_glmulti)


forward_euler_w_history_compiled = torch.compile(forward_euler_w_history)
forward_euler_wo_history_compiled = torch.compile(forward_euler_wo_history)
backward_euler_w_history_compiled = torch.compile(backward_euler_w_history)
backward_euler_wo_history_compiled = torch.compile(backward_euler_wo_history)


TORCH_SOLVERS_FORWARD = {
           "gl-f":forward_gl_compiled,
           "gl-o":forward_gl_compiled,
           "trap-f":forward_trap_compiled,
           "trap-o":forward_trap_compiled,
            "l1-f":forward_l1_compiled,
            "l1-o":forward_l1_compiled,
            "pred-f": forward_pred_compiled,
            "pred-o": forward_pred_compiled,
            "glmulti": forward_glmulti_compiled,
            "euler": forward_euler_w_history_compiled,
}

TORCH_SOLVERS_BACKWARD = {

           "gl-f":backward_gl_compiled,
           "gl-o":backward_euler_w_history_compiled,
           "trap-f":backward_trap_compiled,
           "trap-o":backward_euler_w_history_compiled,
            "l1-f":backward_trap_compiled,
           "l1-o":backward_euler_w_history_compiled,
            "pred-f": backward_pred_compiled,
            "pred-o": backward_euler_w_history_compiled,
            "glmulti": backward_glmulti_cupy,
            "euler": backward_euler_w_history_compiled,
}

CUPY_SOLVERS_FORWARD = {
    "pred-f": forward_pred_cupy,
    "pred-o": forward_pred_cupy,
    "gl-f": forward_gl_cupy,
    "gl-o": forward_gl_cupy,
    "trap-f": forward_trap_cupy,
    "trap-o": forward_trap_cupy,
    "l1-f": forward_l1_cupy,
    "l1-o": forward_l1_cupy,
    "glmulti": forward_glmulti_cupy,
}

CUPY_SOLVERS_BACKWARD = {
    "pred-f": backward_pred_cupy,
    "pred-o": backward_euler_w_history_cupy,
    "gl-f": backward_gl_cupy,
    "gl-o": backward_euler_w_history_cupy,
    "trap-f": backward_trap_cupy,
    "trap-o": backward_euler_w_history_cupy,
    "l1-f": backward_trap_cupy,
    "l1-o": backward_euler_w_history_cupy,
    "glmulti": backward_glmulti_cupy,
}

_SOLVER_BACKENDS = {
    "torch": {"forward": TORCH_SOLVERS_FORWARD, "backward": TORCH_SOLVERS_BACKWARD},
    "cupy": {"forward": CUPY_SOLVERS_FORWARD, "backward": CUPY_SOLVERS_BACKWARD},
}


def _get_solver(backend, method, direction):
    backend = backend.lower()
    if backend not in _SOLVER_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'")
    solver = _SOLVER_BACKENDS[backend][direction].get(method)
    if solver is None:
        raise ValueError(f"Backend '{backend}' does not support method '{method}' for {direction} pass")
    return solver
