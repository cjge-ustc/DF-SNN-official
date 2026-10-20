import math

import torch
import numpy as np

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    cupy = None
    CUPY_AVAILABLE = False


def _to_cupy(tensor):
    return cupy.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


_forward_kernel = None
_backward_kernel = None


def _get_forward_kernel():
    global _forward_kernel
    if _forward_kernel is not None:
        return _forward_kernel
    code = r'''
extern "C" __global__
void direct_lif_euler_forward(
    const float* __restrict__ x,
    const float* __restrict__ z_init,
    float* __restrict__ spike,
    float* __restrict__ z_pre,
    float* __restrict__ spike_input,
    float* __restrict__ z_final,
    const int T,
    const int N,
    const float input_scale,
    const float spike_gain,
    const float spike_bias,
    const float tau,
    const float threshold,
    const float step_size,
    const float reset_strength,
    const float spike_update_dt
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float z = z_init[i];
    for (int t = 0; t < T; ++t) {
        const int idx = t * N + i;
        const float current = x[idx];
        const float dz = (-z + current * input_scale) / tau;
        const float spike_voltage = z + spike_update_dt * dz;
        const float s_in = spike_voltage * spike_gain + spike_bias - threshold;
        const float s = s_in >= 0.0f ? 1.0f : 0.0f;
        z_pre[idx] = z;
        spike_input[idx] = s_in;
        spike[idx] = s;
        z = z + step_size * (dz - reset_strength * s * threshold / tau);
    }
    z_final[i] = z;
}
'''
    _forward_kernel = cupy.RawKernel(code, 'direct_lif_euler_forward')
    return _forward_kernel


def _get_backward_kernel():
    global _backward_kernel
    if _backward_kernel is not None:
        return _backward_kernel
    code = r'''
extern "C" __global__
void direct_lif_euler_backward(
    const float* __restrict__ grad_spike,
    const float* __restrict__ grad_z_final,
    const float* __restrict__ x,
    const float* __restrict__ z_pre,
    const float* __restrict__ spike_input,
    float* __restrict__ grad_x,
    float* __restrict__ grad_z_init,
    float* __restrict__ grad_input_scale,
    float* __restrict__ grad_spike_gain,
    float* __restrict__ grad_spike_bias,
    const int T,
    const int N,
    const float input_scale,
    const float spike_gain,
    const float tau,
    const float step_size,
    const float surrogate_scale,
    const float spike_update_dt
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float pi_half = 1.57079632679489661923f;
    float gz_next = grad_z_final == nullptr ? 0.0f : grad_z_final[i];
    float local_g_scale = 0.0f;
    float local_g_gain = 0.0f;
    float local_g_bias = 0.0f;

    for (int t = T - 1; t >= 0; --t) {
        const int idx = t * N + i;
        const float z = z_pre[idx];
        const float current = x[idx];
        const float s_in = spike_input[idx];
        const float dz_dz = -1.0f / tau;
        const float dz_dx = input_scale / tau;
        const float dz_dscale = current / tau;
        const float svolt_dz = 1.0f + spike_update_dt * dz_dz;
        const float svolt_dx = spike_update_dt * dz_dx;
        const float svolt_dscale = spike_update_dt * dz_dscale;
        const float spike_voltage = z * svolt_dz + current * svolt_dx;

        const float q = pi_half * surrogate_scale * s_in;
        const float surrogate_grad = surrogate_scale / (2.0f * (1.0f + q * q));
        const float g_sin = grad_spike[idx] * surrogate_grad;

        local_g_gain += g_sin * spike_voltage;
        local_g_bias += g_sin;
        const float g_svolt = g_sin * spike_gain;

        float gz = gz_next * (1.0f - step_size / tau) + g_svolt * svolt_dz;
        grad_x[idx] = gz_next * (step_size * dz_dx) + g_svolt * svolt_dx;
        local_g_scale += gz_next * (step_size * dz_dscale) + g_svolt * svolt_dscale;
        gz_next = gz;
    }

    grad_z_init[i] = gz_next;
    atomicAdd(grad_input_scale, local_g_scale);
    atomicAdd(grad_spike_gain, local_g_gain);
    atomicAdd(grad_spike_bias, local_g_bias);
}
'''
    _backward_kernel = cupy.RawKernel(code, 'direct_lif_euler_backward')
    return _backward_kernel


class DirectLIFLikeEulerCuPyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq_flat, z_init_flat, input_scale, spike_gain, spike_bias,
                tau, threshold, step_size, reset_strength, surrogate_scale, spike_update_dt):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available")
        if not x_seq_flat.is_cuda or x_seq_flat.dtype != torch.float32:
            raise ValueError("CuPy direct Euler expects CUDA float32 tensors")
        x_seq_flat = x_seq_flat.contiguous()
        z_init_flat = z_init_flat.contiguous()
        T, N = x_seq_flat.shape
        spike = torch.empty_like(x_seq_flat)
        z_pre = torch.empty_like(x_seq_flat)
        spike_input = torch.empty_like(x_seq_flat)
        z_final = torch.empty_like(z_init_flat)

        threads = 256
        blocks = (N + threads - 1) // threads
        _get_forward_kernel()(
            (blocks,), (threads,),
            (
                _to_cupy(x_seq_flat),
                _to_cupy(z_init_flat),
                _to_cupy(spike),
                _to_cupy(z_pre),
                _to_cupy(spike_input),
                _to_cupy(z_final),
                np.int32(T),
                np.int32(N),
                np.float32(float(input_scale.detach().cpu())),
                np.float32(float(spike_gain.detach().cpu())),
                np.float32(float(spike_bias.detach().cpu())),
                np.float32(float(tau)),
                np.float32(float(threshold)),
                np.float32(float(step_size)),
                np.float32(float(reset_strength)),
                np.float32(float(spike_update_dt)),
            ),
        )
        cupy.cuda.get_current_stream().synchronize()
        ctx.save_for_backward(x_seq_flat, z_pre, spike_input, input_scale, spike_gain)
        ctx.params = (T, N, float(tau), float(step_size), float(surrogate_scale), float(spike_update_dt))
        return spike, z_final

    @staticmethod
    def backward(ctx, grad_spike, grad_z_final):
        x_seq_flat, z_pre, spike_input, input_scale, spike_gain = ctx.saved_tensors
        T, N, tau, step_size, surrogate_scale, spike_update_dt = ctx.params
        grad_spike = grad_spike.contiguous()
        if grad_z_final is None:
            grad_z_final = torch.zeros((N,), device=x_seq_flat.device, dtype=x_seq_flat.dtype)
        else:
            grad_z_final = grad_z_final.contiguous()
        grad_x = torch.empty_like(x_seq_flat)
        grad_z_init = torch.empty((N,), device=x_seq_flat.device, dtype=x_seq_flat.dtype)
        grad_input_scale = torch.zeros((), device=x_seq_flat.device, dtype=x_seq_flat.dtype)
        grad_spike_gain = torch.zeros((), device=x_seq_flat.device, dtype=x_seq_flat.dtype)
        grad_spike_bias = torch.zeros((), device=x_seq_flat.device, dtype=x_seq_flat.dtype)

        threads = 256
        blocks = (N + threads - 1) // threads
        _get_backward_kernel()(
            (blocks,), (threads,),
            (
                _to_cupy(grad_spike),
                _to_cupy(grad_z_final),
                _to_cupy(x_seq_flat),
                _to_cupy(z_pre),
                _to_cupy(spike_input),
                _to_cupy(grad_x),
                _to_cupy(grad_z_init),
                _to_cupy(grad_input_scale),
                _to_cupy(grad_spike_gain),
                _to_cupy(grad_spike_bias),
                np.int32(T),
                np.int32(N),
                np.float32(float(input_scale.detach().cpu())),
                np.float32(float(spike_gain.detach().cpu())),
                np.float32(tau),
                np.float32(step_size),
                np.float32(surrogate_scale),
                np.float32(spike_update_dt),
            ),
        )
        cupy.cuda.get_current_stream().synchronize()
        return (
            grad_x,
            grad_z_init,
            grad_input_scale,
            grad_spike_gain,
            grad_spike_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def lif_like_euler_cupy(x_seq_flat, z_init_flat, input_scale, spike_gain, spike_bias,
                        tau, threshold, step_size, reset_strength, surrogate_scale, spike_update_dt):
    return DirectLIFLikeEulerCuPyFunction.apply(
        x_seq_flat,
        z_init_flat,
        input_scale,
        spike_gain,
        spike_bias,
        tau,
        threshold,
        step_size,
        reset_strength,
        surrogate_scale,
        spike_update_dt,
    )
