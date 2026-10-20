"""
CuPy kernels for LIFNeuronFDE - 完整版本
支持 GL、TRAP、PREDICTOR 三种方法
高精度实现，与 PyTorch 完全对齐
"""

import torch
import math

try:
    import cupy
    from cupy import cuda
    CUPY_AVAILABLE = True
except ImportError:
    cupy = None
    cuda = None
    CUPY_AVAILABLE = False


# =========================================================================================
# GL 方法 (Grünwald-Letnikov)
# =========================================================================================

gl_forward_kernel_code = r'''
extern "C" __global__
void lif_fde_gl_forward_kernel(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ c_rev,
    float* __restrict__ spike_seq,
    float* __restrict__ v_seq,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float v_reset
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    float v_mem = v_init[neuron_idx];
    
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_no_reset = (-v_mem + current_input) / tau;
        
        float hist = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            float hist_sum = 0.0f;
            float hist_c = 0.0f;
            
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                const float y = c_rev[weight_idx] * v_seq[hist_idx * N + neuron_idx] - hist_c;
                const float t = hist_sum + y;
                hist_c = (t - hist_sum) - y;
                hist_sum = t;
            }
            hist = hist_sum;
        }
        
        // Standard GL method: v_post_charge = dv_no_reset - hist
        const float v_post_charge = dv_no_reset - hist;
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq[k % T * N + neuron_idx] = v_mem_new;
        spike_seq[k * N + neuron_idx] = spike;
        v_mem = v_mem_new;
    }
}
'''

gl_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_gl_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ grad_v_seq,
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ c_rev,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float surrogate_scale
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    const float pi_half = 1.57079632679489661923f;
    
    float v_seq_local[256];
    float spike_seq_local[256];
    
    float v_mem = v_init[neuron_idx];
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_no_reset = (-v_mem + current_input) / tau;
        
        float hist = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            float hist_sum = 0.0f;
            float hist_c = 0.0f;
            
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                const float y = c_rev[weight_idx] * v_seq_local[hist_idx] - hist_c;
                const float t = hist_sum + y;
                hist_c = (t - hist_sum) - y;
                hist_sum = t;
            }
            hist = hist_sum;
        }
        
        // Standard GL method: v_post_charge = dv_no_reset - hist
        const float v_post_charge = dv_no_reset - hist;
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq_local[k % T] = v_mem_new;
        spike_seq_local[k] = spike;
        v_mem = v_mem_new;
    }
    
    float grad_v = 0.0f;
    
    for (int t = T - 1; t >= 0; t--) {
        grad_v += grad_v_seq[t * N + neuron_idx];
        
        const float v_mem_new = v_seq_local[t % T];
        const float spike = spike_seq_local[t];
        const float v_post_charge = v_mem_new + spike * v_threshold;
        
        const float v_scaled = v_post_charge - v_threshold;
        const float pi_half_scale_input = pi_half * surrogate_scale * v_scaled;
        const float denom = 1.0f + pi_half_scale_input * pi_half_scale_input;
        const float surrogate_grad = surrogate_scale / (2.0f * denom);
        
        grad_v += grad_spike_seq[t * N + neuron_idx] * surrogate_grad;
        
        // Standard GL method: no beta_factor in gradient
        const float grad_dv_no_reset = grad_v;
        grad_x_seq[t * N + neuron_idx] = grad_dv_no_reset / tau;
        
        const float grad_v_mem = -grad_dv_no_reset / tau;
        
        if (t > 0) {
            grad_v = grad_v_mem;
        } else {
            grad_v_init[neuron_idx] = grad_v_mem;
        }
    }
}
'''


# =========================================================================================
# GL1 方法 (Grünwald-Letnikov with h_beta scaling factor)
# =========================================================================================

gl1_forward_kernel_code = r'''
extern "C" __global__
void lif_fde_gl1_forward_kernel(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ c_rev,
    float* __restrict__ spike_seq,
    float* __restrict__ v_seq,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float v_reset,
    const float h_beta
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    float v_mem = v_init[neuron_idx];
    
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_no_reset = (-v_mem + current_input) / tau;
        
        float hist = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            float hist_sum = 0.0f;
            float hist_c = 0.0f;
            
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                const float y = c_rev[weight_idx] * v_seq[hist_idx * N + neuron_idx] - hist_c;
                const float t = hist_sum + y;
                hist_c = (t - hist_sum) - y;
                hist_sum = t;
            }
            hist = hist_sum;
        }
        
        // GL1 method: v_post_charge = h_beta * dv_no_reset - hist
        const float v_post_charge = h_beta * dv_no_reset - hist;
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq[k % T * N + neuron_idx] = v_mem_new;
        spike_seq[k * N + neuron_idx] = spike;
        v_mem = v_mem_new;
    }
}
'''

gl1_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_gl1_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ grad_v_seq,
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ c_rev,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float surrogate_scale,
    const float h_beta
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    const float pi_half = 1.57079632679489661923f;
    
    float v_seq_local[256];
    float spike_seq_local[256];
    
    float v_mem = v_init[neuron_idx];
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_no_reset = (-v_mem + current_input) / tau;
        
        float hist = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            float hist_sum = 0.0f;
            float hist_c = 0.0f;
            
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                const float y = c_rev[weight_idx] * v_seq_local[hist_idx] - hist_c;
                const float t = hist_sum + y;
                hist_c = (t - hist_sum) - y;
                hist_sum = t;
            }
            hist = hist_sum;
        }
        
        // GL1 method: v_post_charge = h_beta * dv_no_reset - hist
        const float v_post_charge = h_beta * dv_no_reset - hist;
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq_local[k % T] = v_mem_new;
        spike_seq_local[k] = spike;
        v_mem = v_mem_new;
    }
    
    float grad_v = 0.0f;
    
    for (int t = T - 1; t >= 0; t--) {
        grad_v += grad_v_seq[t * N + neuron_idx];
        
        const float v_mem_new = v_seq_local[t % T];
        const float spike = spike_seq_local[t];
        const float v_post_charge = v_mem_new + spike * v_threshold;
        
        const float v_scaled = v_post_charge - v_threshold;
        const float pi_half_scale_input = pi_half * surrogate_scale * v_scaled;
        const float denom = 1.0f + pi_half_scale_input * pi_half_scale_input;
        const float surrogate_grad = surrogate_scale / (2.0f * denom);
        
        grad_v += grad_spike_seq[t * N + neuron_idx] * surrogate_grad;
        
        // GL1 method: grad propagation with h_beta factor
        const float grad_dv_no_reset = h_beta * grad_v;
        grad_x_seq[t * N + neuron_idx] = grad_dv_no_reset / tau;
        
        float grad_v_mem = -grad_dv_no_reset / tau;
        
        // Propagate through history (note: we accumulate grad locally, not write back to grad_v_seq)
        if (t > 0) {
            int history_len = (t < T) ? t : T;
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                // Note: The gradient for history is implicit, we only need local gradient
                grad_v_mem -= c_rev[weight_idx] * grad_v;
            }
        }
        
        grad_v = grad_v_mem;
    }
    
    grad_v_init[neuron_idx] = grad_v;
}
'''


# =========================================================================================
# TRAP 方法 (L1-2 Trapezoidal)
# =========================================================================================

trap_forward_kernel_code = r'''
extern "C" __global__
void lif_fde_trap_forward_kernel(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ trap_weights,
    float* __restrict__ spike_seq,
    float* __restrict__ v_seq,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float v_reset
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    float v_mem = v_init[neuron_idx];
    const float trap_factor = 1.5f * beta + 0.5f;  // trap_factor = 1.5 * beta + 0.5
    const float h_alpha = 1.0f;  // pow(1.0, beta) = 1.0
    
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_no_reset = (-v_mem + current_input) / tau;
        
        // 计算历史贡献（与GL类似，但使用trap权重）
        float hist = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            float hist_sum = 0.0f;
            float hist_c = 0.0f;
            
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                const float y = trap_weights[weight_idx] * v_seq[hist_idx * N + neuron_idx] - hist_c;
                const float t = hist_sum + y;
                hist_c = (t - hist_sum) - y;
                hist_sum = t;
            }
            hist = hist_sum;
        }
        
        // v_post_charge = v_mem + h_alpha * (trap_factor * dv_no_reset - hist)
        const float v_post_charge = v_mem + h_alpha * (trap_factor * dv_no_reset - hist);
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq[k % T * N + neuron_idx] = v_mem_new;
        spike_seq[k * N + neuron_idx] = spike;
        v_mem = v_mem_new;
    }
}
'''

trap_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_trap_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ grad_v_seq,
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ trap_weights,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float surrogate_scale
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    const float trap_factor = 1.5f * beta + 0.5f;
    const float h_alpha = 1.0f;
    const float pi_half = 1.57079632679489661923f;
    
    // 重建前向传播
    float v_seq_local[256];
    float spike_seq_local[256];
    
    float v_mem = v_init[neuron_idx];
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_no_reset = (-v_mem + current_input) / tau;
        
        float hist = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            float hist_sum = 0.0f;
            float hist_c = 0.0f;
            
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                const float y = trap_weights[weight_idx] * v_seq_local[hist_idx] - hist_c;
                const float t = hist_sum + y;
                hist_c = (t - hist_sum) - y;
                hist_sum = t;
            }
            hist = hist_sum;
        }
        
        const float v_post_charge = v_mem + h_alpha * (trap_factor * dv_no_reset - hist);
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq_local[k % T] = v_mem_new;
        spike_seq_local[k] = spike;
        v_mem = v_mem_new;
    }
    
    // 反向传播
    float grad_v = 0.0f;
    
    for (int t = T - 1; t >= 0; t--) {
        grad_v += grad_v_seq[t * N + neuron_idx];
        
        const float v_mem_new = v_seq_local[t % T];
        const float spike = spike_seq_local[t];
        const float v_post_charge = v_mem_new + spike * v_threshold;
        
        const float v_scaled = v_post_charge - v_threshold;
        const float pi_half_scale_input = pi_half * surrogate_scale * v_scaled;
        const float denom = 1.0f + pi_half_scale_input * pi_half_scale_input;
        const float surrogate_grad = surrogate_scale / (2.0f * denom);
        
        grad_v += grad_spike_seq[t * N + neuron_idx] * surrogate_grad;
        
        // grad通过 v_post_charge = v_mem + h_alpha * (trap_factor * dv_no_reset - hist)
        const float grad_scaled_dv = grad_v * h_alpha * trap_factor;
        grad_x_seq[t * N + neuron_idx] = grad_scaled_dv / tau;
        
        const float grad_v_mem = grad_v - grad_scaled_dv / tau;
        
        if (t > 0) {
            grad_v = grad_v_mem;
        } else {
            grad_v_init[neuron_idx] = grad_v_mem;
        }
    }
}
'''


# =========================================================================================
# PREDICTOR 方法
# =========================================================================================

predictor_forward_kernel_code = r'''
extern "C" __global__
void lif_fde_predictor_forward_kernel(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ predictor_weights,
    float* __restrict__ spike_seq,
    float* __restrict__ v_seq,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float v_reset,
    const float h_beta,
    const float gamma_beta_plus_1
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    float v_mem = v_init[neuron_idx];
    
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_current = (-v_mem + current_input) / tau;
        
        // 计算历史积分（使用历史导数）
        float hist_integral = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            float hist_sum = 0.0f;
            float hist_c = 0.0f;
            
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const float hist_v = v_seq[hist_idx * N + neuron_idx];
                const float hist_dv = -hist_v / tau;  // 简化：假设历史输入为0
                
                const int weight_idx = T - history_len + i;
                const float y = predictor_weights[weight_idx] * hist_dv - hist_c;
                const float t = hist_sum + y;
                hist_c = (t - hist_sum) - y;
                hist_sum = t;
            }
            hist_integral = hist_sum;
        }
        
        // w_current = predictor_weights[T] (最后一个元素是w_0)
        const float w_current = predictor_weights[T];
        
        // v_post_charge = v_mem + (h_beta / gamma_beta_plus_1) * (w_current * dv_current + hist_integral)
        const float v_post_charge = v_mem + (h_beta / gamma_beta_plus_1) * (w_current * dv_current + hist_integral);
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq[k % T * N + neuron_idx] = v_mem_new;
        spike_seq[k * N + neuron_idx] = spike;
        v_mem = v_mem_new;
    }
}
'''

predictor_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_predictor_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ grad_v_seq,
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ predictor_weights,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float surrogate_scale,
    const float h_beta,
    const float gamma_beta_plus_1
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    const float pi_half = 1.57079632679489661923f;
    
    // 重建前向传播
    float v_seq_local[256];
    float spike_seq_local[256];
    
    float v_mem = v_init[neuron_idx];
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_current = (-v_mem + current_input) / tau;
        
        float hist_integral = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            float hist_sum = 0.0f;
            float hist_c = 0.0f;
            
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const float hist_v = v_seq_local[hist_idx];
                const float hist_dv = -hist_v / tau;
                
                const int weight_idx = T - history_len + i;
                const float y = predictor_weights[weight_idx] * hist_dv - hist_c;
                const float t = hist_sum + y;
                hist_c = (t - hist_sum) - y;
                hist_sum = t;
            }
            hist_integral = hist_sum;
        }
        
        const float w_current = predictor_weights[T];
        const float v_post_charge = v_mem + (h_beta / gamma_beta_plus_1) * (w_current * dv_current + hist_integral);
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq_local[k % T] = v_mem_new;
        spike_seq_local[k] = spike;
        v_mem = v_mem_new;
    }
    
    // 反向传播
    float grad_v = 0.0f;
    
    for (int t = T - 1; t >= 0; t--) {
        grad_v += grad_v_seq[t * N + neuron_idx];
        
        const float v_mem_new = v_seq_local[t % T];
        const float spike = spike_seq_local[t];
        const float v_post_charge = v_mem_new + spike * v_threshold;
        
        const float v_scaled = v_post_charge - v_threshold;
        const float pi_half_scale_input = pi_half * surrogate_scale * v_scaled;
        const float denom = 1.0f + pi_half_scale_input * pi_half_scale_input;
        const float surrogate_grad = surrogate_scale / (2.0f * denom);
        
        grad_v += grad_spike_seq[t * N + neuron_idx] * surrogate_grad;
        
        // grad通过 v_post_charge = v_mem + (h_beta/gamma) * (w_current * dv_current + hist_integral)
        const float w_current = predictor_weights[T];
        const float scale_factor = h_beta / gamma_beta_plus_1;
        const float grad_dv_current = grad_v * scale_factor * w_current;
        
        grad_x_seq[t * N + neuron_idx] = grad_dv_current / tau;
        
        const float grad_v_mem = grad_v - grad_dv_current / tau;
        
        if (t > 0) {
            grad_v = grad_v_mem;
        } else {
            grad_v_init[neuron_idx] = grad_v_mem;
        }
    }
}
'''


# =========================================================================================
# AutoGrad Functions
# =========================================================================================

class LIFFDEGLFunction(torch.autograd.Function):
    """GL 方法的 CuPy Function"""
    forward_kernel = None
    backward_kernel = None
    
    @staticmethod
    def get_kernels():
        if LIFFDEGLFunction.forward_kernel is None:
            LIFFDEGLFunction.forward_kernel = cupy.RawKernel(gl_forward_kernel_code, 'lif_fde_gl_forward_kernel')
        if LIFFDEGLFunction.backward_kernel is None:
            LIFFDEGLFunction.backward_kernel = cupy.RawKernel(gl_backward_kernel_code, 'lif_fde_gl_backward_kernel')
        return LIFFDEGLFunction.forward_kernel, LIFFDEGLFunction.backward_kernel
    
    @staticmethod
    def forward(ctx, x_seq, v_init, c_rev, tau, beta, v_threshold,
                v_reset, surrogate_grad_scale):
        assert CUPY_AVAILABLE and x_seq.is_cuda
        T, N = x_seq.shape
        
        forward_kernel, _ = LIFFDEGLFunction.get_kernels()
        
        x_seq_f32 = x_seq.float().contiguous()
        v_init_f32 = v_init.float().contiguous()
        c_rev_f32 = c_rev.float().contiguous()
        
        spike_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        v_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        
        import cupy
        x_seq_cp = cupy.from_dlpack(x_seq_f32.detach())
        v_init_cp = cupy.from_dlpack(v_init_f32.detach())
        c_rev_cp = cupy.from_dlpack(c_rev_f32.detach())
        spike_seq_cp = cupy.from_dlpack(spike_seq)
        v_seq_cp = cupy.from_dlpack(v_seq)
        
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        forward_kernel(
            (blocks,), (threads_per_block,),
            (x_seq_cp, v_init_cp, c_rev_cp, spike_seq_cp, v_seq_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(tau), cupy.float32(beta),
             cupy.float32(v_threshold), cupy.float32(v_reset))
        )
        
        ctx.save_for_backward(x_seq_f32, v_init_f32, c_rev_f32)
        ctx.tau = tau
        ctx.beta = beta
        ctx.v_threshold = v_threshold
        ctx.surrogate_grad_scale = surrogate_grad_scale
        ctx.T = T
        ctx.N = N
        
        return spike_seq, v_seq
    
    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        x_seq, v_init, c_rev = ctx.saved_tensors
        T, N = ctx.T, ctx.N
        
        _, backward_kernel = LIFFDEGLFunction.get_kernels()
        
        grad_spike_seq_f32 = grad_spike_seq.float().contiguous()
        grad_v_seq_f32 = grad_v_seq.float().contiguous()
        
        grad_x_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        grad_v_init = torch.zeros(N, dtype=torch.float32, device=x_seq.device)
        
        import cupy
        grad_spike_seq_cp = cupy.from_dlpack(grad_spike_seq_f32.detach())
        grad_v_seq_cp = cupy.from_dlpack(grad_v_seq_f32.detach())
        x_seq_cp = cupy.from_dlpack(x_seq.detach())
        v_init_cp = cupy.from_dlpack(v_init.detach())
        c_rev_cp = cupy.from_dlpack(c_rev.detach())
        grad_x_seq_cp = cupy.from_dlpack(grad_x_seq)
        grad_v_init_cp = cupy.from_dlpack(grad_v_init)
        
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        backward_kernel(
            (blocks,), (threads_per_block,),
            (grad_spike_seq_cp, grad_v_seq_cp, x_seq_cp, v_init_cp, c_rev_cp,
             grad_x_seq_cp, grad_v_init_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(ctx.tau), cupy.float32(ctx.beta),
             cupy.float32(ctx.v_threshold), cupy.float32(ctx.surrogate_grad_scale))
        )
        
        return grad_x_seq, grad_v_init, None, None, None, None, None, None


class LIFFDEGL1Function(torch.autograd.Function):
    """GL1 方法的 CuPy Function (with h_beta scaling)"""
    forward_kernel = None
    backward_kernel = None
    
    @staticmethod
    def get_kernels():
        if LIFFDEGL1Function.forward_kernel is None:
            LIFFDEGL1Function.forward_kernel = cupy.RawKernel(gl1_forward_kernel_code, 'lif_fde_gl1_forward_kernel')
        if LIFFDEGL1Function.backward_kernel is None:
            LIFFDEGL1Function.backward_kernel = cupy.RawKernel(gl1_backward_kernel_code, 'lif_fde_gl1_backward_kernel')
        return LIFFDEGL1Function.forward_kernel, LIFFDEGL1Function.backward_kernel
    
    @staticmethod
    def forward(ctx, x_seq, v_init, c_rev, tau, beta, step_size,
                v_threshold, v_reset, surrogate_grad_scale):
        assert CUPY_AVAILABLE and x_seq.is_cuda
        T, N = x_seq.shape
        
        beta_value = beta if isinstance(beta, float) else beta.item()
        h_beta = math.pow(float(step_size), beta_value)
        
        forward_kernel, _ = LIFFDEGL1Function.get_kernels()
        
        x_seq_f32 = x_seq.float().contiguous()
        v_init_f32 = v_init.float().contiguous()
        c_rev_f32 = c_rev.float().contiguous()
        
        spike_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        v_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        
        import cupy
        x_seq_cp = cupy.from_dlpack(x_seq_f32.detach())
        v_init_cp = cupy.from_dlpack(v_init_f32.detach())
        c_rev_cp = cupy.from_dlpack(c_rev_f32.detach())
        spike_seq_cp = cupy.from_dlpack(spike_seq)
        v_seq_cp = cupy.from_dlpack(v_seq)
        
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        forward_kernel(
            (blocks,), (threads_per_block,),
            (x_seq_cp, v_init_cp, c_rev_cp, spike_seq_cp, v_seq_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(tau), cupy.float32(beta),
             cupy.float32(v_threshold), cupy.float32(v_reset), cupy.float32(h_beta))
        )
        
        ctx.save_for_backward(x_seq_f32, v_init_f32, c_rev_f32)
        ctx.tau = tau
        ctx.beta = beta
        ctx.v_threshold = v_threshold
        ctx.surrogate_grad_scale = surrogate_grad_scale
        ctx.h_beta = h_beta
        ctx.T = T
        ctx.N = N
        
        return spike_seq, v_seq
    
    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        x_seq_f32, v_init_f32, c_rev_f32 = ctx.saved_tensors
        T = ctx.T
        N = ctx.N
        
        _, backward_kernel = LIFFDEGL1Function.get_kernels()
        
        grad_x_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq_f32.device)
        grad_v_init = torch.zeros(N, dtype=torch.float32, device=x_seq_f32.device)
        
        import cupy
        grad_spike_seq_cp = cupy.from_dlpack(grad_spike_seq.float().contiguous())
        grad_v_seq_cp = cupy.from_dlpack(grad_v_seq.float().contiguous())
        x_seq_cp = cupy.from_dlpack(x_seq_f32.detach())
        v_init_cp = cupy.from_dlpack(v_init_f32.detach())
        c_rev_cp = cupy.from_dlpack(c_rev_f32.detach())
        grad_x_seq_cp = cupy.from_dlpack(grad_x_seq)
        grad_v_init_cp = cupy.from_dlpack(grad_v_init)
        
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        backward_kernel(
            (blocks,), (threads_per_block,),
            (grad_spike_seq_cp, grad_v_seq_cp, x_seq_cp, v_init_cp, c_rev_cp,
             grad_x_seq_cp, grad_v_init_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(ctx.tau), cupy.float32(ctx.beta),
             cupy.float32(ctx.v_threshold), cupy.float32(ctx.surrogate_grad_scale),
             cupy.float32(ctx.h_beta))
        )
        
        return grad_x_seq, grad_v_init, None, None, None, None, None, None, None


class LIFFDETRAPFunction(torch.autograd.Function):
    """TRAP 方法的 CuPy Function"""
    forward_kernel = None
    backward_kernel = None
    
    @staticmethod
    def get_kernels():
        if LIFFDETRAPFunction.forward_kernel is None:
            LIFFDETRAPFunction.forward_kernel = cupy.RawKernel(trap_forward_kernel_code, 'lif_fde_trap_forward_kernel')
        if LIFFDETRAPFunction.backward_kernel is None:
            LIFFDETRAPFunction.backward_kernel = cupy.RawKernel(trap_backward_kernel_code, 'lif_fde_trap_backward_kernel')
        return LIFFDETRAPFunction.forward_kernel, LIFFDETRAPFunction.backward_kernel
    
    @staticmethod
    def forward(ctx, x_seq, v_init, trap_weights, tau, beta, v_threshold, v_reset, surrogate_grad_scale):
        assert CUPY_AVAILABLE and x_seq.is_cuda
        T, N = x_seq.shape
        
        forward_kernel, _ = LIFFDETRAPFunction.get_kernels()
        
        x_seq_f32 = x_seq.float().contiguous()
        v_init_f32 = v_init.float().contiguous()
        trap_weights_f32 = trap_weights.float().contiguous()
        
        spike_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        v_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        
        import cupy
        x_seq_cp = cupy.from_dlpack(x_seq_f32.detach())
        v_init_cp = cupy.from_dlpack(v_init_f32.detach())
        trap_weights_cp = cupy.from_dlpack(trap_weights_f32.detach())
        spike_seq_cp = cupy.from_dlpack(spike_seq)
        v_seq_cp = cupy.from_dlpack(v_seq)
        
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        forward_kernel(
            (blocks,), (threads_per_block,),
            (x_seq_cp, v_init_cp, trap_weights_cp, spike_seq_cp, v_seq_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(tau), cupy.float32(beta),
             cupy.float32(v_threshold), cupy.float32(v_reset))
        )
        
        ctx.save_for_backward(x_seq_f32, v_init_f32, trap_weights_f32)
        ctx.tau = tau
        ctx.beta = beta
        ctx.v_threshold = v_threshold
        ctx.surrogate_grad_scale = surrogate_grad_scale
        ctx.T = T
        ctx.N = N
        
        return spike_seq, v_seq
    
    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        x_seq, v_init, trap_weights = ctx.saved_tensors
        T, N = ctx.T, ctx.N
        
        _, backward_kernel = LIFFDETRAPFunction.get_kernels()
        
        grad_spike_seq_f32 = grad_spike_seq.float().contiguous()
        grad_v_seq_f32 = grad_v_seq.float().contiguous()
        
        grad_x_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        grad_v_init = torch.zeros(N, dtype=torch.float32, device=x_seq.device)
        
        import cupy
        grad_spike_seq_cp = cupy.from_dlpack(grad_spike_seq_f32.detach())
        grad_v_seq_cp = cupy.from_dlpack(grad_v_seq_f32.detach())
        x_seq_cp = cupy.from_dlpack(x_seq.detach())
        v_init_cp = cupy.from_dlpack(v_init.detach())
        trap_weights_cp = cupy.from_dlpack(trap_weights.detach())
        grad_x_seq_cp = cupy.from_dlpack(grad_x_seq)
        grad_v_init_cp = cupy.from_dlpack(grad_v_init)
        
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        backward_kernel(
            (blocks,), (threads_per_block,),
            (grad_spike_seq_cp, grad_v_seq_cp, x_seq_cp, v_init_cp, trap_weights_cp,
             grad_x_seq_cp, grad_v_init_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(ctx.tau), cupy.float32(ctx.beta),
             cupy.float32(ctx.v_threshold), cupy.float32(ctx.surrogate_grad_scale))
        )
        
        return grad_x_seq, grad_v_init, None, None, None, None, None, None


class LIFFDEPREDICTORFunction(torch.autograd.Function):
    """PREDICTOR 方法的 CuPy Function"""
    forward_kernel = None
    backward_kernel = None
    
    @staticmethod
    def get_kernels():
        if LIFFDEPREDICTORFunction.forward_kernel is None:
            LIFFDEPREDICTORFunction.forward_kernel = cupy.RawKernel(predictor_forward_kernel_code, 'lif_fde_predictor_forward_kernel')
        if LIFFDEPREDICTORFunction.backward_kernel is None:
            LIFFDEPREDICTORFunction.backward_kernel = cupy.RawKernel(predictor_backward_kernel_code, 'lif_fde_predictor_backward_kernel')
        return LIFFDEPREDICTORFunction.forward_kernel, LIFFDEPREDICTORFunction.backward_kernel
    
    @staticmethod
    def forward(ctx, x_seq, v_init, predictor_weights, tau, beta, v_threshold, v_reset, surrogate_grad_scale, h_beta, gamma_beta_plus_1):
        assert CUPY_AVAILABLE and x_seq.is_cuda
        T, N = x_seq.shape
        
        forward_kernel, _ = LIFFDEPREDICTORFunction.get_kernels()
        
        x_seq_f32 = x_seq.float().contiguous()
        v_init_f32 = v_init.float().contiguous()
        predictor_weights_f32 = predictor_weights.float().contiguous()
        
        spike_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        v_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        
        import cupy
        x_seq_cp = cupy.from_dlpack(x_seq_f32.detach())
        v_init_cp = cupy.from_dlpack(v_init_f32.detach())
        predictor_weights_cp = cupy.from_dlpack(predictor_weights_f32.detach())
        spike_seq_cp = cupy.from_dlpack(spike_seq)
        v_seq_cp = cupy.from_dlpack(v_seq)
        
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        forward_kernel(
            (blocks,), (threads_per_block,),
            (x_seq_cp, v_init_cp, predictor_weights_cp, spike_seq_cp, v_seq_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(tau), cupy.float32(beta),
             cupy.float32(v_threshold), cupy.float32(v_reset),
             cupy.float32(h_beta), cupy.float32(gamma_beta_plus_1))
        )
        
        ctx.save_for_backward(x_seq_f32, v_init_f32, predictor_weights_f32)
        ctx.tau = tau
        ctx.beta = beta
        ctx.v_threshold = v_threshold
        ctx.surrogate_grad_scale = surrogate_grad_scale
        ctx.h_beta = h_beta
        ctx.gamma_beta_plus_1 = gamma_beta_plus_1
        ctx.T = T
        ctx.N = N
        
        return spike_seq, v_seq
    
    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        x_seq, v_init, predictor_weights = ctx.saved_tensors
        T, N = ctx.T, ctx.N
        
        _, backward_kernel = LIFFDEPREDICTORFunction.get_kernels()
        
        grad_spike_seq_f32 = grad_spike_seq.float().contiguous()
        grad_v_seq_f32 = grad_v_seq.float().contiguous()
        
        grad_x_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        grad_v_init = torch.zeros(N, dtype=torch.float32, device=x_seq.device)
        
        import cupy
        grad_spike_seq_cp = cupy.from_dlpack(grad_spike_seq_f32.detach())
        grad_v_seq_cp = cupy.from_dlpack(grad_v_seq_f32.detach())
        x_seq_cp = cupy.from_dlpack(x_seq.detach())
        v_init_cp = cupy.from_dlpack(v_init.detach())
        predictor_weights_cp = cupy.from_dlpack(predictor_weights.detach())
        grad_x_seq_cp = cupy.from_dlpack(grad_x_seq)
        grad_v_init_cp = cupy.from_dlpack(grad_v_init)
        
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        backward_kernel(
            (blocks,), (threads_per_block,),
            (grad_spike_seq_cp, grad_v_seq_cp, x_seq_cp, v_init_cp, predictor_weights_cp,
             grad_x_seq_cp, grad_v_init_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(ctx.tau), cupy.float32(ctx.beta),
             cupy.float32(ctx.v_threshold), cupy.float32(ctx.surrogate_grad_scale),
             cupy.float32(ctx.h_beta), cupy.float32(ctx.gamma_beta_plus_1))
        )
        
        return grad_x_seq, grad_v_init, None, None, None, None, None, None, None, None


# =========================================================================================
# Shared per-time variable-order predictor
# =========================================================================================

vo_predictor_forward_kernel_code = r'''
extern "C" __global__
void lif_fde_vo_predictor_forward_kernel(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ scale,
    const float* __restrict__ weights,
    float* __restrict__ spike_seq,
    float* __restrict__ v_pre_seq,
    float* __restrict__ drift_seq,
    const int T,
    const int N,
    const int B,
    const int per_batch,
    const float tau,
    const float threshold,
    const float residual_mix
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float v = v_init[n];
    const int feature_count = N / B;
    const int b = n / feature_count;
    for (int t = 0; t < T; ++t) {
        const float drift = (-v + x_seq[t * N + n]) / tau;
        float hist = 0.0f;
        for (int i = 0; i < t; ++i) {
            const int wi = per_batch ? ((t * B + b) * T + i) : (t * T + i);
            hist += weights[wi] * drift_seq[i * N + n];
        }
        const float s = scale[per_batch ? (t * B + b) : t];
        const float v_pre = v_init[n] + s * (drift + hist) + residual_mix * (v - v_init[n]);
        const float spike = v_pre >= threshold ? 1.0f : 0.0f;
        v_pre_seq[t * N + n] = v_pre;
        drift_seq[t * N + n] = drift;
        spike_seq[t * N + n] = spike;
        v = v_pre - spike * threshold;
    }
}
'''

vo_predictor_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_vo_predictor_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ scale,
    const float* __restrict__ weights,
    const float* __restrict__ v_pre_seq,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    float* __restrict__ grad_pre_seq,
    const int T,
    const int N,
    const int B,
    const int per_batch,
    const float tau,
    const float threshold,
    const float surrogate_scale,
    const float residual_mix
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float pi_half = 1.57079632679489661923f;
    const int feature_count = N / B;
    const int b = n / feature_count;
    float grad_v_next = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        const float z = v_pre_seq[t * N + n] - threshold;
        const float u = pi_half * surrogate_scale * z;
        const float surrogate_grad = surrogate_scale / (2.0f * (1.0f + u * u));
        const float grad_pre = grad_v_next + grad_spike_seq[t * N + n] * surrogate_grad;

        grad_pre_seq[t * N + n] = grad_pre;
        const float s = scale[per_batch ? (t * B + b) : t];
        grad_x_seq[t * N + n] = grad_pre * s / tau;
        grad_v_next = grad_pre * (residual_mix - s / tau);
    }
    grad_v_init[n] = grad_v_next;
}
'''


class LIFFDEVOPREDICTORFunction(torch.autograd.Function):
    """CuPy forward/backward for a shared differentiable beta schedule."""
    forward_kernel = None
    backward_kernel = None

    @staticmethod
    def get_kernels():
        if LIFFDEVOPREDICTORFunction.forward_kernel is None:
            LIFFDEVOPREDICTORFunction.forward_kernel = cupy.RawKernel(
                vo_predictor_forward_kernel_code,
                'lif_fde_vo_predictor_forward_kernel',
            )
        if LIFFDEVOPREDICTORFunction.backward_kernel is None:
            LIFFDEVOPREDICTORFunction.backward_kernel = cupy.RawKernel(
                vo_predictor_backward_kernel_code,
                'lif_fde_vo_predictor_backward_kernel',
            )
        return (
            LIFFDEVOPREDICTORFunction.forward_kernel,
            LIFFDEVOPREDICTORFunction.backward_kernel,
        )

    @staticmethod
    def forward(ctx, x_seq, v_init, scale, weights, tau, threshold,
                surrogate_grad_scale, residual_mix, batch_size):
        if not CUPY_AVAILABLE or not x_seq.is_cuda:
            raise RuntimeError('LIFFDEVOPREDICTORFunction requires CUDA and CuPy')
        original_dtype = x_seq.dtype
        x_f32 = x_seq.float().contiguous()
        v_f32 = v_init.float().contiguous()
        scale_f32 = scale.float().contiguous()
        weights_f32 = weights.float().contiguous()
        T, N = x_f32.shape
        B = int(batch_size)
        per_batch = int(scale_f32.dim() == 2)
        spike_seq = torch.empty_like(x_f32)
        v_pre_seq = torch.empty_like(x_f32)
        drift_seq = torch.empty_like(x_f32)
        forward_kernel, _ = LIFFDEVOPREDICTORFunction.get_kernels()
        arrays = [x_f32.detach(), v_f32.detach(), scale_f32.detach(),
                  weights_f32.detach(), spike_seq, v_pre_seq, drift_seq]
        cp = [cupy.from_dlpack(tensor) for tensor in arrays]
        threads = 256
        blocks = (N + threads - 1) // threads
        forward_kernel(
            (blocks,), (threads,),
            (*cp, cupy.int32(T), cupy.int32(N), cupy.int32(B), cupy.int32(per_batch), cupy.float32(tau),
             cupy.float32(threshold), cupy.float32(residual_mix)),
        )
        ctx.save_for_backward(scale_f32, weights_f32, v_pre_seq, drift_seq)
        ctx.input_shape = x_f32.shape
        ctx.tau = float(tau)
        ctx.threshold = float(threshold)
        ctx.surrogate_grad_scale = float(surrogate_grad_scale)
        ctx.residual_mix = float(residual_mix)
        ctx.original_dtype = original_dtype
        ctx.batch_size = B
        ctx.per_batch = per_batch
        return spike_seq.to(dtype=original_dtype)

    @staticmethod
    def backward(ctx, grad_spike_seq):
        scale, weights, v_pre_seq, drift_seq = ctx.saved_tensors
        T, N = ctx.input_shape
        grad_spike = grad_spike_seq.float().contiguous()
        grad_x = torch.zeros(ctx.input_shape, device=grad_spike.device, dtype=torch.float32)
        grad_v = torch.zeros(N, device=grad_spike.device, dtype=torch.float32)
        grad_pre = torch.empty(ctx.input_shape, device=grad_spike.device, dtype=torch.float32)
        _, backward_kernel = LIFFDEVOPREDICTORFunction.get_kernels()
        arrays = [grad_spike.detach(), scale.detach(), v_pre_seq.detach(),
                  grad_x, grad_v, grad_pre]
        cp = [cupy.from_dlpack(tensor) for tensor in arrays]
        threads = 256
        blocks = (N + threads - 1) // threads
        backward_kernel(
            (blocks,), (threads,),
            (*cp, cupy.int32(T), cupy.int32(N), cupy.int32(ctx.batch_size), cupy.int32(ctx.per_batch), cupy.float32(ctx.tau),
             cupy.float32(ctx.threshold), cupy.float32(ctx.surrogate_grad_scale), cupy.float32(ctx.residual_mix)),
        )
        if ctx.per_batch:
            B = ctx.batch_size
            F = N // B
            drift_b = drift_seq.reshape(T, B, F)
            grad_b = grad_pre.reshape(T, B, F)
            hist = torch.einsum('tbi,ibf->tbf', weights, drift_b)
            grad_scale = torch.sum(grad_b * (drift_b + hist), dim=2)
            grad_weights = torch.einsum('tbf,ibf->tbi', grad_b * scale[:, :, None], drift_b)
        else:
            hist = torch.matmul(weights, drift_seq)
            grad_scale = torch.sum(grad_pre * (drift_seq + hist), dim=1)
            grad_weights = torch.matmul(grad_pre * scale[:, None], drift_seq.t())
        return (
            grad_x.to(dtype=ctx.original_dtype), grad_v,
            grad_scale, grad_weights, None, None, None, None, None,
        )


# =========================================================================================
# Fixed-order fractional-attention GL1 (FAGL1)
# =========================================================================================

fa_gl1_forward_kernel_code = r'''
extern "C" __global__
void lif_fde_fa_gl1_forward_kernel(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ weights,
    float* __restrict__ spike_seq,
    float* __restrict__ v_pre_seq,
    float* __restrict__ v_hist_seq,
    const int T, const int N, const int B, const int per_batch,
    const float tau, const float h_beta, const float threshold
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int feature_count = N / B;
    const int b = n / feature_count;
    float v = v_init[n];
    for (int t = 0; t < T; ++t) {
        const float drift = (-v + x_seq[t * N + n]) / tau;
        float hist = 0.0f;
        for (int i = 0; i < t; ++i) {
            const int wi = per_batch ? ((t * B + b) * T + i) : (t * T + i);
            hist += weights[wi] * v_hist_seq[i * N + n];
        }
        const float v_pre = h_beta * drift - hist;
        const float spike = v_pre >= threshold ? 1.0f : 0.0f;
        const float v_reset = v_pre - spike * threshold;
        v_pre_seq[t * N + n] = v_pre;
        spike_seq[t * N + n] = spike;
        v_hist_seq[t * N + n] = v_reset;
        v = v_reset;
    }
}
'''

fa_gl1_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_fa_gl1_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ weights,
    const float* __restrict__ v_pre_seq,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    float* __restrict__ grad_pre_seq,
    const int T, const int N, const int B, const int per_batch,
    const float tau, const float h_beta,
    const float threshold, const float surrogate_scale
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float pi_half = 1.57079632679489661923f;
    const int feature_count = N / B;
    const int b = n / feature_count;

    float grad_v_next = 0.0f;
    // Match the established GL1/VOGL1 surrogate-gradient recurrence.
    for (int t = T - 1; t >= 0; --t) {
        const float z = v_pre_seq[t * N + n] - threshold;
        const float u = pi_half * surrogate_scale * z;
        const float surrogate_grad = surrogate_scale / (2.0f * (1.0f + u * u));
        const float grad_pre = grad_v_next + grad_spike_seq[t * N + n] * surrogate_grad;
        grad_pre_seq[t * N + n] = grad_pre;
        grad_x_seq[t * N + n] = grad_pre * h_beta / tau;
        float history_sum = 0.0f;
        for (int i = 0; i < t; ++i) {
            const int wi = per_batch ? ((t * B + b) * T + i) : (t * T + i);
            history_sum += weights[wi];
        }
        grad_v_next = -grad_pre * (h_beta / tau + history_sum);
    }
    grad_v_init[n] = grad_v_next;
}
'''

fa_gl1_full_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_fa_gl1_full_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ weights,
    const float* __restrict__ v_pre_seq,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    float* __restrict__ grad_pre_seq,
    const int T, const int N, const int B, const int per_batch,
    const float tau, const float h_beta,
    const float threshold, const float surrogate_scale
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float pi_half = 1.57079632679489661923f;
    const int feature_count = N / B;
    const int b = n / feature_count;
    float grad_hist[64];
    for (int t = 0; t < T; ++t) grad_hist[t] = 0.0f;
    float grad_initial = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        const float z = v_pre_seq[t * N + n] - threshold;
        const float u = pi_half * surrogate_scale * z;
        const float surrogate_grad = surrogate_scale / (2.0f * (1.0f + u * u));
        const float grad_pre = grad_hist[t] + grad_spike_seq[t * N + n] * surrogate_grad;
        grad_pre_seq[t * N + n] = grad_pre;
        grad_x_seq[t * N + n] = grad_pre * h_beta / tau;
        const float grad_previous = -grad_pre * h_beta / tau;
        if (t == 0) grad_initial += grad_previous;
        else grad_hist[t - 1] += grad_previous;
        for (int i = 0; i < t; ++i) {
            const int wi = per_batch ? ((t * B + b) * T + i) : (t * T + i);
            grad_hist[i] -= weights[wi] * grad_pre;
        }
    }
    grad_v_init[n] = grad_initial;
}
'''

fa_gl1_weight_grad_kernel_code = r'''
extern "C" __global__
void lif_fde_fa_gl1_weight_grad_kernel(
    const float* __restrict__ grad_pre,
    const float* __restrict__ v_hist,
    float* __restrict__ grad_weights,
    const int T, const int B, const int F
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output = blockIdx.x * 8 + warp;
    if (output >= T * B * T) return;
    const int i = output % T;
    const int row = output / T;
    const int b = row % B;
    const int t = row / B;
    if (i >= t) {
        if (lane == 0) grad_weights[output] = 0.0f;
        return;
    }
    float sum = 0.0f;
    const int lhs = (t * B + b) * F;
    const int rhs = (i * B + b) * F;
    for (int f = lane; f < F; f += 32) {
        sum += grad_pre[lhs + f] * v_hist[rhs + f];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) grad_weights[output] = -sum;
}
'''

fa_gl1_global_weight_grad_kernel_code = r'''
extern "C" __global__
void lif_fde_fa_gl1_global_weight_grad_kernel(
    const float* __restrict__ grad_pre,
    const float* __restrict__ v_hist,
    float* __restrict__ grad_weights,
    const int T, const int N
) {
    const int output = blockIdx.x;
    const int t = output / T;
    const int i = output % T;
    if (i >= t) {
        if (threadIdx.x == 0) grad_weights[output] = 0.0f;
        return;
    }
    float sum = 0.0f;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        sum += grad_pre[t * N + n] * v_hist[i * N + n];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_weights[output] = -partial[0];
}
'''


class LIFFDEFAGL1Function(torch.autograd.Function):
    """CuPy-accelerated fixed-order GL1 with per-sample causal attention rows."""
    forward_kernel = None
    backward_kernel = None
    full_backward_kernel = None
    weight_grad_kernel = None
    global_weight_grad_kernel = None

    @staticmethod
    def get_kernels():
        if LIFFDEFAGL1Function.forward_kernel is None:
            LIFFDEFAGL1Function.forward_kernel = cupy.RawKernel(
                fa_gl1_forward_kernel_code, 'lif_fde_fa_gl1_forward_kernel')
        if LIFFDEFAGL1Function.backward_kernel is None:
            LIFFDEFAGL1Function.backward_kernel = cupy.RawKernel(
                fa_gl1_backward_kernel_code, 'lif_fde_fa_gl1_backward_kernel')
        if LIFFDEFAGL1Function.full_backward_kernel is None:
            LIFFDEFAGL1Function.full_backward_kernel = cupy.RawKernel(
                fa_gl1_full_backward_kernel_code,
                'lif_fde_fa_gl1_full_backward_kernel')
        if LIFFDEFAGL1Function.weight_grad_kernel is None:
            LIFFDEFAGL1Function.weight_grad_kernel = cupy.RawKernel(
                fa_gl1_weight_grad_kernel_code, 'lif_fde_fa_gl1_weight_grad_kernel')
        if LIFFDEFAGL1Function.global_weight_grad_kernel is None:
            LIFFDEFAGL1Function.global_weight_grad_kernel = cupy.RawKernel(
                fa_gl1_global_weight_grad_kernel_code,
                'lif_fde_fa_gl1_global_weight_grad_kernel')
        return (LIFFDEFAGL1Function.forward_kernel,
                LIFFDEFAGL1Function.backward_kernel,
                LIFFDEFAGL1Function.full_backward_kernel,
                LIFFDEFAGL1Function.weight_grad_kernel,
                LIFFDEFAGL1Function.global_weight_grad_kernel)

    @staticmethod
    def forward(ctx, x_seq, v_init, weights, tau, h_beta, threshold,
                surrogate_grad_scale, batch_size, full_history_grad=False):
        if not CUPY_AVAILABLE or not x_seq.is_cuda:
            raise RuntimeError('LIFFDEFAGL1Function requires CUDA and CuPy')
        original_dtype = x_seq.dtype
        x_f32 = x_seq.float().contiguous()
        v_f32 = v_init.float().contiguous()
        weights_f32 = weights.float().contiguous()
        T, N = x_f32.shape
        B = int(batch_size)
        if T > 64:
            raise ValueError('FAGL1 CuPy kernel currently supports at most 64 time steps')
        per_batch = int(weights_f32.dim() == 3)
        expected = (T, B, T) if per_batch else (T, T)
        if weights_f32.shape != expected:
            raise ValueError(
                f'FAGL1 weights must have shape {expected}, got {tuple(weights_f32.shape)}')
        if N % B:
            raise ValueError(f'Flattened neuron count {N} is not divisible by batch size {B}')
        spike_seq = torch.empty_like(x_f32)
        v_pre_seq = torch.empty_like(x_f32)
        v_hist_seq = torch.empty_like(x_f32)
        arrays = [x_f32.detach(), v_f32.detach(), weights_f32.detach(),
                  spike_seq, v_pre_seq, v_hist_seq]
        cp = [cupy.from_dlpack(tensor) for tensor in arrays]
        forward_kernel, _, _, _, _ = LIFFDEFAGL1Function.get_kernels()
        threads = 256
        forward_kernel(((N + threads - 1) // threads,), (threads,),
                       (*cp, cupy.int32(T), cupy.int32(N), cupy.int32(B),
                        cupy.int32(per_batch),
                        cupy.float32(tau), cupy.float32(h_beta),
                        cupy.float32(threshold)))
        ctx.save_for_backward(weights_f32, v_pre_seq, v_hist_seq)
        ctx.input_shape = x_f32.shape
        ctx.tau = float(tau)
        ctx.h_beta = float(h_beta)
        ctx.threshold = float(threshold)
        ctx.surrogate_grad_scale = float(surrogate_grad_scale)
        ctx.original_dtype = original_dtype
        ctx.batch_size = B
        ctx.per_batch = per_batch
        ctx.full_history_grad = bool(full_history_grad)
        return spike_seq.to(dtype=original_dtype)

    @staticmethod
    def backward(ctx, grad_spike_seq):
        weights, v_pre_seq, v_hist_seq = ctx.saved_tensors
        T, N = ctx.input_shape
        B = ctx.batch_size
        F = N // B
        grad_spike = grad_spike_seq.float().contiguous()
        grad_x = torch.empty(ctx.input_shape, device=grad_spike.device, dtype=torch.float32)
        grad_v = torch.empty(N, device=grad_spike.device, dtype=torch.float32)
        grad_pre = torch.empty(ctx.input_shape, device=grad_spike.device, dtype=torch.float32)
        arrays = [grad_spike.detach(), weights.detach(), v_pre_seq.detach(),
                  grad_x, grad_v, grad_pre]
        cp = [cupy.from_dlpack(tensor) for tensor in arrays]
        (_, backward_kernel, full_backward_kernel, weight_grad_kernel,
         global_weight_grad_kernel) = (
            LIFFDEFAGL1Function.get_kernels())
        threads = 256
        selected_backward = full_backward_kernel if ctx.full_history_grad else backward_kernel
        selected_backward(((N + threads - 1) // threads,), (threads,),
                          (*cp, cupy.int32(T), cupy.int32(N), cupy.int32(B),
                           cupy.int32(ctx.per_batch),
                           cupy.float32(ctx.tau), cupy.float32(ctx.h_beta),
                           cupy.float32(ctx.threshold),
                           cupy.float32(ctx.surrogate_grad_scale)))
        grad_weights = torch.empty_like(weights)
        grad_cp = cupy.from_dlpack(grad_weights)
        hist_cp = cupy.from_dlpack(v_hist_seq.detach())
        pre_cp = cupy.from_dlpack(grad_pre.detach())
        if ctx.per_batch:
            outputs = T * B * T
            weight_grad_kernel(((outputs + 7) // 8,), (threads,),
                               (pre_cp, hist_cp, grad_cp,
                                cupy.int32(T), cupy.int32(B), cupy.int32(F)))
        else:
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                # Blackwell is dramatically faster for this narrow reduction in TF32.
                torch.backends.cuda.matmul.allow_tf32 = True
                grad_weights = -torch.matmul(grad_pre, v_hist_seq.t())
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        return (grad_x.to(dtype=ctx.original_dtype), grad_v, grad_weights,
                None, None, None, None, None, None)


class LIFFDEFAGL1TorchFunction(torch.autograd.Function):
    """Torch reference with the same established GL1 surrogate-gradient semantics."""

    @staticmethod
    def forward(ctx, x_seq, v_init, weights, tau, h_beta, threshold,
                surrogate_grad_scale, batch_size, full_history_grad=False):
        original_dtype = x_seq.dtype
        x_f32 = x_seq.float()
        v = v_init.float()
        weights_f32 = weights.float()
        T, N = x_f32.shape
        B = int(batch_size)
        F = N // B
        per_batch = weights_f32.dim() == 3
        spikes = []
        v_pre_values = []
        v_history = []
        for t in range(T):
            drift = (-v + x_f32[t]) / float(tau)
            if t:
                states = torch.stack(v_history).reshape(t, B, F)
                row = (weights_f32[t, :, :t].transpose(0, 1) if per_batch
                       else weights_f32[t, :t, None].expand(t, B))
                hist = (row[:, :, None] * states).sum(dim=0).reshape(N)
            else:
                hist = torch.zeros_like(v)
            v_pre = float(h_beta) * drift - hist
            spike = (v_pre >= float(threshold)).to(dtype=torch.float32)
            v = v_pre - spike * float(threshold)
            v_pre_values.append(v_pre)
            v_history.append(v)
            spikes.append(spike)
        v_pre_seq = torch.stack(v_pre_values)
        v_hist_seq = torch.stack(v_history)
        ctx.save_for_backward(weights_f32, v_pre_seq, v_hist_seq)
        ctx.tau = float(tau)
        ctx.h_beta = float(h_beta)
        ctx.threshold = float(threshold)
        ctx.surrogate_grad_scale = float(surrogate_grad_scale)
        ctx.original_dtype = original_dtype
        ctx.batch_size = B
        ctx.full_history_grad = bool(full_history_grad)
        return torch.stack(spikes).to(dtype=original_dtype)

    @staticmethod
    def backward(ctx, grad_spike_seq):
        weights, v_pre_seq, v_hist_seq = ctx.saved_tensors
        T, N = v_pre_seq.shape
        B = ctx.batch_size
        F = N // B
        grad_spike = grad_spike_seq.float()
        grad_x = torch.empty_like(v_pre_seq)
        grad_pre_values = [None] * T
        grad_v_next = torch.zeros(N, device=v_pre_seq.device, dtype=torch.float32)
        grad_history = [torch.zeros_like(grad_v_next) for _ in range(T)]
        for t in range(T - 1, -1, -1):
            z = v_pre_seq[t] - ctx.threshold
            u = (math.pi / 2.0) * ctx.surrogate_grad_scale * z
            surrogate = ctx.surrogate_grad_scale / (2.0 * (1.0 + u.square()))
            state_grad = grad_history[t] if ctx.full_history_grad else grad_v_next
            grad_pre = state_grad + grad_spike[t] * surrogate
            grad_pre_values[t] = grad_pre
            grad_x[t] = grad_pre * ctx.h_beta / ctx.tau
            if ctx.full_history_grad:
                previous_grad = -grad_pre * ctx.h_beta / ctx.tau
                if t == 0:
                    grad_v_next = previous_grad
                else:
                    grad_history[t - 1] = grad_history[t - 1] + previous_grad
                for i in range(t):
                    weight = (weights[t, :, i].repeat_interleave(F)
                              if weights.dim() == 3 else weights[t, i])
                    grad_history[i] = grad_history[i] - weight * grad_pre
            else:
                if weights.dim() == 3:
                    history_sum = weights[t, :, :t].sum(dim=-1).repeat_interleave(F)
                else:
                    history_sum = weights[t, :t].sum().expand(N)
                grad_v_next = -grad_pre * (ctx.h_beta / ctx.tau + history_sum)
        grad_pre_seq = torch.stack(grad_pre_values).reshape(T, B, F)
        hist_b = v_hist_seq.reshape(T, B, F)
        if weights.dim() == 3:
            grad_weights = -torch.einsum('tbf,ibf->tbi', grad_pre_seq, hist_b)
        else:
            grad_weights = -torch.einsum('tbf,ibf->ti', grad_pre_seq, hist_b)
        return (grad_x.to(dtype=ctx.original_dtype), grad_v_next, grad_weights,
                None, None, None, None, None, None)


# =========================================================================================
# Shared per-time variable-order GL1
# =========================================================================================

vo_gl1_forward_kernel_code = r'''
extern "C" __global__
void lif_fde_vo_gl1_forward_kernel(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ scale,
    const float* __restrict__ weights,
    float* __restrict__ spike_seq,
    float* __restrict__ v_pre_seq,
    float* __restrict__ v_hist_seq,
    float* __restrict__ drift_seq,
    const int T, const int N, const int B, const int per_batch,
    const float tau, const float threshold
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int feature_count = N / B;
    const int b = n / feature_count;
    float v = v_init[n];
    for (int t = 0; t < T; ++t) {
        const float drift = (-v + x_seq[t * N + n]) / tau;
        float hist = 0.0f;
        for (int i = 0; i < t; ++i) {
            const int wi = per_batch ? ((t * B + b) * T + i) : (t * T + i);
            hist += weights[wi] * v_hist_seq[i * N + n];
        }
        const float s = scale[per_batch ? (t * B + b) : t];
        const float v_pre = s * drift - hist;
        const float spike = v_pre >= threshold ? 1.0f : 0.0f;
        v_pre_seq[t * N + n] = v_pre;
        drift_seq[t * N + n] = drift;
        spike_seq[t * N + n] = spike;
        v = v_pre - spike * threshold;
        v_hist_seq[t * N + n] = v;
    }
}
'''

vo_gl1_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_vo_gl1_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ scale,
    const float* __restrict__ weights,
    const float* __restrict__ v_pre_seq,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    float* __restrict__ grad_pre_seq,
    const int T, const int N, const int B, const int per_batch,
    const float tau, const float threshold, const float surrogate_scale
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float pi_half = 1.57079632679489661923f;
    const int feature_count = N / B;
    const int b = n / feature_count;
    float grad_v_next = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        const float z = v_pre_seq[t * N + n] - threshold;
        const float u = pi_half * surrogate_scale * z;
        const float surrogate_grad = surrogate_scale / (2.0f * (1.0f + u * u));
        const float grad_pre = grad_v_next + grad_spike_seq[t * N + n] * surrogate_grad;
        grad_pre_seq[t * N + n] = grad_pre;
        const float s = scale[per_batch ? (t * B + b) : t];
        grad_x_seq[t * N + n] = grad_pre * s / tau;
        float history_sum = 0.0f;
        for (int i = 0; i < t; ++i) {
            const int wi = per_batch ? ((t * B + b) * T + i) : (t * T + i);
            history_sum += weights[wi];
        }
        grad_v_next = -grad_pre * (s / tau + history_sum);
    }
    grad_v_init[n] = grad_v_next;
}
'''

vo_gl1_full_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_vo_gl1_full_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ scale,
    const float* __restrict__ weights,
    const float* __restrict__ v_pre_seq,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    float* __restrict__ grad_pre_seq,
    const int T, const int N, const int B, const int per_batch,
    const float tau, const float threshold, const float surrogate_scale
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float pi_half = 1.57079632679489661923f;
    const int feature_count = N / B;
    const int b = n / feature_count;
    float grad_hist[64];
    for (int t = 0; t < T; ++t) grad_hist[t] = 0.0f;
    float grad_initial = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        const float z = v_pre_seq[t * N + n] - threshold;
        const float u = pi_half * surrogate_scale * z;
        const float surrogate_grad = surrogate_scale / (2.0f * (1.0f + u * u));
        const float grad_pre = grad_hist[t] + grad_spike_seq[t * N + n] * surrogate_grad;
        grad_pre_seq[t * N + n] = grad_pre;
        const float s = scale[per_batch ? (t * B + b) : t];
        grad_x_seq[t * N + n] = grad_pre * s / tau;
        const float grad_previous = -grad_pre * s / tau;
        if (t == 0) grad_initial += grad_previous;
        else grad_hist[t - 1] += grad_previous;
        for (int i = 0; i < t; ++i) {
            const int wi = per_batch ? ((t * B + b) * T + i) : (t * T + i);
            grad_hist[i] -= weights[wi] * grad_pre;
        }
    }
    grad_v_init[n] = grad_initial;
}
'''

vo_gl1_parameter_grad_kernel_code = r'''
extern "C" __global__
void lif_fde_vo_gl1_parameter_grad_kernel(
    const float* __restrict__ grad_pre,
    const float* __restrict__ drift_seq,
    const float* __restrict__ v_hist_seq,
    float* __restrict__ grad_scale,
    float* __restrict__ grad_weights,
    const int T, const int N
) {
    const int output = blockIdx.x;
    const int t = output < T ? output : (output - T) / T;
    const int i = output < T ? t : (output - T) % T;
    const float* lhs = grad_pre + t * N;
    const float* rhs = output < T ? drift_seq + t * N : v_hist_seq + i * N;
    float sum = 0.0f;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        sum += lhs[n] * rhs[n];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (output < T) {
            grad_scale[t] = partial[0];
        } else {
            grad_weights[t * T + i] = -partial[0];
        }
    }
}
'''


class LIFFDEVOGL1Function(torch.autograd.Function):
    forward_kernel = None
    backward_kernel = None
    full_backward_kernel = None
    parameter_grad_kernel = None

    @staticmethod
    def get_kernels():
        if LIFFDEVOGL1Function.forward_kernel is None:
            LIFFDEVOGL1Function.forward_kernel = cupy.RawKernel(
                vo_gl1_forward_kernel_code, 'lif_fde_vo_gl1_forward_kernel')
        if LIFFDEVOGL1Function.backward_kernel is None:
            LIFFDEVOGL1Function.backward_kernel = cupy.RawKernel(
                vo_gl1_backward_kernel_code, 'lif_fde_vo_gl1_backward_kernel')
        if LIFFDEVOGL1Function.full_backward_kernel is None:
            LIFFDEVOGL1Function.full_backward_kernel = cupy.RawKernel(
                vo_gl1_full_backward_kernel_code,
                'lif_fde_vo_gl1_full_backward_kernel')
        if LIFFDEVOGL1Function.parameter_grad_kernel is None:
            LIFFDEVOGL1Function.parameter_grad_kernel = cupy.RawKernel(
                vo_gl1_parameter_grad_kernel_code,
                'lif_fde_vo_gl1_parameter_grad_kernel')
        return (LIFFDEVOGL1Function.forward_kernel,
                LIFFDEVOGL1Function.backward_kernel,
                LIFFDEVOGL1Function.full_backward_kernel,
                LIFFDEVOGL1Function.parameter_grad_kernel)

    @staticmethod
    def forward(ctx, x_seq, v_init, scale, weights, tau, threshold,
                surrogate_grad_scale, batch_size, full_history_grad=False):
        if not CUPY_AVAILABLE or not x_seq.is_cuda:
            raise RuntimeError('LIFFDEVOGL1Function requires CUDA and CuPy')
        original_dtype = x_seq.dtype
        x_f32, v_f32 = x_seq.float().contiguous(), v_init.float().contiguous()
        scale_f32, weights_f32 = scale.float().contiguous(), weights.float().contiguous()
        T, N = x_f32.shape
        B, per_batch = int(batch_size), int(scale_f32.dim() == 2)
        spike_seq = torch.empty_like(x_f32)
        v_pre_seq = torch.empty_like(x_f32)
        v_hist_seq = torch.empty_like(x_f32)
        drift_seq = torch.empty_like(x_f32)
        forward_kernel, _, _, _ = LIFFDEVOGL1Function.get_kernels()
        arrays = [x_f32.detach(), v_f32.detach(), scale_f32.detach(), weights_f32.detach(),
                  spike_seq, v_pre_seq, v_hist_seq, drift_seq]
        cp = [cupy.from_dlpack(tensor) for tensor in arrays]
        threads = 256
        forward_kernel(((N + threads - 1) // threads,), (threads,),
                       (*cp, cupy.int32(T), cupy.int32(N), cupy.int32(B),
                        cupy.int32(per_batch), cupy.float32(tau), cupy.float32(threshold)))
        ctx.save_for_backward(scale_f32, weights_f32, v_pre_seq, v_hist_seq, drift_seq)
        ctx.input_shape = x_f32.shape
        ctx.tau, ctx.threshold = float(tau), float(threshold)
        ctx.surrogate_grad_scale = float(surrogate_grad_scale)
        ctx.original_dtype, ctx.batch_size, ctx.per_batch = original_dtype, B, per_batch
        ctx.full_history_grad = bool(full_history_grad)
        return spike_seq.to(dtype=original_dtype)

    @staticmethod
    def backward(ctx, grad_spike_seq):
        scale, weights, v_pre_seq, v_hist_seq, drift_seq = ctx.saved_tensors
        T, N = ctx.input_shape
        grad_spike = grad_spike_seq.float().contiguous()
        grad_x = torch.zeros(ctx.input_shape, device=grad_spike.device, dtype=torch.float32)
        grad_v = torch.zeros(N, device=grad_spike.device, dtype=torch.float32)
        grad_pre = torch.empty(ctx.input_shape, device=grad_spike.device, dtype=torch.float32)
        _, backward_kernel, full_backward_kernel, parameter_grad_kernel = LIFFDEVOGL1Function.get_kernels()
        arrays = [grad_spike.detach(), scale.detach(), weights.detach(), v_pre_seq.detach(), grad_x, grad_v, grad_pre]
        cp = [cupy.from_dlpack(tensor) for tensor in arrays]
        threads = 256
        selected_backward = full_backward_kernel if ctx.full_history_grad else backward_kernel
        selected_backward(((N + threads - 1) // threads,), (threads,),
                          (*cp, cupy.int32(T), cupy.int32(N), cupy.int32(ctx.batch_size),
                           cupy.int32(ctx.per_batch), cupy.float32(ctx.tau),
                           cupy.float32(ctx.threshold), cupy.float32(ctx.surrogate_grad_scale)))
        need_scale_grad = ctx.needs_input_grad[2]
        need_weights_grad = ctx.needs_input_grad[3]
        grad_scale = grad_weights = None
        if ctx.per_batch and (need_scale_grad or need_weights_grad):
            B, F = ctx.batch_size, N // ctx.batch_size
            grad_b = grad_pre.reshape(T, B, F)
            drift_b = drift_seq.reshape(T, B, F)
            hist_b = v_hist_seq.reshape(T, B, F)
            grad_scale = torch.sum(grad_b * drift_b, dim=2)
            grad_weights = -torch.einsum('tbf,ibf->tbi', grad_b, hist_b)
        else:
            grad_scale = torch.sum(grad_pre * drift_seq, dim=1)
            grad_weights = -torch.matmul(grad_pre, v_hist_seq.t())
        return (grad_x.to(dtype=ctx.original_dtype), grad_v, grad_scale,
                grad_weights, None, None, None, None, None)


# =========================================================================================
# Distributed-order GL with detached voltage history
# =========================================================================================

glmulti_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_glmulti_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ scale,
    const float* __restrict__ v_pre_seq,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    float* __restrict__ grad_pre_seq,
    const int T, const int N,
    const float tau, const float threshold, const float surrogate_scale
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float pi_half = 1.57079632679489661923f;
    float grad_v_next = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        const float z = v_pre_seq[t * N + n] - threshold;
        const float u = pi_half * surrogate_scale * z;
        const float surrogate_grad = surrogate_scale / (2.0f * (1.0f + u * u));
        const float grad_pre = grad_v_next + grad_spike_seq[t * N + n] * surrogate_grad;
        grad_pre_seq[t * N + n] = grad_pre;
        const float s = scale[t];
        grad_x_seq[t * N + n] = grad_pre * s / tau;
        // Match LIFNeuronFDE._forward_glmulti: voltage history is detached,
        // while the immediately preceding voltage still enters the drift.
        grad_v_next = -grad_pre * s / tau;
    }
    grad_v_init[n] = grad_v_next;
}
'''

glmulti_full_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_glmulti_full_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ scale,
    const float* __restrict__ weights,
    const float* __restrict__ v_pre_seq,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    float* __restrict__ grad_pre_seq,
    const int T, const int N,
    const float tau, const float threshold, const float surrogate_scale
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float pi_half = 1.57079632679489661923f;
    float grad_hist[64];
    for (int t = 0; t < T; ++t) grad_hist[t] = 0.0f;
    float grad_initial = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        const float z = v_pre_seq[t * N + n] - threshold;
        const float u = pi_half * surrogate_scale * z;
        const float surrogate_grad = surrogate_scale / (2.0f * (1.0f + u * u));
        const float grad_pre = grad_hist[t] + grad_spike_seq[t * N + n] * surrogate_grad;
        grad_pre_seq[t * N + n] = grad_pre;
        const float grad_drift = grad_pre * scale[t];
        grad_x_seq[t * N + n] = grad_drift / tau;
        const float grad_previous = -grad_drift / tau;
        if (t == 0) grad_initial += grad_previous;
        else grad_hist[t - 1] += grad_previous;
        for (int i = 0; i < t; ++i) {
            grad_hist[i] -= weights[t * T + i] * grad_pre;
        }
    }
    grad_v_init[n] = grad_initial;
}
'''


class LIFFDEGLMultiFunction(torch.autograd.Function):
    """Fused multi-term GL with gradients for the aggregated kernel."""
    forward_kernel = None
    backward_kernel = None
    full_backward_kernel = None

    @staticmethod
    def get_kernels():
        if LIFFDEGLMultiFunction.forward_kernel is None:
            LIFFDEGLMultiFunction.forward_kernel = cupy.RawKernel(
                vo_gl1_forward_kernel_code, 'lif_fde_vo_gl1_forward_kernel')
        if LIFFDEGLMultiFunction.backward_kernel is None:
            LIFFDEGLMultiFunction.backward_kernel = cupy.RawKernel(
                glmulti_backward_kernel_code, 'lif_fde_glmulti_backward_kernel')
        if LIFFDEGLMultiFunction.full_backward_kernel is None:
            LIFFDEGLMultiFunction.full_backward_kernel = cupy.RawKernel(
                glmulti_full_backward_kernel_code,
                'lif_fde_glmulti_full_backward_kernel')
        return (LIFFDEGLMultiFunction.forward_kernel,
                LIFFDEGLMultiFunction.backward_kernel,
                LIFFDEGLMultiFunction.full_backward_kernel)

    @staticmethod
    def forward(ctx, x_seq, v_init, scale, weights, tau, threshold,
                surrogate_grad_scale, batch_size, full_history_grad=False):
        if not CUPY_AVAILABLE or not x_seq.is_cuda:
            raise RuntimeError('LIFFDEGLMultiFunction requires CUDA and CuPy')
        original_dtype = x_seq.dtype
        x_f32, v_f32 = x_seq.float().contiguous(), v_init.float().contiguous()
        scale_f32, weights_f32 = scale.float().contiguous(), weights.float().contiguous()
        T, N = x_f32.shape
        if full_history_grad and T > 64:
            raise ValueError('Full glmulti history gradients currently support T <= 64')
        B = int(batch_size)
        if scale_f32.shape != (T,) or weights_f32.shape != (T, T):
            raise ValueError('glmulti scale/weights must have shapes [T] and [T,T]')
        spike_seq = torch.empty_like(x_f32)
        v_pre_seq = torch.empty_like(x_f32)
        v_hist_seq = torch.empty_like(x_f32)
        drift_seq = torch.empty_like(x_f32)
        forward_kernel, _, _ = LIFFDEGLMultiFunction.get_kernels()
        arrays = [x_f32.detach(), v_f32.detach(), scale_f32.detach(), weights_f32.detach(),
                  spike_seq, v_pre_seq, v_hist_seq, drift_seq]
        cp = [cupy.from_dlpack(tensor) for tensor in arrays]
        threads = 256
        forward_kernel(((N + threads - 1) // threads,), (threads,),
                       (*cp, cupy.int32(T), cupy.int32(N), cupy.int32(B),
                        cupy.int32(0), cupy.float32(tau), cupy.float32(threshold)))
        ctx.save_for_backward(scale_f32, weights_f32, v_pre_seq, v_hist_seq, drift_seq)
        ctx.input_shape = x_f32.shape
        ctx.tau = float(tau)
        ctx.threshold = float(threshold)
        ctx.surrogate_grad_scale = float(surrogate_grad_scale)
        ctx.original_dtype = original_dtype
        ctx.full_history_grad = bool(full_history_grad)
        return spike_seq.to(dtype=original_dtype)

    @staticmethod
    def backward(ctx, grad_spike_seq):
        scale, weights, v_pre_seq, v_hist_seq, drift_seq = ctx.saved_tensors
        T, N = ctx.input_shape
        grad_spike = grad_spike_seq.float().contiguous()
        grad_x = torch.empty(ctx.input_shape, device=grad_spike.device, dtype=torch.float32)
        grad_v = torch.empty(N, device=grad_spike.device, dtype=torch.float32)
        grad_pre = torch.empty(ctx.input_shape, device=grad_spike.device, dtype=torch.float32)
        _, backward_kernel, full_backward_kernel = LIFFDEGLMultiFunction.get_kernels()
        threads = 256
        if ctx.full_history_grad:
            arrays = [grad_spike.detach(), scale.detach(), weights.detach(),
                      v_pre_seq.detach(), grad_x, grad_v, grad_pre]
            cp = [cupy.from_dlpack(tensor) for tensor in arrays]
            full_backward_kernel(
                ((N + threads - 1) // threads,), (threads,),
                (*cp, cupy.int32(T), cupy.int32(N), cupy.float32(ctx.tau),
                 cupy.float32(ctx.threshold), cupy.float32(ctx.surrogate_grad_scale)))
        else:
            arrays = [grad_spike.detach(), scale.detach(), v_pre_seq.detach(),
                      grad_x, grad_v, grad_pre]
            cp = [cupy.from_dlpack(tensor) for tensor in arrays]
            backward_kernel(
                ((N + threads - 1) // threads,), (threads,),
                (*cp, cupy.int32(T), cupy.int32(N), cupy.float32(ctx.tau),
                 cupy.float32(ctx.threshold), cupy.float32(ctx.surrogate_grad_scale)))
        grad_scale = grad_weights = None
        if ctx.needs_input_grad[2]:
            grad_scale = torch.sum(grad_pre * drift_seq, dim=1)
        if ctx.needs_input_grad[3]:
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                if torch.cuda.get_device_capability(grad_spike.device)[0] >= 12:
                    torch.backends.cuda.matmul.allow_tf32 = True
                grad_weights = -torch.matmul(grad_pre, v_hist_seq.t())
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        return (grad_x.to(dtype=ctx.original_dtype), grad_v, grad_scale,
                grad_weights, None, None, None, None, None)


# =========================================================================================
# TRAP1 方法 (带h_beta缩放因子的TRAP)
# =========================================================================================

trap1_forward_kernel_code = r'''
extern "C" __global__
void lif_fde_trap1_forward_kernel(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ trap_weights,
    float* __restrict__ spike_seq,
    float* __restrict__ v_seq,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float v_reset,
    const float h_beta
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    float v_mem = v_init[neuron_idx];
    
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_no_reset = (-v_mem + current_input) / tau;
        
        float hist = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            float hist_sum = 0.0f;
            float hist_c = 0.0f;
            
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                const float y = trap_weights[weight_idx] * v_seq[hist_idx * N + neuron_idx] - hist_c;
                const float t = hist_sum + y;
                hist_c = (t - hist_sum) - y;
                hist_sum = t;
            }
            hist = hist_sum;
        }
        
        // TRAP1方法：v_post_charge = v_mem + h_beta * dv_no_reset - hist
        const float v_post_charge = v_mem + h_beta * dv_no_reset - hist;
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq[k % T * N + neuron_idx] = v_mem_new;
        spike_seq[k * N + neuron_idx] = spike;
        v_mem = v_mem_new;
    }
}
'''

trap1_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_trap1_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ grad_v_seq,
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ trap_weights,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float surrogate_scale,
    const float h_beta
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    const float pi_half = 1.57079632679489661923f;
    
    float v_seq_local[256];
    float spike_seq_local[256];
    
    // Forward pass reconstruction
    float v_mem = v_init[neuron_idx];
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_no_reset = (-v_mem + current_input) / tau;
        
        float hist = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                hist += trap_weights[weight_idx] * v_seq_local[hist_idx];
            }
        }
        
        const float v_post_charge = v_mem + h_beta * dv_no_reset - hist;
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq_local[k % T] = v_mem_new;
        spike_seq_local[k] = spike;
        v_mem = v_mem_new;
    }
    
    // Backward pass
    float grad_v = 0.0f;
    
    for (int t = T - 1; t >= 0; t--) {
        grad_v += grad_v_seq[t * N + neuron_idx];
        
        const float v_mem_new = v_seq_local[t % T];
        const float spike = spike_seq_local[t];
        const float v_post_charge = v_mem_new + spike * v_threshold;
        
        // Surrogate gradient
        const float v_scaled = v_post_charge - v_threshold;
        const float pi_half_scale_input = pi_half * surrogate_scale * v_scaled;
        const float denom = 1.0f + pi_half_scale_input * pi_half_scale_input;
        const float surrogate_grad = surrogate_scale / (2.0f * denom);
        
        grad_v += grad_spike_seq[t * N + neuron_idx] * surrogate_grad;
        
        // TRAP1 backward: 使用h_beta缩放因子
        grad_x_seq[t * N + neuron_idx] = h_beta * grad_v / tau;
        float grad_v_mem = grad_v * (1.0f - h_beta / tau);
        
        // History contribution
        if (t > 0) {
            int history_len = (t < T) ? t : T;
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                grad_v_mem -= trap_weights[weight_idx] * grad_v;
            }
        }
        
        grad_v = grad_v_mem;
    }
    
    grad_v_init[neuron_idx] = grad_v;
}
'''


class LIFFDETRAP1Function(torch.autograd.Function):
    """TRAP1方法的CuPy Function - 带h_beta缩放因子"""
    forward_kernel = None
    backward_kernel = None
    
    @staticmethod
    def get_kernels():
        if LIFFDETRAP1Function.forward_kernel is None:
            LIFFDETRAP1Function.forward_kernel = cupy.RawKernel(trap1_forward_kernel_code, 'lif_fde_trap1_forward_kernel')
        if LIFFDETRAP1Function.backward_kernel is None:
            LIFFDETRAP1Function.backward_kernel = cupy.RawKernel(trap1_backward_kernel_code, 'lif_fde_trap1_backward_kernel')
        return LIFFDETRAP1Function.forward_kernel, LIFFDETRAP1Function.backward_kernel
    
    @staticmethod
    def forward(ctx, x_seq, v_init, trap_weights, tau, beta, v_threshold, v_reset, surrogate_grad_scale):
        """
        Args:
            x_seq: [T, N] 输入序列
            v_init: [N] 初始膜电位
            trap_weights: [T+1] TRAP权重系数
            tau: 时间常数
            beta: 分数阶参数
            v_threshold: 阈值
            v_reset: 重置电压
            surrogate_grad_scale: 替代梯度缩放
        """
        assert CUPY_AVAILABLE and x_seq.is_cuda
        T, N = x_seq.shape
        
        # 计算h_beta
        h_beta = math.pow(1.0, beta) if isinstance(beta, float) else math.pow(1.0, beta.item())
        
        forward_kernel, _ = LIFFDETRAP1Function.get_kernels()
        
        # 转换为float32并确保连续
        x_seq_f32 = x_seq.float().contiguous()
        v_init_f32 = v_init.float().contiguous()
        trap_weights_f32 = trap_weights.float().contiguous()
        
        # 准备输出
        spike_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        v_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        
        # 使用dlpack转换为CuPy数组
        x_seq_cp = cupy.from_dlpack(x_seq_f32.detach())
        v_init_cp = cupy.from_dlpack(v_init_f32.detach())
        trap_weights_cp = cupy.from_dlpack(trap_weights_f32.detach())
        spike_seq_cp = cupy.from_dlpack(spike_seq)
        v_seq_cp = cupy.from_dlpack(v_seq)
        
        # 执行kernel
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        forward_kernel(
            (blocks,), (threads_per_block,),
            (x_seq_cp, v_init_cp, trap_weights_cp, spike_seq_cp, v_seq_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(tau), cupy.float32(beta),
             cupy.float32(v_threshold), cupy.float32(v_reset), cupy.float32(h_beta))
        )
        
        # 保存用于backward
        ctx.save_for_backward(x_seq_f32, v_init_f32, trap_weights_f32)
        ctx.tau = tau
        ctx.beta = beta
        ctx.v_threshold = v_threshold
        ctx.surrogate_grad_scale = surrogate_grad_scale
        ctx.h_beta = h_beta
        ctx.T = T
        ctx.N = N
        
        return spike_seq, v_seq
    
    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        x_seq, v_init, trap_weights = ctx.saved_tensors
        T, N = ctx.T, ctx.N
        
        _, backward_kernel = LIFFDETRAP1Function.get_kernels()
        
        # 准备梯度输出
        grad_x_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        grad_v_init = torch.zeros(N, dtype=torch.float32, device=x_seq.device)
        
        # 确保梯度是float32和连续的
        grad_spike_seq_f32 = grad_spike_seq.float().contiguous()
        grad_v_seq_f32 = grad_v_seq.float().contiguous()
        
        # 使用dlpack转换
        grad_spike_seq_cp = cupy.from_dlpack(grad_spike_seq_f32.detach())
        grad_v_seq_cp = cupy.from_dlpack(grad_v_seq_f32.detach())
        x_seq_cp = cupy.from_dlpack(x_seq.detach())
        v_init_cp = cupy.from_dlpack(v_init.detach())
        trap_weights_cp = cupy.from_dlpack(trap_weights.detach())
        grad_x_seq_cp = cupy.from_dlpack(grad_x_seq)
        grad_v_init_cp = cupy.from_dlpack(grad_v_init)
        
        # 执行kernel
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        backward_kernel(
            (blocks,), (threads_per_block,),
            (grad_spike_seq_cp, grad_v_seq_cp, x_seq_cp, v_init_cp, trap_weights_cp,
             grad_x_seq_cp, grad_v_init_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(ctx.tau), cupy.float32(ctx.beta),
             cupy.float32(ctx.v_threshold), cupy.float32(ctx.surrogate_grad_scale),
             cupy.float32(ctx.h_beta))
        )
        
        return grad_x_seq, grad_v_init, None, None, None, None, None, None


# =========================================================================================
# PREDICTOR1 方法 (带h_beta缩放因子的PREDICTOR)
# =========================================================================================

predictor1_forward_kernel_code = r'''
extern "C" __global__
void lif_fde_predictor1_forward_kernel(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ predictor_weights,
    float* __restrict__ spike_seq,
    float* __restrict__ v_seq,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float v_reset,
    const float h_beta,
    const float gamma_beta_plus_1
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    float v_mem = v_init[neuron_idx];
    
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_current = (-v_mem + current_input) / tau;
        
        float hist_integral = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            float hist_sum = 0.0f;
            float hist_c = 0.0f;
            
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                // 重新计算历史导数
                const float hist_v = v_seq[hist_idx * N + neuron_idx];
                const float hist_dv = -hist_v / tau;
                const float y = predictor_weights[weight_idx] * hist_dv - hist_c;
                const float t = hist_sum + y;
                hist_c = (t - hist_sum) - y;
                hist_sum = t;
            }
            hist_integral = hist_sum;
        }
        
        // PREDICTOR1方法：v_post_charge = v_mem + h_beta * (w_0 * dv_current + hist_integral)
        const float w_current = predictor_weights[T];
        const float v_post_charge = v_mem + h_beta * (w_current * dv_current + hist_integral);
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq[k % T * N + neuron_idx] = v_mem_new;
        spike_seq[k * N + neuron_idx] = spike;
        v_mem = v_mem_new;
    }
}
'''

predictor1_backward_kernel_code = r'''
extern "C" __global__
void lif_fde_predictor1_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ grad_v_seq,
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ predictor_weights,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    const int T,
    const int N,
    const float tau,
    const float beta,
    const float v_threshold,
    const float surrogate_scale,
    const float h_beta,
    const float gamma_beta_plus_1
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    const float pi_half = 1.57079632679489661923f;
    
    float v_seq_local[256];
    float spike_seq_local[256];
    
    // Forward pass reconstruction
    float v_mem = v_init[neuron_idx];
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_current = (-v_mem + current_input) / tau;
        
        float hist_integral = 0.0f;
        if (k > 0) {
            int history_len = (k < T) ? k : T;
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                const float hist_v = v_seq_local[hist_idx];
                const float hist_dv = -hist_v / tau;
                hist_integral += predictor_weights[weight_idx] * hist_dv;
            }
        }
        
        const float w_current = predictor_weights[T];
        const float v_post_charge = v_mem + h_beta * (w_current * dv_current + hist_integral);
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq_local[k % T] = v_mem_new;
        spike_seq_local[k] = spike;
        v_mem = v_mem_new;
    }
    
    // Backward pass
    float grad_v = 0.0f;
    
    for (int t = T - 1; t >= 0; t--) {
        grad_v += grad_v_seq[t * N + neuron_idx];
        
        const float v_mem_new = v_seq_local[t % T];
        const float spike = spike_seq_local[t];
        const float v_post_charge = v_mem_new + spike * v_threshold;
        
        // Surrogate gradient
        const float v_scaled = v_post_charge - v_threshold;
        const float pi_half_scale_input = pi_half * surrogate_scale * v_scaled;
        const float denom = 1.0f + pi_half_scale_input * pi_half_scale_input;
        const float surrogate_grad = surrogate_scale / (2.0f * denom);
        
        grad_v += grad_spike_seq[t * N + neuron_idx] * surrogate_grad;
        
        // PREDICTOR1 backward: 使用h_beta缩放因子
        const float w_current = predictor_weights[T];
        grad_x_seq[t * N + neuron_idx] = h_beta * w_current * grad_v / tau;
        float grad_v_mem = grad_v * (1.0f - h_beta * w_current / tau);
        
        // History contribution
        if (t > 0) {
            int history_len = (t < T) ? t : T;
            for (int i = 0; i < history_len; i++) {
                const int hist_idx = i % T;
                const int weight_idx = T - history_len + i;
                // 梯度通过历史导数传播
                grad_v_mem -= h_beta * predictor_weights[weight_idx] * grad_v / tau;
            }
        }
        
        grad_v = grad_v_mem;
    }
    
    grad_v_init[neuron_idx] = grad_v;
}
'''


class LIFFDEPREDICTOR1Function(torch.autograd.Function):
    """PREDICTOR1方法的CuPy Function - 带h_beta缩放因子"""
    forward_kernel = None
    backward_kernel = None
    
    @staticmethod
    def get_kernels():
        if LIFFDEPREDICTOR1Function.forward_kernel is None:
            LIFFDEPREDICTOR1Function.forward_kernel = cupy.RawKernel(predictor1_forward_kernel_code, 'lif_fde_predictor1_forward_kernel')
        if LIFFDEPREDICTOR1Function.backward_kernel is None:
            LIFFDEPREDICTOR1Function.backward_kernel = cupy.RawKernel(predictor1_backward_kernel_code, 'lif_fde_predictor1_backward_kernel')
        return LIFFDEPREDICTOR1Function.forward_kernel, LIFFDEPREDICTOR1Function.backward_kernel
    
    @staticmethod
    def forward(ctx, x_seq, v_init, predictor_weights, tau, beta, v_threshold, v_reset, 
                surrogate_grad_scale, h_beta, gamma_beta_plus_1):
        """
        Args:
            x_seq: [T, N] 输入序列
            v_init: [N] 初始膜电位
            predictor_weights: [T+1] PREDICTOR权重系数
            h_beta: h^beta
            gamma_beta_plus_1: Gamma(beta+1)
        """
        assert CUPY_AVAILABLE and x_seq.is_cuda
        T, N = x_seq.shape
        
        forward_kernel, _ = LIFFDEPREDICTOR1Function.get_kernels()
        
        # 转换为float32并确保连续
        x_seq_f32 = x_seq.float().contiguous()
        v_init_f32 = v_init.float().contiguous()
        predictor_weights_f32 = predictor_weights.float().contiguous()
        
        # 准备输出
        spike_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        v_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        
        # 使用dlpack转换为CuPy数组
        x_seq_cp = cupy.from_dlpack(x_seq_f32.detach())
        v_init_cp = cupy.from_dlpack(v_init_f32.detach())
        predictor_weights_cp = cupy.from_dlpack(predictor_weights_f32.detach())
        spike_seq_cp = cupy.from_dlpack(spike_seq)
        v_seq_cp = cupy.from_dlpack(v_seq)
        
        # 执行kernel
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        forward_kernel(
            (blocks,), (threads_per_block,),
            (x_seq_cp, v_init_cp, predictor_weights_cp, spike_seq_cp, v_seq_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(tau), cupy.float32(beta),
             cupy.float32(v_threshold), cupy.float32(v_reset), 
             cupy.float32(h_beta), cupy.float32(gamma_beta_plus_1))
        )
        
        # 保存用于backward
        ctx.save_for_backward(x_seq_f32, v_init_f32, predictor_weights_f32)
        ctx.tau = tau
        ctx.beta = beta
        ctx.v_threshold = v_threshold
        ctx.surrogate_grad_scale = surrogate_grad_scale
        ctx.h_beta = h_beta
        ctx.gamma_beta_plus_1 = gamma_beta_plus_1
        ctx.T = T
        ctx.N = N
        
        return spike_seq, v_seq
    
    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        x_seq, v_init, predictor_weights = ctx.saved_tensors
        T, N = ctx.T, ctx.N
        
        _, backward_kernel = LIFFDEPREDICTOR1Function.get_kernels()
        
        # 准备梯度输出
        grad_x_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        grad_v_init = torch.zeros(N, dtype=torch.float32, device=x_seq.device)
        
        # 确保梯度是float32和连续的
        grad_spike_seq_f32 = grad_spike_seq.float().contiguous()
        grad_v_seq_f32 = grad_v_seq.float().contiguous()
        
        # 使用dlpack转换
        grad_spike_seq_cp = cupy.from_dlpack(grad_spike_seq_f32.detach())
        grad_v_seq_cp = cupy.from_dlpack(grad_v_seq_f32.detach())
        x_seq_cp = cupy.from_dlpack(x_seq.detach())
        v_init_cp = cupy.from_dlpack(v_init.detach())
        predictor_weights_cp = cupy.from_dlpack(predictor_weights.detach())
        grad_x_seq_cp = cupy.from_dlpack(grad_x_seq)
        grad_v_init_cp = cupy.from_dlpack(grad_v_init)
        
        # 执行kernel
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        backward_kernel(
            (blocks,), (threads_per_block,),
            (grad_spike_seq_cp, grad_v_seq_cp, x_seq_cp, v_init_cp, predictor_weights_cp,
             grad_x_seq_cp, grad_v_init_cp,
             cupy.int32(T), cupy.int32(N), cupy.float32(ctx.tau), cupy.float32(ctx.beta),
             cupy.float32(ctx.v_threshold), cupy.float32(ctx.surrogate_grad_scale),
             cupy.float32(ctx.h_beta), cupy.float32(ctx.gamma_beta_plus_1))
        )
        
        return grad_x_seq, grad_v_init, None, None, None, None, None, None, None, None


# =========================================================================================
# EULER 方法 (标准LIF，无历史项)
# =========================================================================================

euler_forward_kernel_code = r'''
extern "C" __global__
void lif_euler_forward_kernel(
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    float* __restrict__ spike_seq,
    float* __restrict__ v_seq,
    const int T,
    const int N,
    const float tau,
    const float v_threshold,
    const float v_reset
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    float v_mem = v_init[neuron_idx];
    
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        
        // 标准LIF欧拉方法：dv/dt = (-v + I) / tau
        const float dv_no_reset = (-v_mem + current_input) / tau;
        
        // v_{n+1} = v_n + dt * dv/dt, dt=1
        const float v_post_charge = v_mem + dv_no_reset;
        
        // 脉冲检测
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        
        // 硬重置
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq[k * N + neuron_idx] = v_mem_new;
        spike_seq[k * N + neuron_idx] = spike;
        v_mem = v_mem_new;
    }
}
'''

euler_backward_kernel_code = r'''
extern "C" __global__
void lif_euler_backward_kernel(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ grad_v_seq,
    const float* __restrict__ x_seq,
    const float* __restrict__ v_init,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_v_init,
    const int T,
    const int N,
    const float tau,
    const float v_threshold,
    const float surrogate_scale
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= N) return;
    
    const float pi_half = 1.57079632679489661923f;
    
    // 局部缓存：重建前向传播
    float v_seq_local[256];
    float spike_seq_local[256];
    
    float v_mem = v_init[neuron_idx];
    for (int k = 0; k < T; k++) {
        const float current_input = x_seq[k * N + neuron_idx];
        const float dv_no_reset = (-v_mem + current_input) / tau;
        const float v_post_charge = v_mem + dv_no_reset;
        const float spike = (v_post_charge >= v_threshold) ? 1.0f : 0.0f;
        const float v_mem_new = v_post_charge - spike * v_threshold;
        
        v_seq_local[k] = v_mem_new;
        spike_seq_local[k] = spike;
        v_mem = v_mem_new;
    }
    
    // 反向传播：Euler标准LIF
    // 前向: v_post_charge = v_mem + (-v_mem + x)/tau = x/tau + v_mem*(1-1/tau)
    //      spike = surrogate(v_post_charge - threshold)
    //      v_mem_new = v_post_charge - spike.detach() * threshold
    
    float grad_v_mem = 0.0f;
    
    for (int t = T - 1; t >= 0; t--) {
        // 来自v_mem_new的梯度
        grad_v_mem += grad_v_seq[t * N + neuron_idx];
        
        // 重建v_post_charge
        const float v_mem_new = v_seq_local[t];
        const float spike = spike_seq_local[t];
        const float v_post_charge = v_mem_new + spike * v_threshold;
        
        // 来自spike的梯度（通过surrogate函数）
        const float v_scaled = v_post_charge - v_threshold;
        const float pi_half_scale_input = pi_half * surrogate_scale * v_scaled;
        const float denom = 1.0f + pi_half_scale_input * pi_half_scale_input;
        const float surrogate_grad = surrogate_scale / (2.0f * denom);
        
        const float grad_v_post = grad_v_mem + grad_spike_seq[t * N + neuron_idx] * surrogate_grad;
        
        // Euler梯度: v_post = x/tau + v_mem*(1-1/tau)
        grad_x_seq[t * N + neuron_idx] = grad_v_post / tau;
        grad_v_mem = grad_v_post * (1.0f - 1.0f / tau);
    }
    
    grad_v_init[neuron_idx] = grad_v_mem;
}
'''


class LIFEulerFunction(torch.autograd.Function):
    """Euler方法的CuPy Function（标准LIF，无历史项）"""
    forward_kernel = None
    backward_kernel = None
    
    @staticmethod
    def get_kernels():
        if LIFEulerFunction.forward_kernel is None:
            LIFEulerFunction.forward_kernel = cupy.RawKernel(euler_forward_kernel_code, 'lif_euler_forward_kernel')
        if LIFEulerFunction.backward_kernel is None:
            LIFEulerFunction.backward_kernel = cupy.RawKernel(euler_backward_kernel_code, 'lif_euler_backward_kernel')
        return LIFEulerFunction.forward_kernel, LIFEulerFunction.backward_kernel
    
    @staticmethod
    def forward(ctx, x_seq, v_init, tau, v_threshold, v_reset, surrogate_grad_scale):
        """
        Euler方法前向传播（标准LIF）
        
        Args:
            x_seq: [T, N] 输入序列
            v_init: [N] 初始膜电位
            tau: 时间常数
            v_threshold: 阈值
            v_reset: 重置电压
            surrogate_grad_scale: 替代梯度缩放
        """
        assert CUPY_AVAILABLE and x_seq.is_cuda
        T, N = x_seq.shape
        
        forward_kernel, _ = LIFEulerFunction.get_kernels()
        
        x_seq_f32 = x_seq.float().contiguous()
        v_init_f32 = v_init.float().contiguous()
        
        spike_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        v_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq.device)
        
        import cupy
        x_seq_cp = cupy.from_dlpack(x_seq_f32.detach())
        v_init_cp = cupy.from_dlpack(v_init_f32.detach())
        spike_seq_cp = cupy.from_dlpack(spike_seq)
        v_seq_cp = cupy.from_dlpack(v_seq)
        
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        forward_kernel(
            (blocks,), (threads_per_block,),
            (x_seq_cp, v_init_cp, spike_seq_cp, v_seq_cp,
             cupy.int32(T), cupy.int32(N), 
             cupy.float32(tau), cupy.float32(v_threshold), cupy.float32(v_reset))
        )
        
        ctx.save_for_backward(x_seq_f32, v_init_f32)
        ctx.tau = tau
        ctx.v_threshold = v_threshold
        ctx.surrogate_grad_scale = surrogate_grad_scale
        ctx.T = T
        ctx.N = N
        
        return spike_seq, v_seq
    
    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        x_seq_f32, v_init_f32 = ctx.saved_tensors
        T = ctx.T
        N = ctx.N
        
        _, backward_kernel = LIFEulerFunction.get_kernels()
        
        grad_x_seq = torch.zeros((T, N), dtype=torch.float32, device=x_seq_f32.device)
        grad_v_init = torch.zeros(N, dtype=torch.float32, device=x_seq_f32.device)
        
        import cupy
        grad_spike_seq_cp = cupy.from_dlpack(grad_spike_seq.float().contiguous())
        grad_v_seq_cp = cupy.from_dlpack(grad_v_seq.float().contiguous())
        x_seq_cp = cupy.from_dlpack(x_seq_f32.detach())
        v_init_cp = cupy.from_dlpack(v_init_f32.detach())
        grad_x_seq_cp = cupy.from_dlpack(grad_x_seq)
        grad_v_init_cp = cupy.from_dlpack(grad_v_init)
        
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        backward_kernel(
            (blocks,), (threads_per_block,),
            (grad_spike_seq_cp, grad_v_seq_cp, x_seq_cp, v_init_cp,
             grad_x_seq_cp, grad_v_init_cp,
             cupy.int32(T), cupy.int32(N), 
             cupy.float32(ctx.tau), cupy.float32(ctx.v_threshold),
             cupy.float32(ctx.surrogate_grad_scale))
        )
        
        return grad_x_seq, grad_v_init, None, None, None, None


# =========================================================================================
# 测试函数
# =========================================================================================

def test_method(method_name):
    """测试单个方法的一致性"""
    if not CUPY_AVAILABLE:
        print(f"CuPy not available, skipping {method_name} test")
        return False
    
    print("=" * 80)
    print(f"Testing {method_name.upper()} Method")
    print("=" * 80)
    
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from neuron import LIFNeuronFDE
    
    T, N, features = 20, 64, 128
    tau, beta, threshold = 2.0, 0.7, 1.0
    
    print(f"\nConfiguration: T={T}, N={N}, features={features}")
    print(f"tau={tau}, beta={beta}, threshold={threshold}")
    
    # 创建两个神经元
    neuron_torch = LIFNeuronFDE(
        tau=tau, threshold=threshold, beta=beta, method=method_name,
        T=T, batchsize=N, step_mode='m', backend='torch'
    ).cuda()
    
    neuron_cupy = LIFNeuronFDE(
        tau=tau, threshold=threshold, beta=beta, method=method_name,
        T=T, batchsize=N, step_mode='m', backend='cupy'
    ).cuda()
    
    torch.manual_seed(42)
    x_seq = torch.randn(T, N, features).cuda() * 5.0
    
    # 前向传播
    print("\n1. Forward Pass:")
    neuron_torch.reset()
    y_torch = neuron_torch(x_seq)
    
    neuron_cupy.reset()
    y_cupy = neuron_cupy(x_seq)
    
    max_diff = (y_torch - y_cupy).abs().max().item()
    mean_diff = (y_torch - y_cupy).abs().mean().item()
    
    print(f"  Max diff: {max_diff:.10e}")
    print(f"  Mean diff: {mean_diff:.10e}")
    print(f"  Torch spike rate: {y_torch.mean().item():.6f}")
    print(f"  CuPy spike rate: {y_cupy.mean().item():.6f}")
    
    forward_passed = max_diff < 1e-5
    print(f"  {'✓ PASSED' if forward_passed else '✗ FAILED'} (threshold: 1e-5)")
    
    # 反向传播
    print("\n2. Backward Pass:")
    neuron_torch.reset()
    x_torch = x_seq.clone().requires_grad_(True)
    y_torch = neuron_torch(x_torch)
    loss_torch = y_torch.mean()
    loss_torch.backward()
    grad_torch = x_torch.grad.clone()
    
    neuron_cupy.reset()
    x_cupy = x_seq.clone().requires_grad_(True)
    y_cupy = neuron_cupy(x_cupy)
    loss_cupy = y_cupy.mean()
    loss_cupy.backward()
    grad_cupy = x_cupy.grad.clone()
    
    max_grad_diff = (grad_torch - grad_cupy).abs().max().item()
    mean_grad_diff = (grad_torch - grad_cupy).abs().mean().item()
    
    print(f"  Max grad diff: {max_grad_diff:.10e}")
    print(f"  Mean grad diff: {mean_grad_diff:.10e}")
    print(f"  Torch grad norm: {grad_torch.norm().item():.8f}")
    print(f"  CuPy grad norm: {grad_cupy.norm().item():.8f}")
    
    backward_passed = max_grad_diff < 1e-4
    print(f"  {'✓ PASSED' if backward_passed else '✗ FAILED'} (threshold: 1e-4)")
    
    # 总结
    print("\n" + "=" * 80)
    if forward_passed and backward_passed:
        print(f"✓ {method_name.upper()} Method: ALL TESTS PASSED")
    else:
        print(f"✗ {method_name.upper()} Method: SOME TESTS FAILED")
    print("=" * 80)
    
    return forward_passed and backward_passed


if __name__ == "__main__":
    results = {}
    
    # 测试三种方法
    for method in ['gl', 'trap', 'predictor']:
        try:
            results[method] = test_method(method)
        except Exception as e:
            print(f"\n✗ {method.upper()} test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[method] = False
    
    # 最终总结
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for method, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {method.upper():12s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("✓✓✓ ALL METHODS PASSED ✓✓✓")
    else:
        print("✗ Some methods failed")
    print("=" * 80)
    
    exit(0 if all_passed else 1)
