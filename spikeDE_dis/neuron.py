import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .surrogate import sigmoid_surrogate, arctan_surrogate, piecewise_linear_surrogate, gaussian_surrogate


surrogate_f = {"sigmoid_surrogate":sigmoid_surrogate,
           "arctan_surrogate":arctan_surrogate,
           "piecewise_linear_surrogate":piecewise_linear_surrogate,
           "gaussian_surrogate":gaussian_surrogate,
}


class BaseNeuron(nn.Module):
    def __init__(self, tau=0.5, threshold=1.0, surrogate_grad_scale=5.0, surrogate_opt="arctan_surrogate",
                 tau_learnable=False, surrogate_alpha=None, surrogate_function=None):
        super(BaseNeuron, self).__init__()
        if surrogate_alpha is not None:
            surrogate_grad_scale = surrogate_alpha
        if surrogate_function is not None:
            surrogate_map = {
                'sigmoid': 'sigmoid_surrogate',
                'arctan': 'arctan_surrogate',
                'piecewise_linear': 'piecewise_linear_surrogate',
                'gaussian': 'gaussian_surrogate'
            }
            surrogate_opt = surrogate_map.get(surrogate_function, surrogate_function)
        # Store the initial tau value
        self.initial_tau = tau

        # Set tau to be learnable if specified
        if tau_learnable:
            # Create a learnable parameter initialized to zero
            self.tau_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            # The actual tau will be computed in the forward pass
        else:
            self.tau_param = None
            self.tau = tau

        self.threshold = threshold
        self.surrogate_grad_scale = surrogate_grad_scale
        self.surrogate_alpha = surrogate_grad_scale
        self.surrogate_function = surrogate_function if surrogate_function is not None else surrogate_opt
        self.surrogate_f = surrogate_f[surrogate_opt]
        self.tau_learnable = tau_learnable

    def get_tau(self):
        # Compute and return the current tau value
        if self.tau_learnable:
            # tau can only increase from the initial value
            return self.initial_tau * (1 + torch.exp(self.tau_param))
        else:
            return self.tau

    # def __call__(self, *args, **kwargs):
    #     # Allow single-argument calls: treat as initial call with v_mem=0 and current_input=args[0]
    #     if len(args) == 1 and 'current_input' not in kwargs:
    #         x = args[0]
    #         return x
    #     # Otherwise pass through to forward
    #     return super(BaseNeuron, self).__call__(*args, **kwargs)

    def forward(self, v_mem, current_input):
        raise NotImplementedError("Neuron forward method must be overridden.")

# class IFNeuron(BaseNeuron):
#     def forward(self, v_mem, current_input=None):
#         if current_input is None:
#             return v_mem
#         tau = self.get_tau()
#         v_scaled = v_mem - self.threshold
#         spike_out = self.surrogate_f(v_scaled, self.surrogate_grad_scale)
#         # v_mem = v_mem * (1.0 + spike_out.detach())
#         dv_dt = (-spike_out * self.threshold + current_input) / tau  # No leak term for IF neuron
#         return dv_dt, spike_out

# class LIFNeuron(BaseNeuron):
#     def forward(self, v_mem, current_input=None):
#         if current_input is None:
#             # print('---Please provide current_input for LIFNeuron---')
#             return v_mem
#         tau = self.get_tau()
#         v_scaled = v_mem - self.threshold
#         spike_out = self.surrogate_f(v_scaled, self.surrogate_grad_scale)
#         dv_dt = (-v_mem + current_input - spike_out.detach() * self.threshold) / tau
#         return dv_dt, spike_out

class IFNeuron(BaseNeuron):
    def forward(self, v_mem, current_input=None):
        if current_input is None:
            return v_mem
        tau = self.get_tau()
        dt = 1.0  # 若外部 step_size 可变，这里用传入的 dt 这边我们强制dt=1.0进行膜电位预估
        # 1) 先按 LIF 漂移（不含 reset）
        dv_no_reset = (current_input) / tau #+ 1
        # print("*"*36,v_mem.shape, current_input.shape)
        v_post_charge = v_mem + dt * dv_no_reset  # fire-after-charge 的判阈值点
        # 2) 在充电后的电压上“开火”
        spike = self.surrogate_f(v_post_charge - self.threshold,
                                 self.surrogate_grad_scale)
        # 3) reset 的影响写进导数，但对 reset 的 spike 做 detach
        dv_dt = dv_no_reset - (spike.detach() * self.threshold) / tau
        return dv_dt, spike


class LIFNeuron(BaseNeuron):
    state_multiplier = 1
    state_names = ("v",)

    def forward(self, v_mem, current_input=None):
        if current_input is None:
            return v_mem
        tau = self.get_tau()
        dt = 1.0  # 若外部 step_size 可变，这里用传入的 dt 这边我们强制dt=1.0进行膜电位预估
        # 1) 先按 LIF 漂移（不含 reset）
        dv_no_reset = (-v_mem + current_input) / tau #+ 1
        # print("*"*36,v_mem.shape, current_input.shape)
        v_post_charge = v_mem + dt * dv_no_reset  # fire-after-charge 的判阈值点
        # 2) 在充电后的电压上“开火”
        spike = self.surrogate_f(v_post_charge - self.threshold,
                                 self.surrogate_grad_scale)
        # 3) reset 的影响写进导数，但对 reset 的 spike 做 detach
        dv_dt = dv_no_reset - (spike.detach() * self.threshold) / tau
        return dv_dt, spike

class LIFNeuron_OrderOne(BaseNeuron):
    def forward(self, v_mem, current_input=None):
        if current_input is None:
            return v_mem

        tau = self.get_tau()
        # 1) 先按 LIF 漂移（不含 reset）
        dv_no_reset = (-v_mem + current_input) / tau #
        v_post_charge = v_mem + dv_no_reset  # fire-after-charge 的判阈值点

        # 2) 在充电后的电压上“开火”
        spike = self.surrogate_f(v_post_charge - self.threshold,
                                 self.surrogate_grad_scale)
        # 3) reset 的影响写进导数，但对 reset 的 spike 做 detach
        v_mem = v_post_charge - (spike.detach() * self.threshold) #/ tau
        return v_mem, spike


class HHNeuron(BaseNeuron):
    """Hodgkin-Huxley neuron with packed ODE state [V, m, h, n]."""

    state_multiplier = 4
    state_names = ("V", "m", "h", "n")

    def __init__(self,
                 tau=1.0,
                 threshold=1.0,
                 surrogate_grad_scale=5.0,
                 surrogate_opt="arctan_surrogate",
                 tau_learnable=False,
                 input_scale=40.0,
                 v_threshold=-55.0,
                 v_rest=-65.0,
                 c_m=1.0,
                 g_na=120.0,
                 g_k=36.0,
                 g_l=0.3,
                 e_na=50.0,
                 e_k=-77.0,
                 e_l=-54.387,
                 normalize_spike=False,
                 learn_input_scale=False,
                 learn_spike_affine=False,
                 spike_after_update=False,
                 spike_update_dt=1.0,
                 reset_strength=0.0,
                 dynamics_mode='bio',
                 gate_rate_scale=0.1,
                 train_mix=1.0):
        super().__init__(
            tau=tau,
            threshold=threshold,
            surrogate_grad_scale=surrogate_grad_scale,
            surrogate_opt=surrogate_opt,
            tau_learnable=tau_learnable,
        )
        self.v_threshold = v_threshold
        self.v_rest = v_rest
        self.c_m = c_m
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.e_na = e_na
        self.e_k = e_k
        self.e_l = e_l
        self.normalize_spike = normalize_spike
        self.learn_input_scale = learn_input_scale
        self.learn_spike_affine = learn_spike_affine
        self.spike_after_update = spike_after_update
        self.spike_update_dt = spike_update_dt
        self.reset_strength = reset_strength
        if dynamics_mode not in {'bio', 'train_scale'}:
            raise ValueError(f"Unsupported HH dynamics_mode: {dynamics_mode}")
        self.dynamics_mode = dynamics_mode
        self.gate_rate_scale = gate_rate_scale
        self.train_mix = train_mix
        input_scale = max(float(input_scale), 1e-8)
        if learn_input_scale:
            self.log_input_scale = nn.Parameter(torch.tensor(math.log(input_scale), dtype=torch.float32))
        else:
            self.register_buffer('input_scale_buffer', torch.tensor(input_scale, dtype=torch.float32), persistent=False)
        if learn_spike_affine:
            self.spike_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.spike_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.register_buffer('spike_gain_buffer', torch.tensor(1.0, dtype=torch.float32), persistent=False)
            self.register_buffer('spike_bias_buffer', torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def get_input_scale(self):
        if self.learn_input_scale:
            return self.log_input_scale.exp()
        return self.input_scale_buffer

    def _apply_spike_affine(self, normalized_voltage):
        if self.learn_spike_affine:
            return normalized_voltage * self.spike_gain + self.spike_bias
        return normalized_voltage * self.spike_gain_buffer + self.spike_bias_buffer

    @staticmethod
    def _safe_rate(numerator, denominator):
        eps = torch.finfo(denominator.dtype).eps
        safe_denominator = torch.where(
            denominator.abs() < eps,
            torch.full_like(denominator, eps),
            denominator,
        )
        return numerator / safe_denominator

    def _alpha_m(self, v):
        return self._safe_rate(0.1 * (v + 40.0), 1.0 - torch.exp(-(v + 40.0) / 10.0))

    def _beta_m(self, v):
        return 4.0 * torch.exp(-(v + 65.0) / 18.0)

    def _alpha_h(self, v):
        return 0.07 * torch.exp(-(v + 65.0) / 20.0)

    def _beta_h(self, v):
        return 1.0 / (1.0 + torch.exp(-(v + 35.0) / 10.0))

    def _alpha_n(self, v):
        return self._safe_rate(0.01 * (v + 55.0), 1.0 - torch.exp(-(v + 55.0) / 10.0))

    def _beta_n(self, v):
        return 0.125 * torch.exp(-(v + 65.0) / 80.0)

    def _gate_inf(self, alpha, beta):
        return alpha / (alpha + beta)

    def initial_state(self, output_shape, device=None, dtype=None):
        if self.dynamics_mode == 'train_scale':
            v = torch.zeros(output_shape, device=device, dtype=dtype)
            rate_v = torch.full(output_shape, self.v_rest, device=device, dtype=dtype)
        else:
            v = torch.full(output_shape, self.v_rest, device=device, dtype=dtype)
            rate_v = v
        m = self._gate_inf(self._alpha_m(rate_v), self._beta_m(rate_v))
        h = self._gate_inf(self._alpha_h(rate_v), self._beta_h(rate_v))
        n = self._gate_inf(self._alpha_n(rate_v), self._beta_n(rate_v))
        return torch.cat((v, m, h, n), dim=1)

    def forward(self, v_mem, current_input=None):
        if current_input is None:
            return v_mem

        if self.dynamics_mode == 'train_scale':
            return self._forward_train_scale(v_mem, current_input)

        v, m, h, n = torch.chunk(v_mem, self.state_multiplier, dim=1)
        v_rate = torch.clamp(v, min=-100.0, max=100.0)
        m = m.clamp(0.0, 1.0)
        h = h.clamp(0.0, 1.0)
        n = n.clamp(0.0, 1.0)

        i_ext = current_input * self.get_input_scale()
        i_na = self.g_na * (m ** 3) * h * (v_rate - self.e_na)
        i_k = self.g_k * (n ** 4) * (v_rate - self.e_k)
        i_l = self.g_l * (v_rate - self.e_l)

        tau = self.get_tau()
        dv_dt = (i_ext - i_na - i_k - i_l) / (self.c_m * tau)
        dm_dt = self._alpha_m(v_rate) * (1.0 - m) - self._beta_m(v_rate) * m
        dh_dt = self._alpha_h(v_rate) * (1.0 - h) - self._beta_h(v_rate) * h
        dn_dt = self._alpha_n(v_rate) * (1.0 - n) - self._beta_n(v_rate) * n

        spike_voltage = v + self.spike_update_dt * dv_dt if self.spike_after_update else v
        if self.normalize_spike:
            denom = self.v_threshold - self.v_rest
            if abs(denom) < 1e-6:
                denom = 1.0
            normalized_voltage = (spike_voltage - self.v_rest) / denom
            spike_input = self._apply_spike_affine(normalized_voltage) - self.threshold
        else:
            spike_input = spike_voltage - self.v_threshold
        spike = self.surrogate_f(spike_input, self.surrogate_grad_scale)
        if self.reset_strength != 0.0:
            spike_reset = spike.detach()
            dv_dt = dv_dt + (self.reset_strength / tau) * spike_reset * (self.v_rest - v)
        dstate_dt = torch.cat((dv_dt, dm_dt, dh_dt, dn_dt), dim=1)
        return dstate_dt, spike

    def _forward_train_scale(self, v_mem, current_input):
        z, m, h, n = torch.chunk(v_mem, self.state_multiplier, dim=1)
        z_rate = torch.clamp(z, min=-4.0, max=4.0)
        m = m.clamp(0.0, 1.0)
        h = h.clamp(0.0, 1.0)
        n = n.clamp(0.0, 1.0)

        voltage_span = self.v_threshold - self.v_rest
        if abs(voltage_span) < 1e-6:
            voltage_span = 1.0
        bio_v = self.v_rest + z_rate * voltage_span
        bio_v = torch.clamp(bio_v, min=-100.0, max=100.0)

        tau = self.get_tau()
        input_current = current_input * self.get_input_scale()
        dz_lif = (-z + input_current) / tau

        i_ext = input_current * voltage_span
        i_na = self.g_na * (m ** 3) * h * (bio_v - self.e_na)
        i_k = self.g_k * (n ** 4) * (bio_v - self.e_k)
        i_l = self.g_l * (bio_v - self.e_l)
        dz_hh = (i_ext - i_na - i_k - i_l) / (self.c_m * tau * voltage_span)

        mix = min(max(float(self.train_mix), 0.0), 1.0)
        dz_no_reset = (1.0 - mix) * dz_lif + mix * dz_hh

        dm_dt = self.gate_rate_scale * (self._alpha_m(bio_v) * (1.0 - m) - self._beta_m(bio_v) * m)
        dh_dt = self.gate_rate_scale * (self._alpha_h(bio_v) * (1.0 - h) - self._beta_h(bio_v) * h)
        dn_dt = self.gate_rate_scale * (self._alpha_n(bio_v) * (1.0 - n) - self._beta_n(bio_v) * n)

        spike_voltage = z + self.spike_update_dt * dz_no_reset if self.spike_after_update else z
        spike_input = self._apply_spike_affine(spike_voltage) - self.threshold
        spike = self.surrogate_f(spike_input, self.surrogate_grad_scale)

        dz_dt = dz_no_reset
        if self.reset_strength != 0.0:
            dz_dt = dz_dt - (self.reset_strength / tau) * spike.detach() * self.threshold
        dstate_dt = torch.cat((dz_dt, dm_dt, dh_dt, dn_dt), dim=1)
        return dstate_dt, spike

    def extra_repr(self):
        return (
            f'v_rest={self.v_rest}, v_threshold={self.v_threshold}, '
            f'input_scale={float(self.get_input_scale().detach().cpu()):.4g}, c_m={self.c_m}, '
            f'normalize_spike={self.normalize_spike}, '
            f'learn_input_scale={self.learn_input_scale}, '
            f'learn_spike_affine={self.learn_spike_affine}, '
            f'spike_after_update={self.spike_after_update}, '
            f'spike_update_dt={self.spike_update_dt}, '
            f'reset_strength={self.reset_strength}, '
            f'dynamics_mode={self.dynamics_mode}, '
            f'gate_rate_scale={self.gate_rate_scale}, '
            f'train_mix={self.train_mix}'
        )


class IzhikevichNeuron(BaseNeuron):
    """Izhikevich neuron with packed ODE state [v, u]."""

    state_multiplier = 2
    state_names = ("v", "u")

    def __init__(self,
                 tau=1.0,
                 threshold=1.0,
                 surrogate_grad_scale=5.0,
                 surrogate_opt="arctan_surrogate",
                 tau_learnable=False,
                 a=0.02,
                 b=0.2,
                 c=-65.0,
                 d=8.0,
                 input_scale=5.0,
                 v_threshold=30.0,
                 reset_strength=1.0,
                 normalize_spike=False,
                 learn_input_scale=False,
                 learn_spike_affine=False,
                 spike_after_update=False,
                 spike_update_dt=1.0,
                 dynamics_mode='bio',
                 train_mix=1.0):
        super().__init__(
            tau=tau,
            threshold=threshold,
            surrogate_grad_scale=surrogate_grad_scale,
            surrogate_opt=surrogate_opt,
            tau_learnable=tau_learnable,
        )
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_threshold = v_threshold
        self.reset_strength = reset_strength
        self.normalize_spike = normalize_spike
        self.learn_input_scale = learn_input_scale
        self.learn_spike_affine = learn_spike_affine
        self.spike_after_update = spike_after_update
        self.spike_update_dt = spike_update_dt
        if dynamics_mode not in {'bio', 'train_scale'}:
            raise ValueError(f"Unsupported Izhikevich dynamics_mode: {dynamics_mode}")
        self.dynamics_mode = dynamics_mode
        self.train_mix = train_mix
        input_scale = max(float(input_scale), 1e-8)
        if learn_input_scale:
            self.log_input_scale = nn.Parameter(torch.tensor(math.log(input_scale), dtype=torch.float32))
        else:
            self.register_buffer('input_scale_buffer', torch.tensor(input_scale, dtype=torch.float32), persistent=False)
        if learn_spike_affine:
            self.spike_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.spike_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.register_buffer('spike_gain_buffer', torch.tensor(1.0, dtype=torch.float32), persistent=False)
            self.register_buffer('spike_bias_buffer', torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def get_input_scale(self):
        if self.learn_input_scale:
            return self.log_input_scale.exp()
        return self.input_scale_buffer

    def _apply_spike_affine(self, normalized_voltage):
        if self.learn_spike_affine:
            return normalized_voltage * self.spike_gain + self.spike_bias
        return normalized_voltage * self.spike_gain_buffer + self.spike_bias_buffer

    def initial_state(self, output_shape, device=None, dtype=None):
        if self.dynamics_mode == 'train_scale':
            v = torch.zeros(output_shape, device=device, dtype=dtype)
            u = torch.full(output_shape, self.b * self.c, device=device, dtype=dtype)
        else:
            v = torch.full(output_shape, self.c, device=device, dtype=dtype)
            u = self.b * v
        return torch.cat((v, u), dim=1)

    def forward(self, v_mem, current_input=None):
        if current_input is None:
            return v_mem
        if self.dynamics_mode == 'train_scale':
            return self._forward_train_scale(v_mem, current_input)

        v, u = torch.chunk(v_mem, self.state_multiplier, dim=1)
        v_rate = torch.clamp(v, min=-100.0, max=100.0)
        tau = self.get_tau()
        i_ext = current_input * self.get_input_scale()
        dv_no_reset = (0.04 * (v_rate ** 2) + 5.0 * v_rate + 140.0 - u + i_ext) / tau
        du_no_reset = self.a * (self.b * v_rate - u) / tau

        spike_voltage = v + self.spike_update_dt * dv_no_reset if self.spike_after_update else v
        if self.normalize_spike:
            denom = self.v_threshold - self.c
            if abs(denom) < 1e-6:
                denom = 1.0
            normalized_voltage = (spike_voltage - self.c) / denom
            spike_input = self._apply_spike_affine(normalized_voltage) - self.threshold
        else:
            spike_input = spike_voltage - self.v_threshold
        spike = self.surrogate_f(spike_input, self.surrogate_grad_scale)
        spike_reset = spike.detach()

        reset_strength = self.reset_strength / tau
        dv_dt = dv_no_reset + reset_strength * spike_reset * (self.c - v)
        du_dt = du_no_reset + reset_strength * spike_reset * self.d

        dstate_dt = torch.cat((dv_dt, du_dt), dim=1)
        return dstate_dt, spike

    def _forward_train_scale(self, v_mem, current_input):
        z, u = torch.chunk(v_mem, self.state_multiplier, dim=1)
        z_rate = torch.clamp(z, min=-4.0, max=4.0)
        tau = self.get_tau()
        i_ext = current_input * self.get_input_scale()
        voltage_span = self.v_threshold - self.c
        if abs(voltage_span) < 1e-6:
            voltage_span = 1.0
        bio_v = torch.clamp(self.c + z_rate * voltage_span, min=-100.0, max=100.0)

        dz_lif = (-z + i_ext) / tau
        dv_izh = (0.04 * (bio_v ** 2) + 5.0 * bio_v + 140.0 - u + i_ext * voltage_span) / tau
        dz_izh = dv_izh / voltage_span
        mix = min(max(float(self.train_mix), 0.0), 1.0)
        dz_no_reset = (1.0 - mix) * dz_lif + mix * dz_izh
        du_dt = self.a * (self.b * bio_v - u) / tau

        spike_voltage = z + self.spike_update_dt * dz_no_reset if self.spike_after_update else z
        spike_input = self._apply_spike_affine(spike_voltage) - self.threshold
        spike = self.surrogate_f(spike_input, self.surrogate_grad_scale)
        dz_dt = dz_no_reset - (self.reset_strength / tau) * spike.detach() * self.threshold
        du_dt = du_dt + (self.reset_strength / tau) * spike.detach() * self.d
        return torch.cat((dz_dt, du_dt), dim=1), spike

    def extra_repr(self):
        return (
            f'a={self.a}, b={self.b}, c={self.c}, d={self.d}, '
            f'v_threshold={self.v_threshold}, '
            f'input_scale={float(self.get_input_scale().detach().cpu()):.4g}, '
            f'normalize_spike={self.normalize_spike}, '
            f'learn_input_scale={self.learn_input_scale}, '
            f'learn_spike_affine={self.learn_spike_affine}, '
            f'spike_after_update={self.spike_after_update}, '
            f'spike_update_dt={self.spike_update_dt}, '
            f'dynamics_mode={self.dynamics_mode}, '
            f'train_mix={self.train_mix}'
        )


class ResonateFireNeuron(BaseNeuron):
    """Resonate-and-Fire neuron with packed ODE state [v, u]."""

    state_multiplier = 2
    state_names = ("v", "u")

    def __init__(self,
                 tau=1.0,
                 threshold=1.0,
                 surrogate_grad_scale=5.0,
                 surrogate_opt="arctan_surrogate",
                 tau_learnable=False,
                 input_scale=1.0,
                 reset_strength=1.0,
                 alpha=0.2,
                 omega=1.0,
                 learn_input_scale=False,
                 learn_spike_affine=False,
                 spike_after_update=False,
                 spike_update_dt=1.0,
                 dynamics_mode='bio',
                 train_mix=1.0):
        super().__init__(
            tau=tau,
            threshold=threshold,
            surrogate_grad_scale=surrogate_grad_scale,
            surrogate_opt=surrogate_opt,
            tau_learnable=tau_learnable,
        )
        if dynamics_mode not in {'bio', 'train_scale'}:
            raise ValueError(f"Unsupported RAF dynamics_mode: {dynamics_mode}")
        self.reset_strength = reset_strength
        self.alpha = alpha
        self.omega = omega
        self.learn_input_scale = learn_input_scale
        self.learn_spike_affine = learn_spike_affine
        self.spike_after_update = spike_after_update
        self.spike_update_dt = spike_update_dt
        self.dynamics_mode = dynamics_mode
        self.train_mix = train_mix
        self._init_calibration(input_scale, learn_input_scale, learn_spike_affine)

    def _init_calibration(self, input_scale, learn_input_scale, learn_spike_affine):
        input_scale = max(float(input_scale), 1e-8)
        if learn_input_scale:
            self.log_input_scale = nn.Parameter(torch.tensor(math.log(input_scale), dtype=torch.float32))
        else:
            self.register_buffer('input_scale_buffer', torch.tensor(input_scale, dtype=torch.float32), persistent=False)
        if learn_spike_affine:
            self.spike_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.spike_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.register_buffer('spike_gain_buffer', torch.tensor(1.0, dtype=torch.float32), persistent=False)
            self.register_buffer('spike_bias_buffer', torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def get_input_scale(self):
        if self.learn_input_scale:
            return self.log_input_scale.exp()
        return self.input_scale_buffer

    def _apply_spike_affine(self, normalized_voltage):
        if self.learn_spike_affine:
            return normalized_voltage * self.spike_gain + self.spike_bias
        return normalized_voltage * self.spike_gain_buffer + self.spike_bias_buffer

    def initial_state(self, output_shape, device=None, dtype=None):
        return torch.zeros((output_shape[0], output_shape[1] * self.state_multiplier, *output_shape[2:]),
                           device=device, dtype=dtype)

    def forward(self, v_mem, current_input=None):
        if current_input is None:
            return v_mem

        v, u = torch.chunk(v_mem, self.state_multiplier, dim=1)
        tau = self.get_tau()
        i_ext = current_input * self.get_input_scale()
        v_rate = torch.clamp(v, min=-4.0, max=4.0)
        u_rate = torch.clamp(u, min=-4.0, max=4.0)

        dv_lif = (-v + i_ext) / tau
        dv_raf = u_rate / tau
        du_dt = (-2.0 * self.alpha * u_rate - (self.omega ** 2) * v_rate + i_ext) / tau

        if self.dynamics_mode == 'train_scale':
            mix = min(max(float(self.train_mix), 0.0), 1.0)
            dv_no_reset = (1.0 - mix) * dv_lif + mix * dv_raf
            spike_voltage = v + self.spike_update_dt * dv_no_reset if self.spike_after_update else v
            spike_input = self._apply_spike_affine(spike_voltage) - self.threshold
        else:
            dv_no_reset = dv_raf
            spike_voltage = v + self.spike_update_dt * dv_no_reset if self.spike_after_update else v
            spike_input = spike_voltage - self.threshold
        spike = self.surrogate_f(spike_input, self.surrogate_grad_scale)
        dv_dt = dv_no_reset - (self.reset_strength / tau) * spike.detach() * self.threshold
        return torch.cat((dv_dt, du_dt), dim=1), spike

    def extra_repr(self):
        return (
            f'input_scale={float(self.get_input_scale().detach().cpu()):.4g}, '
            f'alpha={self.alpha}, omega={self.omega}, reset_strength={self.reset_strength}, '
            f'learn_input_scale={self.learn_input_scale}, learn_spike_affine={self.learn_spike_affine}, '
            f'spike_after_update={self.spike_after_update}, spike_update_dt={self.spike_update_dt}, '
            f'dynamics_mode={self.dynamics_mode}, train_mix={self.train_mix}'
        )


class HindmarshRoseNeuron(BaseNeuron):
    """Hindmarsh-Rose neuron with packed ODE state [x, y, z]."""

    state_multiplier = 3

    def __init__(self,
                 tau=1.0,
                 threshold=1.0,
                 surrogate_grad_scale=5.0,
                 surrogate_opt="arctan_surrogate",
                 tau_learnable=False,
                 input_scale=1.0,
                 reset_strength=1.0,
                 a=1.0,
                 b=3.0,
                 c=1.0,
                 d=5.0,
                 r=0.006,
                 s=4.0,
                 x_rest=-1.6,
                 x_threshold=1.0,
                 learn_input_scale=False,
                 learn_spike_affine=False,
                 spike_after_update=False,
                 spike_update_dt=1.0,
                 dynamics_mode='bio',
                 train_mix=1.0):
        super().__init__(
            tau=tau,
            threshold=threshold,
            surrogate_grad_scale=surrogate_grad_scale,
            surrogate_opt=surrogate_opt,
            tau_learnable=tau_learnable,
        )
        if dynamics_mode not in {'bio', 'train_scale'}:
            raise ValueError(f"Unsupported HR dynamics_mode: {dynamics_mode}")
        self.reset_strength = reset_strength
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.r = r
        self.s = s
        self.x_rest = x_rest
        self.x_threshold = x_threshold
        self.learn_input_scale = learn_input_scale
        self.learn_spike_affine = learn_spike_affine
        self.spike_after_update = spike_after_update
        self.spike_update_dt = spike_update_dt
        self.dynamics_mode = dynamics_mode
        self.train_mix = train_mix
        ResonateFireNeuron._init_calibration(self, input_scale, learn_input_scale, learn_spike_affine)

    get_input_scale = ResonateFireNeuron.get_input_scale
    _apply_spike_affine = ResonateFireNeuron._apply_spike_affine

    def initial_state(self, output_shape, device=None, dtype=None):
        if self.dynamics_mode == 'train_scale':
            x = torch.zeros(output_shape, device=device, dtype=dtype)
        else:
            x = torch.full(output_shape, self.x_rest, device=device, dtype=dtype)
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)
        return torch.cat((x, y, z), dim=1)

    def forward(self, v_mem, current_input=None):
        if current_input is None:
            return v_mem

        x, y, z = torch.chunk(v_mem, self.state_multiplier, dim=1)
        tau = self.get_tau()
        i_ext = current_input * self.get_input_scale()

        if self.dynamics_mode == 'train_scale':
            x_rate = torch.clamp(x, min=-4.0, max=4.0)
            y_rate = torch.clamp(y, min=-4.0, max=4.0)
            z_rate = torch.clamp(z, min=-4.0, max=4.0)
            dx_lif = (-x + i_ext) / tau
            dx_hr = (y_rate - self.a * (x_rate ** 3) + self.b * (x_rate ** 2) - z_rate + i_ext) / tau
            mix = min(max(float(self.train_mix), 0.0), 1.0)
            dx_no_reset = (1.0 - mix) * dx_lif + mix * dx_hr
            spike_voltage = x + self.spike_update_dt * dx_no_reset if self.spike_after_update else x
            spike_input = self._apply_spike_affine(spike_voltage) - self.threshold
        else:
            x_rate = torch.clamp(x, min=-5.0, max=5.0)
            y_rate = torch.clamp(y, min=-20.0, max=20.0)
            z_rate = torch.clamp(z, min=-20.0, max=20.0)
            dx_no_reset = (y_rate - self.a * (x_rate ** 3) + self.b * (x_rate ** 2) - z_rate + i_ext) / tau
            spike_voltage = x + self.spike_update_dt * dx_no_reset if self.spike_after_update else x
            denom = self.x_threshold - self.x_rest
            if abs(denom) < 1e-6:
                denom = 1.0
            spike_input = (spike_voltage - self.x_rest) / denom - self.threshold
        dy_dt = (self.c - self.d * (x_rate ** 2) - y_rate) / tau
        dz_dt = self.r * (self.s * (x_rate - self.x_rest) - z_rate) / tau
        spike = self.surrogate_f(spike_input, self.surrogate_grad_scale)
        dx_dt = dx_no_reset - (self.reset_strength / tau) * spike.detach() * self.threshold
        return torch.cat((dx_dt, dy_dt, dz_dt), dim=1), spike

    def extra_repr(self):
        return (
            f'input_scale={float(self.get_input_scale().detach().cpu()):.4g}, '
            f'a={self.a}, b={self.b}, c={self.c}, d={self.d}, r={self.r}, s={self.s}, '
            f'x_rest={self.x_rest}, x_threshold={self.x_threshold}, reset_strength={self.reset_strength}, '
            f'learn_input_scale={self.learn_input_scale}, learn_spike_affine={self.learn_spike_affine}, '
            f'spike_after_update={self.spike_after_update}, spike_update_dt={self.spike_update_dt}, '
            f'dynamics_mode={self.dynamics_mode}, train_mix={self.train_mix}'
        )


class MorrisLecarNeuron(BaseNeuron):
    """Morris-Lecar neuron with packed ODE state [v, w]."""

    state_multiplier = 2

    def __init__(self,
                 tau=1.0,
                 threshold=1.0,
                 surrogate_grad_scale=5.0,
                 surrogate_opt="arctan_surrogate",
                 tau_learnable=False,
                 input_scale=1.0,
                 reset_strength=1.0,
                 v_rest=-60.0,
                 v_threshold=-20.0,
                 c_m=20.0,
                 g_ca=4.4,
                 g_k=8.0,
                 g_l=2.0,
                 v_ca=120.0,
                 v_k=-84.0,
                 v_l=-60.0,
                 v1=-1.2,
                 v2=18.0,
                 v3=2.0,
                 v4=30.0,
                 phi=0.04,
                 learn_input_scale=False,
                 learn_spike_affine=False,
                 spike_after_update=False,
                 spike_update_dt=1.0,
                 dynamics_mode='bio',
                 train_mix=1.0):
        super().__init__(
            tau=tau,
            threshold=threshold,
            surrogate_grad_scale=surrogate_grad_scale,
            surrogate_opt=surrogate_opt,
            tau_learnable=tau_learnable,
        )
        if dynamics_mode not in {'bio', 'train_scale'}:
            raise ValueError(f"Unsupported ML dynamics_mode: {dynamics_mode}")
        self.reset_strength = reset_strength
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.c_m = c_m
        self.g_ca = g_ca
        self.g_k = g_k
        self.g_l = g_l
        self.v_ca = v_ca
        self.v_k = v_k
        self.v_l = v_l
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4
        self.phi = phi
        self.learn_input_scale = learn_input_scale
        self.learn_spike_affine = learn_spike_affine
        self.spike_after_update = spike_after_update
        self.spike_update_dt = spike_update_dt
        self.dynamics_mode = dynamics_mode
        self.train_mix = train_mix
        ResonateFireNeuron._init_calibration(self, input_scale, learn_input_scale, learn_spike_affine)

    get_input_scale = ResonateFireNeuron.get_input_scale
    _apply_spike_affine = ResonateFireNeuron._apply_spike_affine

    def _m_inf(self, v):
        return 0.5 * (1.0 + torch.tanh((v - self.v1) / self.v2))

    def _w_inf(self, v):
        return 0.5 * (1.0 + torch.tanh((v - self.v3) / self.v4))

    def _lambda_w(self, v):
        return torch.cosh((v - self.v3) / (2.0 * self.v4))

    def initial_state(self, output_shape, device=None, dtype=None):
        if self.dynamics_mode == 'train_scale':
            v = torch.zeros(output_shape, device=device, dtype=dtype)
            bio_v = torch.full(output_shape, self.v_rest, device=device, dtype=dtype)
        else:
            v = torch.full(output_shape, self.v_rest, device=device, dtype=dtype)
            bio_v = v
        w = self._w_inf(bio_v)
        return torch.cat((v, w), dim=1)

    def forward(self, v_mem, current_input=None):
        if current_input is None:
            return v_mem

        v, w = torch.chunk(v_mem, self.state_multiplier, dim=1)
        tau = self.get_tau()
        i_ext = current_input * self.get_input_scale()
        w_rate = w.clamp(0.0, 1.0)

        voltage_span = self.v_threshold - self.v_rest
        if abs(voltage_span) < 1e-6:
            voltage_span = 1.0
        if self.dynamics_mode == 'train_scale':
            z_rate = torch.clamp(v, min=-4.0, max=4.0)
            bio_v = torch.clamp(self.v_rest + z_rate * voltage_span, min=-100.0, max=100.0)
            dv_lif = (-v + i_ext) / tau
            i_ca = self.g_ca * self._m_inf(bio_v) * (bio_v - self.v_ca)
            i_k = self.g_k * w_rate * (bio_v - self.v_k)
            i_l = self.g_l * (bio_v - self.v_l)
            dv_ml = (i_ext * voltage_span - i_ca - i_k - i_l) / (self.c_m * tau * voltage_span)
            mix = min(max(float(self.train_mix), 0.0), 1.0)
            dv_no_reset = (1.0 - mix) * dv_lif + mix * dv_ml
            spike_voltage = v + self.spike_update_dt * dv_no_reset if self.spike_after_update else v
            spike_input = self._apply_spike_affine(spike_voltage) - self.threshold
        else:
            bio_v = torch.clamp(v, min=-100.0, max=100.0)
            i_ca = self.g_ca * self._m_inf(bio_v) * (bio_v - self.v_ca)
            i_k = self.g_k * w_rate * (bio_v - self.v_k)
            i_l = self.g_l * (bio_v - self.v_l)
            dv_no_reset = (i_ext - i_ca - i_k - i_l) / (self.c_m * tau)
            spike_voltage = v + self.spike_update_dt * dv_no_reset if self.spike_after_update else v
            spike_input = (spike_voltage - self.v_rest) / voltage_span - self.threshold
        dw_dt = self.phi * self._lambda_w(bio_v) * (self._w_inf(bio_v) - w_rate) / tau
        spike = self.surrogate_f(spike_input, self.surrogate_grad_scale)
        dv_dt = dv_no_reset - (self.reset_strength / tau) * spike.detach() * self.threshold
        return torch.cat((dv_dt, dw_dt), dim=1), spike

    def extra_repr(self):
        return (
            f'input_scale={float(self.get_input_scale().detach().cpu()):.4g}, '
            f'v_rest={self.v_rest}, v_threshold={self.v_threshold}, c_m={self.c_m}, '
            f'learn_input_scale={self.learn_input_scale}, learn_spike_affine={self.learn_spike_affine}, '
            f'spike_after_update={self.spike_after_update}, spike_update_dt={self.spike_update_dt}, '
            f'dynamics_mode={self.dynamics_mode}, train_mix={self.train_mix}'
        )


class DirectEulerNeuronBase(nn.Module):
    """Direct multi-step Euler adapter for packed-state biological ODE cells."""

    def __init__(self, cell, step_mode='m', step_size=0.1, backend='torch', store_state_seq=False):
        super().__init__()
        if step_mode not in {'s', 'm'}:
            raise ValueError(f"Unsupported step_mode: {step_mode}")
        if backend not in {'torch', 'cupy'}:
            raise ValueError(f"Unsupported backend: {backend}")
        self.cell = cell
        self.step_mode = step_mode
        self.step_size = step_size
        self.backend = backend
        self.store_state_seq = store_state_seq
        self.state = None
        self.state_seq = None

    @property
    def state_multiplier(self):
        return getattr(self.cell, 'state_multiplier', 1)

    def reset(self):
        self.state = None
        self.state_seq = None

    def _ensure_state(self, current_input):
        expected_channels = current_input.shape[1] * self.state_multiplier
        if (
            self.state is None
            or self.state.shape[0] != current_input.shape[0]
            or self.state.shape[1] != expected_channels
            or self.state.shape[2:] != current_input.shape[2:]
            or self.state.device != current_input.device
            or self.state.dtype != current_input.dtype
        ):
            if hasattr(self.cell, 'initial_state'):
                self.state = self.cell.initial_state(
                    current_input.shape,
                    device=current_input.device,
                    dtype=current_input.dtype,
                )
            else:
                self.state = torch.zeros(
                    (current_input.shape[0], expected_channels, *current_input.shape[2:]),
                    device=current_input.device,
                    dtype=current_input.dtype,
                )

    def forward(self, x):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        if self.step_mode == 'm':
            return self.multi_step_forward(x)
        raise ValueError(f"Unsupported step_mode: {self.step_mode}")

    def single_step_forward(self, current_input):
        if self.backend == 'cupy':
            raise NotImplementedError(
                f"{self.__class__.__name__} backend='cupy' is not implemented yet. "
                "Use backend='torch' until the verified CuPy backward kernels are added."
            )
        self._ensure_state(current_input)
        dstate_dt, spike = self.cell(self.state, current_input)
        self.state = self.state + self.step_size * dstate_dt
        return spike

    def multi_step_forward(self, x_seq):
        if x_seq.dim() < 3:
            raise ValueError(f"expected x_seq with shape [T, N, ...], got {tuple(x_seq.shape)}")
        if self.backend == 'cupy':
            return self._multi_step_forward_cupy(x_seq)
        spikes = []
        states = [] if self.store_state_seq else None
        for t in range(x_seq.shape[0]):
            spike = self.single_step_forward(x_seq[t])
            spikes.append(spike)
            if states is not None:
                states.append(self.state)
        if states is not None:
            self.state_seq = torch.stack(states)
        return torch.stack(spikes)

    def _multi_step_forward_cupy(self, x_seq):
        if self.step_mode != 'm':
            raise NotImplementedError("CuPy backend is only supported for multi-step direct Euler")
        if not x_seq.is_cuda:
            raise ValueError("CuPy backend requires CUDA tensors")
        if getattr(self.cell, 'dynamics_mode', None) != 'train_scale':
            raise NotImplementedError("CuPy direct Euler currently supports only dynamics_mode='train_scale'")
        if abs(float(getattr(self.cell, 'train_mix', 1.0))) > 1e-12:
            raise NotImplementedError("CuPy direct Euler currently supports only train_mix=0.0")
        if self.store_state_seq:
            raise NotImplementedError("CuPy direct Euler does not support store_state_seq=True")
        try:
            from .direct_euler_cupy import lif_like_euler_cupy
        except ImportError as exc:
            raise RuntimeError(f"CuPy direct Euler backend is unavailable: {exc}") from exc

        original_dtype = x_seq.dtype
        x_float = x_seq.float().contiguous()
        self._ensure_state(x_float[0])
        z = torch.chunk(self.state.float(), self.state_multiplier, dim=1)[0]
        z_shape = z.shape
        x_flat = x_float.flatten(1)
        z_flat = z.flatten()
        tau = self.cell.get_tau()
        if isinstance(tau, torch.Tensor):
            tau = float(tau.detach().cpu())
        input_scale = self.cell.get_input_scale().to(device=x_float.device, dtype=x_float.dtype)
        if getattr(self.cell, 'learn_spike_affine', False):
            spike_gain = self.cell.spike_gain.to(device=x_float.device, dtype=x_float.dtype)
            spike_bias = self.cell.spike_bias.to(device=x_float.device, dtype=x_float.dtype)
        else:
            spike_gain = self.cell.spike_gain_buffer.to(device=x_float.device, dtype=x_float.dtype)
            spike_bias = self.cell.spike_bias_buffer.to(device=x_float.device, dtype=x_float.dtype)
        spike_flat, z_final_flat = lif_like_euler_cupy(
            x_flat,
            z_flat,
            input_scale,
            spike_gain,
            spike_bias,
            float(tau),
            float(self.cell.threshold),
            float(self.step_size),
            float(getattr(self.cell, 'reset_strength', 1.0)),
            float(self.cell.surrogate_grad_scale),
            float(getattr(self.cell, 'spike_update_dt', 1.0)),
        )
        z_final = z_final_flat.reshape(z_shape)
        if self.state_multiplier == 1:
            self.state = z_final
        else:
            chunks = list(torch.chunk(self.state.float(), self.state_multiplier, dim=1))
            chunks[0] = z_final
            self.state = torch.cat(chunks, dim=1)
        return spike_flat.reshape_as(x_float).to(original_dtype)

    def get_input_scale(self):
        return self.cell.get_input_scale()

    @property
    def learn_input_scale(self):
        return getattr(self.cell, 'learn_input_scale', False)

    @property
    def learn_spike_affine(self):
        return getattr(self.cell, 'learn_spike_affine', False)

    @property
    def dynamics_mode(self):
        return getattr(self.cell, 'dynamics_mode', 'NA')

    @property
    def train_mix(self):
        return getattr(self.cell, 'train_mix', float('nan'))

    def extra_repr(self):
        return (
            f'cell={self.cell.__class__.__name__}, step_mode={self.step_mode}, '
            f'step_size={self.step_size}, backend={self.backend}, '
            f'store_state_seq={self.store_state_seq}'
        )


class HHNeuronEuler(DirectEulerNeuronBase):
    def __init__(self, step_mode='m', step_size=0.1, backend='torch', store_state_seq=False, **kwargs):
        super().__init__(HHNeuron(**kwargs), step_mode=step_mode, step_size=step_size,
                         backend=backend, store_state_seq=store_state_seq)


class IzhikevichNeuronEuler(DirectEulerNeuronBase):
    def __init__(self, step_mode='m', step_size=0.1, backend='torch', store_state_seq=False, **kwargs):
        super().__init__(IzhikevichNeuron(**kwargs), step_mode=step_mode, step_size=step_size,
                         backend=backend, store_state_seq=store_state_seq)


class ResonateFireNeuronEuler(DirectEulerNeuronBase):
    def __init__(self, step_mode='m', step_size=0.1, backend='torch', store_state_seq=False, **kwargs):
        super().__init__(ResonateFireNeuron(**kwargs), step_mode=step_mode, step_size=step_size,
                         backend=backend, store_state_seq=store_state_seq)


class HindmarshRoseNeuronEuler(DirectEulerNeuronBase):
    def __init__(self, step_mode='m', step_size=0.1, backend='torch', store_state_seq=False, **kwargs):
        super().__init__(HindmarshRoseNeuron(**kwargs), step_mode=step_mode, step_size=step_size,
                         backend=backend, store_state_seq=store_state_seq)


class MorrisLecarNeuronEuler(DirectEulerNeuronBase):
    def __init__(self, step_mode='m', step_size=0.1, backend='torch', store_state_seq=False, **kwargs):
        super().__init__(MorrisLecarNeuron(**kwargs), step_mode=step_mode, step_size=step_size,
                         backend=backend, store_state_seq=store_state_seq)


class LIFNeuronFDE(BaseNeuron):
    """
    统一的分数阶微分方程LIF神经元，支持GL、PREDICTOR、TRAP三种计算方法
    已优化：支持渐进式梯度detach以加速反向传播
    """
    def __init__(self,
                 tau=0.5,
                 threshold=1.0,
                 surrogate_alpha=4.0,
                 surrogate_function='sigmoid',
                 tau_learnable=False,
                 method='gl',  # 'gl', 'predictor', 'trap'
                 beta=0.5,
                 beta_learnable=False,
                 batchsize=32,
                 T=16,
                 step_size=1.0,
                 memory=-1,
                 name='',
                 step_mode='s',
                 v_reset=0.,
                 store_v_seq=False,
                 backend='torch',
                 detach_grad_steps=1,
                 reset_mode=None,
                 detach_reset=False,
                 decay_input=True,
                 multi_beta=None,
                 multi_coefficient=None,
                 learn_coefficient=True,
                 coefficient_constraint='none',
                 shared_multi_coefficient=None,
                 coefficient_update_interval=1,
                 glmulti_history_grad='truncated',
                 history_grad_mode='approx',
                 vo_solver='abm',
                 vo_reset_in_drift=False,
                 vo_residual_mix=0.0,
                 attn_dim=4,
                 attn_temperature=1.0,
                 attn_mode='per_sample',
                 attn_init='orthogonal_zero_score',
                 attn_stats_interval=32,
                 # 保持向后兼容的参数
                 surrogate_grad_scale=None,
                 surrogate_opt=None):
        """
        统一的分数阶微分方程LIF神经元，支持单步和多步模式
        
        Args:
            tau: 膜电位时间常数
            threshold: 阈值电压
            surrogate_alpha: 替代梯度缩放因子
            surrogate_function: 替代函数类型 ('sigmoid', 'arctan', 'piecewise_linear', 'gaussian')
            tau_learnable: tau是否可学习
            method: FDE计算方法 ('gl', 'predictor', 'trap')
            beta: 分数阶参数 (0-1之间)
            beta_learnable: beta是否可学习
            batchsize: 批次大小
            T: 时间步数
            step_size: 步长
            memory: 内存窗口大小
            name: 神经元名称
            step_mode: 步进模式，'s'为单步，'m'为多步
            v_reset: 重置电压
            store_v_seq: 是否存储电压序列
            backend: 后端类型，'torch'或'cupy'
            detach_grad_steps: 多步模式下保留梯度的时间步数，1=只保留最后1步（最快），-1=保留所有步（完全等价）
            reset_mode: 重置模式 ('hard' 或 'soft')
            detach_reset: 是否在重置时detach梯度
            decay_input: 控制输入是否经过tau衰减
            multi_beta: glmulti方法的多个分数阶参数
            multi_coefficient: glmulti方法的多个中间系数
            learn_coefficient: glmulti方法的中间系数是否可学习
            
            # 向后兼容参数（已废弃，请使用surrogate_alpha和surrogate_function）
            surrogate_grad_scale: 废弃，使用surrogate_alpha代替
            surrogate_opt: 废弃，使用surrogate_function代替
        """
        # 处理向后兼容
        if surrogate_grad_scale is not None:
            surrogate_alpha = surrogate_grad_scale
        if surrogate_opt is not None:
            # 转换旧的surrogate_opt到新的surrogate_function
            opt_to_func = {
                'sigmoid_surrogate': 'sigmoid',
                'arctan_surrogate': 'arctan',
                'piecewise_linear_surrogate': 'piecewise_linear',
                'gaussian_surrogate': 'gaussian'
            }
            surrogate_function = opt_to_func.get(surrogate_opt, 'sigmoid')
        
        super().__init__(
            tau=tau,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
            surrogate_function=surrogate_function,
            tau_learnable=tau_learnable
        )
        
        # 验证method参数
        if method not in ['gl', 'gl1', 'fagl1', 'vogl1', 'predictor', 'predictor1', 'vopredictor',
                          'trap', 'trap1', 'euler', 'glmulti']:
            raise ValueError(
                f"不支持的method: {method}. 支持的方法: 'gl', 'gl1', 'predictor', "
                "'predictor1', 'vopredictor', 'fagl1', 'vogl1', 'trap', 'trap1', 'euler', 'glmulti'"
            )
        
        self.method = method
        self.vo_solver = vo_solver
        self.vo_reset_in_drift = bool(vo_reset_in_drift)
        self.vo_residual_mix = float(vo_residual_mix)
        self.attn_dim = int(attn_dim)
        self.attn_temperature = float(attn_temperature)
        self.attn_mode = str(attn_mode)
        self.attn_init = str(attn_init)
        self.attn_stats_interval = int(attn_stats_interval)
        if self.attn_dim <= 0:
            raise ValueError('attn_dim must be positive')
        if self.attn_temperature <= 0.0:
            raise ValueError('attn_temperature must be positive')
        if self.attn_mode not in ('batch_shared', 'per_sample'):
            raise ValueError("attn_mode must be 'batch_shared' or 'per_sample'")
        if self.attn_init not in ('zero_q', 'orthogonal_zero_score'):
            raise ValueError("attn_init must be 'zero_q' or 'orthogonal_zero_score'")
        if self.attn_stats_interval <= 0:
            raise ValueError('attn_stats_interval must be positive')
        if (self.method == 'fagl1' and self.attn_init == 'orthogonal_zero_score'
                and self.attn_dim < 4):
            raise ValueError('orthogonal_zero_score requires attn_dim >= 4')
        if self.vo_solver not in ('abm', 'l1', 'abm_pc'):
            raise ValueError(f'Unsupported variable-order solver: {self.vo_solver}')
        self.T = T
        self.name = name
        self.h = step_size
        
        # 存储初始beta值
        self.initial_beta = beta
        self.beta_learnable = beta_learnable
        
        # 设置beta为可学习参数或固定值
        if beta_learnable:
            # 创建可学习参数，初始化为对应于初始beta值的logit
            # 由于get_beta()使用sigmoid，我们需要计算sigmoid的反函数
            # beta = sigmoid(beta_param) => beta_param = logit(beta)
            beta_clamped = max(1e-6, min(1.0 - 1e-6, beta))  # 确保在有效范围内
            initial_logit = math.log(beta_clamped / (1.0 - beta_clamped))
            self.beta_param = nn.Parameter(torch.tensor(initial_logit), requires_grad=True)
        else:
            self.beta_param = None
            self.beta = beta
        if self.method == 'fagl1' and self.beta_learnable:
            raise ValueError("method='fagl1' uses a fixed fractional order")

        if self.method == 'fagl1':
            # Do not perturb the backbone RNG stream: paired GL1/FAGL1 runs must
            # start from identical convolutional weights under the same seed.
            attention_seed = 1729 + sum(ord(character) for character in str(name))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(attention_seed)
                self.fagl_q_proj = nn.Linear(2, self.attn_dim, bias=False)
                self.fagl_k_proj = nn.Linear(2, self.attn_dim, bias=False)
                if self.attn_init == 'zero_q':
                    nn.init.zeros_(self.fagl_q_proj.weight)
                    nn.init.xavier_uniform_(self.fagl_k_proj.weight)
                else:
                    # Q and K occupy disjoint output coordinates, hence Q^T K=0
                    # exactly and every initial attention row is uniform.  Unlike
                    # zero-Q initialization, both projections receive gradients on
                    # the first optimization step.
                    nn.init.zeros_(self.fagl_q_proj.weight)
                    nn.init.zeros_(self.fagl_k_proj.weight)
                    rows = torch.randperm(self.attn_dim)
                    signs = torch.randint(0, 2, (4,)).mul_(2).sub_(1)
                    gain = math.sqrt(4.0 / 3.0)
                    self.fagl_q_proj.weight.data[rows[0], 0] = gain * signs[0]
                    self.fagl_q_proj.weight.data[rows[1], 1] = gain * signs[1]
                    self.fagl_k_proj.weight.data[rows[2], 0] = gain * signs[2]
                    self.fagl_k_proj.weight.data[rows[3], 1] = gain * signs[3]
            self.fagl_q_proj._skip_direct_fde_init = True
            self.fagl_k_proj._skip_direct_fde_init = True
        else:
            self.fagl_q_proj = None
            self.fagl_k_proj = None
        self._last_attention_entropy = None
        self._fagl_stats_counter = 0
        self._fagl_static_cache = {}
        
        self.batchsize = batchsize
        self.step_mode = step_mode
        self.v_reset = v_reset
        self.store_v_seq = store_v_seq
        self.backend = backend
        self.detach_grad_steps = detach_grad_steps  # 保留梯度的时间步数
        
        # 重置和输入衰减配置 - surrogate_alpha已在BaseNeuron中设置
        self.detach_reset = detach_reset
        self.decay_input = decay_input

        self.learn_coefficient = bool(learn_coefficient)
        self.coefficient_constraint = str(coefficient_constraint)
        self.coefficient_update_interval = int(coefficient_update_interval)
        self.glmulti_history_grad = str(glmulti_history_grad)
        self.history_grad_mode = str(history_grad_mode)
        self._coefficient_forward_count = 0
        if self.coefficient_constraint not in ('none', 'softplus', 'softmax'):
            raise ValueError("coefficient_constraint must be 'none', 'softplus', or 'softmax'")
        if self.coefficient_update_interval <= 0:
            raise ValueError('coefficient_update_interval must be positive')
        if self.glmulti_history_grad not in ('truncated', 'full'):
            raise ValueError("glmulti_history_grad must be 'truncated' or 'full'")
        if self.history_grad_mode not in ('approx', 'full'):
            raise ValueError("history_grad_mode must be 'approx' or 'full'")
        if self.method == 'glmulti':
            if multi_beta is None or multi_coefficient is None:
                raise ValueError("method='glmulti' requires multi_beta and multi_coefficient.")
            if len(multi_beta) != len(multi_coefficient):
                raise ValueError(
                    f"multi_beta (len={len(multi_beta)}) and multi_coefficient "
                    f"(len={len(multi_coefficient)}) must have the same length"
                )
            multi_beta_tensor = torch.tensor(multi_beta, dtype=torch.float32)
            if torch.any((multi_beta_tensor <= 0.0) | (multi_beta_tensor > 1.0)):
                raise ValueError("All multi_beta values must be in (0, 1].")
            self.register_buffer('multi_beta', multi_beta_tensor)
            coefficient_tensor = torch.tensor(multi_coefficient, dtype=torch.float32)
            if self.learn_coefficient:
                if self.coefficient_constraint in ('softplus', 'softmax') and torch.any(coefficient_tensor <= 0.0):
                    raise ValueError(
                        f'{self.coefficient_constraint}-constrained coefficients must be initialized above zero.'
                    )
                if self.coefficient_constraint in ('softplus', 'softmax'):
                    if self.coefficient_constraint == 'softplus':
                        raw = torch.log(torch.expm1(coefficient_tensor.clamp_min(1e-6)))
                    else:
                        # Softmax is invariant to a common scale, so log-masses preserve
                        # the requested initial relative distribution.
                        raw = torch.log(coefficient_tensor)
                    if shared_multi_coefficient is not None:
                        if shared_multi_coefficient.shape != raw.shape:
                            raise ValueError('Shared coefficient parameter has the wrong shape.')
                        self.multi_coefficient_raw = shared_multi_coefficient
                    else:
                        self.multi_coefficient_raw = nn.Parameter(raw)
                    self.multi_coefficient = None
                else:
                    self.multi_coefficient = (
                        shared_multi_coefficient
                        if shared_multi_coefficient is not None
                        else nn.Parameter(coefficient_tensor)
                    )
                    self.multi_coefficient_raw = None
            else:
                self.register_buffer('multi_coefficient', coefficient_tensor)
                self.multi_coefficient_raw = None

            basis = torch.zeros(multi_beta_tensor.numel(), self.T + 1, dtype=torch.float32)
            basis[:, 0] = 1.0
            for lag in range(1, self.T + 1):
                basis[:, lag] = (1.0 - (1.0 + multi_beta_tensor) / lag) * basis[:, lag - 1]
            self.register_buffer('_glmulti_basis', basis)
        else:
            self.multi_beta = None
            self.multi_coefficient = None
            self.multi_coefficient_raw = None
            self._glmulti_basis = None
        
        # 重置模式：如果未指定，根据v_reset判断
        if reset_mode is None:
            if v_reset is None:
                self.reset_mode = 'soft'
            else:
                self.reset_mode = 'hard'
        else:
            self.reset_mode = reset_mode

        # Initialize these as None, will be set to proper device/dtype later
        self.weights = None  # 统一的权重系数
        self.h_beta = None

        self.mem_hist = None
        self.drift_hist = None
        self._last_drift = None
        if memory == -1:
            self.memory = None
        else:
            self.memory = memory

        self.k = 0
        self._shape = None
        self._device = None
        self._dtype = None
        self._initialized = False
        
        # 多步模式状态管理
        self.v = v_reset
        self.v_seq = None

    def reset(self):
        """重置神经元状态"""
        self.k = 0
        # 确保重置时分离计算图
        if isinstance(self.v, torch.Tensor):
            self.v = torch.full_like(self.v, self.v_reset).detach()
        else:
            self.v = self.v_reset
        if self.store_v_seq:
            self.v_seq = None
        # Clean self.mem_hist but keep initialization status
        if self.mem_hist is not None:
            # 完全重新创建张量以避免任何计算图连接
            self.mem_hist = torch.zeros(
                self.mem_hist.shape, 
                device=self.mem_hist.device, 
                dtype=self.mem_hist.dtype
            )
        if self.drift_hist is not None:
            self.drift_hist = torch.zeros_like(self.drift_hist)
        self._last_drift = None
        # 注意：不重置_initialized状态，避免重复初始化开销

    def _reset(self):
        """内部重置方法（保持向后兼容）"""
        self.reset()

    def neuronal_reset(self, v, spike):
        """
        神经元重置
        
        Args:
            v: 膜电位
            spike: 脉冲输出
            
        Returns:
            重置后的膜电位
        """
        if hasattr(self, 'detach_reset') and self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        
        if self.reset_mode == 'soft':
            # 软重置: V = V - spike * V_threshold
            v_new = v - spike_d * self.threshold
        else:
            # 硬重置: V = (1 - spike) * V + spike * v_reset
            v_new = (1.0 - spike_d) * v + spike_d * self.v_reset
        
        return v_new

    def get_beta(self):
        """获取当前beta值，确保在0-1范围内"""
        if self.beta_learnable:
            # 使用sigmoid函数将参数映射到(0, 1)范围
            raw_beta = torch.sigmoid(self.beta_param)
            # 将范围限制在(1e-6, 1-1e-6)，避免边界值问题
            beta = raw_beta.clamp(min=1e-6, max=1.0 - 1e-6)
            return beta
        else:
            return self.beta

    def _compute_weights(self, T, device, dtype):
        """根据method参数计算相应的权重系数"""
        if self.method == 'gl':
            return self._get_gl_weights(T, device, dtype)
        elif self.method == 'gl1':
            return self._get_gl_weights(T, device, dtype)  # GL1使用相同的权重
        elif self.method == 'fagl1':
            return None  # attention rows depend on the current input states
        elif self.method == 'glmulti':
            return None  # glmulti weights depend on learnable coefficients and are recomputed in forward
        elif self.method == 'trap':
            return self._get_trap_weights(T, device, dtype)
        elif self.method == 'trap1':
            return self._get_trap_weights(T, device, dtype)  # TRAP1使用相同的权重
        elif self.method == 'predictor':
            return self._get_predictor_weights(T, device, dtype)
        elif self.method == 'predictor1':
            return self._get_predictor_weights(T, device, dtype)  # PREDICTOR1使用相同的权重
        elif self.method == 'vopredictor':
            return None  # weights depend on the shared beta at each time step
        elif self.method == 'vogl1':
            return None  # GL coefficients depend on beta at each time step
        elif self.method == 'euler':
            return self._get_euler_weights(T, device, dtype)
        else:
            raise ValueError(f"不支持的method: {self.method}")

    def _get_gl_weights(self, T, device, dtype):
        """计算GL方法的权重系数（原_get_cj_方法）"""
        # 获取beta值用于权重计算，使用detach()避免inplace操作问题
        beta_tensor = self.get_beta()
        if isinstance(beta_tensor, torch.Tensor):
            beta_val = beta_tensor.detach()  # 阻止梯度流动到权重计算，避免inplace操作问题
        else:
            beta_val = beta_tensor
        c = torch.zeros(T + 1, device=device, dtype=dtype)
        c[0] = 1
        for j in range(1, T + 1):
            c[j] = (1 - (1 + beta_val) / j) * c[j - 1]
        c_rev = torch.flip(c, dims=[0])
        return c_rev

    def _get_glmulti_weights(self, T, device, dtype, coefficient=None):
        """计算多项分布式GL方法的聚合权重。"""
        if self.multi_beta is None:
            raise ValueError("glmulti requires multi_beta and multi_coefficient.")

        beta = self.multi_beta.to(device=device, dtype=dtype)
        if coefficient is None:
            coefficient = self.get_multi_coefficient()
        coefficient = coefficient.to(device=device, dtype=dtype)
        if beta.numel() != coefficient.numel():
            raise ValueError("multi_beta and multi_coefficient must have the same length.")

        if T != self.T:
            raise ValueError(f'glmulti was initialized for T={self.T}, got T={T}')
        c = self._glmulti_basis.to(device=device, dtype=dtype)

        h = torch.tensor(self.h, device=device, dtype=dtype)
        weighted_h = coefficient * torch.pow(h, -beta)
        c_tilde = torch.sum(weighted_h.unsqueeze(1) * c, dim=0)
        c_tilde_0_inv = 1.0 / c_tilde[0]
        c_tilde_rev = torch.flip(c_tilde, dims=[0])
        return c_tilde_rev, c_tilde_0_inv

    def get_multi_coefficient(self):
        """Return the effective distributed-order masses."""
        if self.method != 'glmulti':
            return None
        if self.coefficient_constraint == 'softplus' and self.learn_coefficient:
            return F.softplus(self.multi_coefficient_raw)
        if self.coefficient_constraint == 'softmax' and self.learn_coefficient:
            return torch.softmax(self.multi_coefficient_raw, dim=0)
        return self.multi_coefficient

    def _build_glmulti_schedule(self, T, device):
        """Collapse all distributed orders into one differentiable GL kernel."""
        coefficient = self.get_multi_coefficient()
        if torch.is_grad_enabled() and self.learn_coefficient:
            update_coefficient = (
                self._coefficient_forward_count % self.coefficient_update_interval == 0
            )
            self._coefficient_forward_count += 1
            if not update_coefficient:
                coefficient = coefficient.detach()
        c_rev, c0_inv = self._get_glmulti_weights(
            T, device, torch.float32, coefficient=coefficient
        )
        c = torch.flip(c_rev, dims=[0])
        weights = torch.zeros((T, T), device=device, dtype=torch.float32)
        for t in range(1, T):
            weights[t, :t] = torch.flip(c[1:t + 1], dims=[0]) * c0_inv
        return {
            'scale': c0_inv.expand(T),
            'weights': weights,
        }

    def _get_trap_weights(self, T, device, dtype):
        """计算梯形方法的权重系数 - L1-2 trapezoidal scheme"""
        # 获取beta值用于权重计算，使用detach()避免inplace操作问题
        beta_tensor = self.get_beta()
        if isinstance(beta_tensor, torch.Tensor):
            beta_val = beta_tensor.detach()  # 阻止梯度流动到权重计算，避免inplace操作问题
        else:
            beta_val = beta_tensor
        
        # L1-2 trapezoidal scheme 权重系数
        # w_0^{(α)} = 1
        # w_j^{(α)} = (j+1)^{1-α} - 2j^{1-α} + (j-1)^{1-α}, j ≥ 1
        w = torch.zeros(T + 1, device=device, dtype=dtype)
        w[0] = 1.0  # w_0 = 1
        
        alpha = 1.0 - beta_val  # 转换为标准分数阶参数
        
        for j in range(1, T + 1):
            if j == 1:
                # w_1 = 2^{1-α} - 2 * 1^{1-α} + 0^{1-α} = 2^{1-α} - 2
                w[j] = torch.pow(torch.tensor(2.0, device=device, dtype=dtype), alpha) - 2.0
            else:
                # w_j = (j+1)^{1-α} - 2j^{1-α} + (j-1)^{1-α}
                term1 = torch.pow(torch.tensor(j + 1.0, device=device, dtype=dtype), alpha)
                term2 = 2.0 * torch.pow(torch.tensor(j, device=device, dtype=dtype), alpha)
                term3 = torch.pow(torch.tensor(j - 1.0, device=device, dtype=dtype), alpha)
                w[j] = term1 - term2 + term3
        
        # 梯形方法使用逆序权重
        w_rev = torch.flip(w, dims=[0])
        return w_rev

    def _get_predictor_weights(self, T, device, dtype):
        """计算预测子方法的权重系数"""
        # 预测子方法的权重系数 w_j = j^(β-1) - (j-1)^(β-1)
        # 获取beta值用于权重计算，使用detach()避免inplace操作问题
        beta_tensor = self.get_beta()
        if isinstance(beta_tensor, torch.Tensor):
            beta_val = beta_tensor.detach()  # 阻止梯度流动到权重计算，避免inplace操作问题
        else:
            beta_val = beta_tensor
        w = torch.zeros(T + 1, device=device, dtype=dtype)
        w[0] = 1.0  # w_0 = 1
        
        for j in range(1, T + 1):
            if j == 1:
                w[j] = 1.0  # w_1 = 1
            else:
                # w_j = j^(β-1) - (j-1)^(β-1)
                w[j] = torch.pow(torch.tensor(j, dtype=dtype, device=device), beta_val - 1) - \
                       torch.pow(torch.tensor(j - 1, dtype=dtype, device=device), beta_val - 1)
        
        # 预测子方法使用逆序权重
        w_rev = torch.flip(w, dims=[0])
        return w_rev

    def _get_euler_weights(self, T, device, dtype):
        """欧拉方法不需要复杂权重系数，返回None"""
        # 欧拉方法是一阶方法，不需要历史权重
        return None

    def initialize(self, v_mem):
        """初始化方法"""
        self._shape = v_mem.shape
        self._device = v_mem.device
        self._dtype = v_mem.dtype
        
        # 动态更新batchsize以匹配实际输入
        self.batchsize = v_mem.shape[0]

        # 计算权重系数
        self.weights = self._compute_weights(self.T, self._device, self._dtype)
        
        # 对于euler方法，不需要h_beta参数
        if self.method in ('euler', 'glmulti', 'vopredictor', 'vogl1', 'fagl1'):
            self.h_beta = None
        else:
            # 使用detach确保h_beta不参与梯度计算
            beta_val = self.get_beta().detach() if isinstance(self.get_beta(), torch.Tensor) else self.get_beta()
            self.h_beta = torch.pow(
                torch.tensor(self.h, device=self._device, dtype=self._dtype),
                beta_val
            )

        # 使用更大的固定大小来容纳不同的batch size
        self.max_batchsize = self.batchsize
        if self.method == 'vopredictor':
            self.mem_hist = torch.empty(
                (self.T, self.max_batchsize) + self._shape[1:],
                device=self._device, dtype=self._dtype
            )
            self.drift_hist = torch.zeros(
                (self.T, self.max_batchsize) + self._shape[1:],
                device=self._device,
                dtype=self._dtype
            )
        else:
            self.mem_hist = torch.empty(
                (self.T, self.max_batchsize) + self._shape[1:],
                device=self._device,
                dtype=self._dtype
            )
        self._initialized = True
        return v_mem

    def v_float_to_tensor(self, x):
        """将浮点数膜电位转换为张量"""
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)
        elif isinstance(self.v, torch.Tensor):
            # 处理动态batch size：如果输入的batch size与当前膜电位不匹配，需要调整
            if self.v.shape[0] != x.shape[0]:
                # batch size变化了，需要调整膜电位大小
                if x.shape[0] < self.v.shape[0]:
                    # batch size变小了（如最后一个batch），截取前面的部分
                    self.v = self.v[:x.shape[0]].contiguous()
                else:
                    # batch size变大了，扩展膜电位（用v_reset填充新的batch）
                    extra_size = x.shape[0] - self.v.shape[0]
                    extra_v = torch.full((extra_size,) + self.v.shape[1:], 
                                        self.v_reset,
                                        device=self.v.device, 
                                        dtype=self.v.dtype)
                    self.v = torch.cat([self.v, extra_v], dim=0)

    def forward(self, x, beta_schedule=None):
        """前向传播，根据step_mode选择单步或多步"""
        if self.step_mode == 's':
            return self.single_step_forward(x, beta_override=beta_schedule)
        elif self.step_mode == 'm':
            return self.multi_step_forward(x, beta_schedule=beta_schedule)
        else:
            raise ValueError(f"不支持的step_mode: {self.step_mode}")
    
    def single_step_forward(self, current_input, beta_override=None):
        """单步前向传播"""
        # 如果未初始化，先初始化
        if not self._initialized:
            self.initialize(current_input)
            # 初始化膜电位状态
            self.v_float_to_tensor(current_input)
        
        # 确保膜电位已转换为张量
        self.v_float_to_tensor(current_input)

        # 根据method选择相应的前向传播方法
        if self.method == 'gl':
            v_mem_new, spike = self._forward_gl(self.v, current_input, self.k)
        elif self.method == 'gl1':
            v_mem_new, spike = self._forward_gl1(self.v, current_input, self.k)
        elif self.method == 'fagl1':
            raise RuntimeError("method='fagl1' requires multi-step input for causal attention")
        elif self.method == 'glmulti':
            v_mem_new, spike = self._forward_glmulti(self.v, current_input, self.k)
        elif self.method == 'trap':
            v_mem_new, spike = self._forward_trap(self.v, current_input, self.k)
        elif self.method == 'trap1':
            v_mem_new, spike = self._forward_trap1(self.v, current_input, self.k)
        elif self.method == 'predictor':
            v_mem_new, spike = self._forward_predictor(self.v, current_input, self.k)
        elif self.method == 'predictor1':
            v_mem_new, spike = self._forward_predictor1(self.v, current_input, self.k)
        elif self.method == 'vopredictor':
            if beta_override is None:
                raise ValueError("method='vopredictor' requires a shared beta value")
            v_mem_new, spike = self._forward_vopredictor(
                self.v, current_input, self.k, beta_override
            )
        elif self.method == 'vogl1':
            if beta_override is None:
                raise ValueError("method='vogl1' requires a shared beta value")
            v_mem_new, spike = self._forward_vogl1(
                self.v, current_input, self.k, beta_override
            )
        elif self.method == 'euler':
            v_mem_new, spike = self._forward_euler(self.v, current_input, self.k)
        else:
            raise ValueError(f"不支持的method: {self.method}")

        # 更新历史记录（确保不越界并处理动态batch size）
        mem_idx = self.k % self.T
        if self.mem_hist is not None and mem_idx < self.mem_hist.shape[0]:
            current_batch_size = v_mem_new.shape[0]
            # 只更新当前batch size对应的部分，避免尺寸不匹配
            if current_batch_size <= self.mem_hist.shape[1]:
                self.mem_hist[mem_idx, :current_batch_size] = v_mem_new.detach()  # 分离计算图，避免就地操作
            else:
                # 如果当前batch size超过了预分配的大小，需要重新分配内存
                self.mem_hist[mem_idx] = v_mem_new.detach()[:self.mem_hist.shape[1]]
        if self.method == 'vopredictor' and self.drift_hist is not None:
            current_batch_size = v_mem_new.shape[0]
            self.drift_hist[mem_idx, :current_batch_size] = self._last_drift[
                :current_batch_size
            ].detach()
        self.k += 1  # 递增时间步
        
        # 不再detach v，允许梯度跨时间步传播，这是标准的BPTT实现
        self.v = v_mem_new  # 保留梯度图，允许跨时间步的梯度传播

        return spike
    
    def multi_step_forward(self, x_seq, beta_schedule=None):
        """
        多步前向传播
        
        Args:
            x_seq: 输入序列，shape为[T, N, *]
            
        Returns:
            输出脉冲序列，shape为[T, N, *]
        """
        # Full-history fixed-order GL1 uses the fused sequence primitive so the
        # backward pass can scatter every future history contribution to the
        # exact source state.
        if self.method == 'gl1' and self.history_grad_mode == 'full':
            if beta_schedule is not None:
                raise ValueError("method='gl1' rejects beta_schedule")
            if self.backend == 'cupy' and x_seq.is_cuda:
                return self._multi_step_forward_gl1_full_cupy(x_seq)
            return self._multi_step_forward_gl1_full_torch(x_seq)

        # FAGL1 is fixed-order and builds causal, per-sample attention rows from x_seq.
        if self.method == 'fagl1':
            if beta_schedule is not None:
                raise ValueError("method='fagl1' uses a fixed order and rejects beta_schedule")
            if self.backend == 'cupy' and x_seq.is_cuda:
                return self._multi_step_forward_fagl1_cupy(x_seq)
            return self._multi_step_forward_fagl1_torch(x_seq)

        # CuPy 后端加速（支持 GL/GL1/TRAP/TRAP1/PREDICTOR/PREDICTOR1/EULER 方法和任意维度输入）
        if self.method in ('vopredictor', 'vogl1'):
            if beta_schedule is None:
                raise ValueError(f"method='{self.method}' requires beta_schedule")
            beta_values = beta_schedule['beta'] if isinstance(beta_schedule, dict) else beta_schedule
            if beta_values.shape[0] != x_seq.shape[0]:
                raise ValueError(
                    f'beta_schedule has {beta_values.shape[0]} time values, '
                    f'but input has {x_seq.shape[0]} time steps'
                )
            if self.backend == 'cupy' and x_seq.is_cuda and self.vo_solver == 'abm':
                if self.method == 'vogl1':
                    return self._multi_step_forward_vogl1_cupy(x_seq, beta_schedule)
                return self._multi_step_forward_vopredictor_cupy(x_seq, beta_schedule)
        if self.backend == 'cupy' and self.method == 'glmulti' and x_seq.is_cuda:
            return self._multi_step_forward_glmulti_cupy(x_seq)
        elif (self.backend == 'cupy' and self.method in ['gl', 'gl1', 'trap', 'trap1', 'predictor', 'predictor1', 'euler'] and x_seq.is_cuda):
            return self._multi_step_forward_cupy(x_seq)
        
        # 标准 Torch 后端
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        
        # 重置时间步计数器（重要：确保每次多步前向传播都从0开始）
        self.k = 0
        
        for t in range(T):
            if self.method in ('vopredictor', 'vogl1'):
                beta_values = beta_schedule['beta'] if isinstance(beta_schedule, dict) else beta_schedule
                beta_t = beta_values[t]
            else:
                beta_t = None
            y = self.single_step_forward(x_seq[t], beta_override=beta_t)
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        
        return torch.stack(y_seq)

    def _build_fagl1_weights(self, x_seq):
        """Return mean-one causal attention multiplied by fixed-order GL rows."""
        if self.fagl_q_proj is None or self.fagl_k_proj is None:
            raise RuntimeError('FAGL1 attention projections are not initialized')
        # Keep the state-conditioned attention path differentiable with respect to
        # X, as in the multiplicative FADE kernel.
        x_f32 = x_seq.float()
        reduce_dims = (tuple(range(1, x_f32.dim())) if self.attn_mode == 'batch_shared'
                       else tuple(range(2, x_f32.dim())))
        variance, mean = torch.var_mean(x_f32, dim=reduce_dims, unbiased=False)
        rms = torch.sqrt((variance + mean.square()).clamp_min(1e-8))
        descriptor = torch.stack((
            mean / (rms + 1e-6),
            torch.log(rms.clamp_min(1e-6)),
        ), dim=-1)
        query = self.fagl_q_proj(descriptor)
        key = self.fagl_k_proj(descriptor)
        if self.attn_mode == 'batch_shared':
            scores = torch.mm(query, key.transpose(0, 1))
        else:
            query_btd = query.permute(1, 0, 2)
            key_bdt = key.permute(1, 2, 0)
            scores = torch.bmm(query_btd, key_bdt).permute(1, 0, 2)
        scores = scores / (math.sqrt(float(self.attn_dim)) * self.attn_temperature)

        beta = self.get_beta()
        beta_value = float(beta.detach().item()) if isinstance(beta, torch.Tensor) else float(beta)
        T, B = x_seq.shape[:2]
        cache_key = (x_seq.device.type, x_seq.device.index, T, beta_value)
        static = self._fagl_static_cache.get(cache_key)
        if static is None:
            coefficient = 1.0
            gl_coefficients = [1.0]
            for lag_index in range(1, T + 1):
                coefficient *= 1.0 - (1.0 + beta_value) / float(lag_index)
                gl_coefficients.append(coefficient)
            gl = torch.tensor(gl_coefficients, device=x_seq.device, dtype=torch.float32)
            t_index = torch.arange(T, device=x_seq.device)
            i_index = torch.arange(T, device=x_seq.device)
            causal = i_index.unsqueeze(0) < t_index.unsqueeze(1)
            safe_causal = causal | ((t_index == 0)[:, None] & (i_index == 0)[None, :])
            lag = (t_index[:, None] - i_index[None, :]).clamp_min(0)
            static = {
                't_index': t_index,
                'causal': causal,
                'safe_causal': safe_causal,
                'gl_kernel': gl[lag] * causal,
                'entropy_normalizer': torch.log(t_index[2:].float()),
            }
            self._fagl_static_cache[cache_key] = static
        t_index = static['t_index']
        causal = static['causal']
        safe_causal = static['safe_causal']
        gl_kernel_2d = static['gl_kernel']
        if self.attn_mode == 'batch_shared':
            probability = torch.softmax(scores.masked_fill(~safe_causal, -torch.inf), dim=-1)
            probability = probability * causal
            weights = probability * t_index[:, None] * gl_kernel_2d
        else:
            probability = torch.softmax(
                scores.masked_fill(~safe_causal[:, None, :], -torch.inf), dim=-1)
            probability = probability * causal[:, None, :]
            weights = probability * t_index[:, None, None] * gl_kernel_2d[:, None, :]

        if T > 2 and self._fagl_stats_counter % self.attn_stats_interval == 0:
            with torch.no_grad():
                probability_stats = probability.detach()
                entropy = -(probability_stats.clamp_min(1e-8).log() * probability_stats).sum(dim=-1)
                normalizer = static['entropy_normalizer']
                if self.attn_mode == 'per_sample':
                    normalizer = normalizer[:, None]
                self._last_attention_entropy = (entropy[2:] / normalizer).mean()
        else:
            if T <= 2 and self._last_attention_entropy is None:
                self._last_attention_entropy = weights.new_tensor(1.0)
        self._fagl_stats_counter += 1
        return weights

    def get_attention_stats(self):
        if self.method != 'fagl1' or self._last_attention_entropy is None:
            return None
        return {'entropy': float(self._last_attention_entropy.cpu())}

    def _initial_voltage_flat(self, x_seq):
        x_flat = x_seq.flatten(1)
        if isinstance(self.v, torch.Tensor) and self.v.shape == x_seq[0].shape:
            return x_flat, self.v.flatten().float()
        return x_flat, torch.full_like(x_flat[0], float(self.v_reset), dtype=torch.float32)

    def _build_gl1_causal_weights(self, time_steps, device):
        """Return the fixed GL1 history matrix used by every neuron element."""
        beta = self.get_beta()
        beta_value = (float(beta.detach().item())
                      if isinstance(beta, torch.Tensor) else float(beta))
        coefficients = [1.0]
        for lag in range(1, time_steps + 1):
            coefficients.append(
                (1.0 - (1.0 + beta_value) / float(lag)) * coefficients[-1]
            )
        rows = torch.zeros(
            (time_steps, time_steps), device=device, dtype=torch.float32
        )
        for t in range(1, time_steps):
            rows[t, :t] = torch.tensor(
                coefficients[t:0:-1], device=device, dtype=torch.float32
            )
        return rows

    def _multi_step_forward_gl1_full_cupy(self, x_seq):
        from .fde_cupy_kernels_all import CUPY_AVAILABLE, LIFFDEFAGL1Function
        if not CUPY_AVAILABLE:
            raise RuntimeError('CuPy backend requested for full-history gl1 but CuPy is unavailable')
        x_flat, v_init = self._initial_voltage_flat(x_seq)
        weights = self._build_gl1_causal_weights(x_seq.shape[0], x_seq.device)
        beta = self.get_beta()
        beta_value = (float(beta.detach().item())
                      if isinstance(beta, torch.Tensor) else float(beta))
        h_beta = float(self.h) ** beta_value
        spike_flat = LIFFDEFAGL1Function.apply(
            x_flat, v_init, weights, float(self.get_tau()), h_beta,
            float(self.threshold), float(self.surrogate_grad_scale), x_seq.shape[1],
            True,
        )
        self.v = torch.full_like(x_seq[0], float(self.v_reset)).detach()
        self.k = x_seq.shape[0]
        return spike_flat.reshape_as(x_seq)

    def _multi_step_forward_gl1_full_torch(self, x_seq):
        from .fde_cupy_kernels_all import LIFFDEFAGL1TorchFunction
        x_flat, v_init = self._initial_voltage_flat(x_seq)
        weights = self._build_gl1_causal_weights(x_seq.shape[0], x_seq.device)
        beta = self.get_beta()
        beta_value = (float(beta.detach().item())
                      if isinstance(beta, torch.Tensor) else float(beta))
        h_beta = float(self.h) ** beta_value
        spike_flat = LIFFDEFAGL1TorchFunction.apply(
            x_flat, v_init, weights, float(self.get_tau()), h_beta,
            float(self.threshold), float(self.surrogate_grad_scale), x_seq.shape[1],
            True,
        )
        self.v = torch.full_like(x_seq[0], float(self.v_reset)).detach()
        self.k = x_seq.shape[0]
        return spike_flat.reshape_as(x_seq)

    def _multi_step_forward_fagl1_cupy(self, x_seq):
        from .fde_cupy_kernels_all import CUPY_AVAILABLE, LIFFDEFAGL1Function
        if not CUPY_AVAILABLE:
            raise RuntimeError('CuPy backend requested for fagl1 but CuPy is unavailable')
        weights = self._build_fagl1_weights(x_seq)
        x_flat, v_init = self._initial_voltage_flat(x_seq)
        beta = self.get_beta()
        beta_value = float(beta.detach().item()) if isinstance(beta, torch.Tensor) else float(beta)
        h_beta = float(self.h) ** beta_value
        spike_flat = LIFFDEFAGL1Function.apply(
            x_flat, v_init, weights, float(self.get_tau()), h_beta,
            float(self.threshold), float(self.surrogate_grad_scale), x_seq.shape[1],
            self.history_grad_mode == 'full',
        )
        self.v = torch.full_like(x_seq[0], float(self.v_reset)).detach()
        self.k = x_seq.shape[0]
        return spike_flat.reshape_as(x_seq)

    def _multi_step_forward_fagl1_torch(self, x_seq):
        from .fde_cupy_kernels_all import LIFFDEFAGL1TorchFunction
        weights = self._build_fagl1_weights(x_seq)
        x_flat, v_init = self._initial_voltage_flat(x_seq)
        beta = self.get_beta()
        beta_value = float(beta.detach().item()) if isinstance(beta, torch.Tensor) else float(beta)
        h_beta = float(self.h) ** beta_value
        spike_flat = LIFFDEFAGL1TorchFunction.apply(
            x_flat, v_init, weights, float(self.get_tau()), h_beta,
            float(self.threshold), float(self.surrogate_grad_scale), x_seq.shape[1],
            self.history_grad_mode == 'full',
        )
        self.v = torch.full_like(x_seq[0], float(self.v_reset)).detach()
        self.k = x_seq.shape[0]
        return spike_flat.reshape_as(x_seq)

    def _multi_step_forward_vogl1_cupy(self, x_seq, beta_schedule):
        from .fde_cupy_kernels_all import CUPY_AVAILABLE, LIFFDEVOGL1Function
        if not CUPY_AVAILABLE:
            raise RuntimeError('CuPy backend requested for vogl1 but CuPy is unavailable')
        if isinstance(beta_schedule, dict):
            scale = beta_schedule['scale']
            weights = beta_schedule['weights']
        else:
            terms = self.build_vogl1_schedule(beta_schedule, self.h)
            scale, weights = terms['scale'], terms['weights']
        x_flat = x_seq.flatten(1)
        if isinstance(self.v, torch.Tensor) and self.v.shape == x_seq[0].shape:
            v_init = self.v.flatten().to(dtype=torch.float32)
        else:
            v_init = torch.full_like(x_flat[0], float(self.v_reset), dtype=torch.float32)
        spike_flat = LIFFDEVOGL1Function.apply(
            x_flat, v_init, scale, weights, float(self.get_tau()),
            float(self.threshold), float(self.surrogate_grad_scale), x_seq.shape[1],
            self.history_grad_mode == 'full',
        )
        self.v = torch.full_like(x_seq[0], float(self.v_reset)).detach()
        self.k = x_seq.shape[0]
        return spike_flat.reshape_as(x_seq)

    def _multi_step_forward_glmulti_cupy(self, x_seq):
        """Fused distributed-order GL using the variable-GL CuPy primitive."""
        from .fde_cupy_kernels_all import CUPY_AVAILABLE, LIFFDEGLMultiFunction
        if not CUPY_AVAILABLE:
            raise RuntimeError('CuPy backend requested for glmulti but CuPy is unavailable')
        T = x_seq.shape[0]
        terms = self._build_glmulti_schedule(T, x_seq.device)
        x_flat, v_init = self._initial_voltage_flat(x_seq)
        spike_flat = LIFFDEGLMultiFunction.apply(
            x_flat, v_init, terms['scale'], terms['weights'], float(self.get_tau()),
            float(self.threshold), float(self.surrogate_grad_scale), x_seq.shape[1],
            self.glmulti_history_grad == 'full',
        )
        self.v = torch.full_like(x_seq[0], float(self.v_reset)).detach()
        self.k = T
        return spike_flat.reshape_as(x_seq)

    @staticmethod
    def build_vogl1_schedule(beta_schedule, step_size):
        """Build current-order GL rows and h**beta scales for shared beta."""
        beta = beta_schedule.to(dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
        T = beta.shape[0]
        h = torch.as_tensor(step_size, device=beta.device, dtype=torch.float32)
        scale = torch.pow(h, beta)
        rows = []
        for t in range(T):
            c = torch.ones_like(beta[t]).unsqueeze(0)
            coefficients = [c]
            for lag in range(1, t + 1):
                c = (1.0 - (1.0 + beta[t]) / float(lag)) * c
                coefficients.append(c)
            active = torch.cat(coefficients, dim=0)
            active = torch.flip(active[1:], dims=[0])
            if beta.dim() == 1:
                row = torch.nn.functional.pad(active, (0, T - active.shape[0]))
            else:
                row = torch.nn.functional.pad(active, (0, 0, 0, T - active.shape[0]))
                row = row.transpose(0, 1)
            rows.append(row)
        return {'beta': beta, 'scale': scale, 'weights': torch.stack(rows)}

    def _multi_step_forward_vopredictor_cupy(self, x_seq, beta_schedule):
        from .fde_cupy_kernels_all import (
            CUPY_AVAILABLE, LIFFDEVOPREDICTORFunction,
        )
        if not CUPY_AVAILABLE:
            raise RuntimeError('CuPy backend requested for vopredictor but CuPy is unavailable')

        T = x_seq.shape[0]
        if isinstance(beta_schedule, dict):
            scale = beta_schedule['scale']
            weights = beta_schedule['weights']
        else:
            terms = self.build_vopredictor_schedule(beta_schedule, self.h)
            scale = terms['scale']
            weights = terms['weights']

        x_flat = x_seq.flatten(1)
        if isinstance(self.v, torch.Tensor) and self.v.shape == x_seq[0].shape:
            v_init = self.v.flatten().to(dtype=torch.float32)
        else:
            v_init = torch.full_like(x_flat[0], float(self.v_reset), dtype=torch.float32)
        spike_flat = LIFFDEVOPREDICTORFunction.apply(
            x_flat, v_init, scale, weights, float(self.get_tau()),
            float(self.threshold), float(self.surrogate_grad_scale),
            self.vo_residual_mix, x_seq.shape[1],
        )
        self.v = torch.full_like(x_seq[0], float(self.v_reset)).detach()
        self.k = T
        return spike_flat.reshape_as(x_seq)

    @staticmethod
    def build_vopredictor_schedule(beta_schedule, step_size):
        """Build differentiable terms once for all neurons sharing a beta schedule."""
        beta = beta_schedule.to(dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
        T = beta.shape[0]
        h = torch.as_tensor(step_size, device=beta.device, dtype=torch.float32)
        scale = torch.pow(h, beta) / torch.exp(torch.lgamma(beta + 1.0))
        rows = []
        for t in range(T):
            if t == 0:
                active = (torch.empty(0, device=beta.device, dtype=torch.float32)
                          if beta.dim() == 1 else
                          torch.empty((0, beta.shape[1]), device=beta.device, dtype=torch.float32))
            elif t == 1:
                active = torch.ones_like(beta[t]).unsqueeze(0)
            else:
                lags = torch.arange(t, 1, -1, device=beta.device, dtype=torch.float32)
                exponent = beta[t] - 1.0
                if beta.dim() == 1:
                    active = torch.cat((torch.pow(lags, exponent) - torch.pow(lags - 1.0, exponent), torch.ones(1, device=beta.device)))
                else:
                    active = torch.cat((torch.pow(lags[:, None], exponent[None]) - torch.pow((lags - 1.0)[:, None], exponent[None]), torch.ones((1, beta.shape[1]), device=beta.device)), dim=0)
            if beta.dim() == 1:
                rows.append(torch.cat((active, torch.zeros(T - t, device=beta.device))))
            else:
                rows.append(torch.cat((active, torch.zeros((T - t, beta.shape[1]), device=beta.device)), dim=0).transpose(0, 1))
        return {'beta': beta_schedule, 'scale': scale, 'weights': torch.stack(rows)}
    
    def _multi_step_forward_cupy(self, x_seq):
        """
        使用 CuPy 加速的多步前向传播（支持 GL/TRAP/PREDICTOR）
        
        Args:
            x_seq: [T, N, *] 输入序列（任意维度）
            
        Returns:
            spike_seq: [T, N, *] 脉冲序列
        """
        # 尝试导入对应的CuPy Function（统一使用fde_cupy_kernels_all.py）
        try:
            # 使用统一的kernel文件（已移除beta_factor的标准实现）
            if self.method == 'gl':
                from .fde_cupy_kernels_all import LIFFDEGLFunction, CUPY_AVAILABLE
                CuPyFunction = LIFFDEGLFunction
            elif self.method == 'gl1':
                from .fde_cupy_kernels_all import LIFFDEGL1Function, CUPY_AVAILABLE
                CuPyFunction = LIFFDEGL1Function
            elif self.method == 'trap':
                from .fde_cupy_kernels_all import LIFFDETRAPFunction, CUPY_AVAILABLE
                CuPyFunction = LIFFDETRAPFunction
            elif self.method == 'trap1':
                from .fde_cupy_kernels_all import LIFFDETRAP1Function, CUPY_AVAILABLE
                CuPyFunction = LIFFDETRAP1Function
            elif self.method == 'predictor':
                from .fde_cupy_kernels_all import LIFFDEPREDICTORFunction, CUPY_AVAILABLE
                CuPyFunction = LIFFDEPREDICTORFunction
            elif self.method == 'predictor1':
                from .fde_cupy_kernels_all import LIFFDEPREDICTOR1Function, CUPY_AVAILABLE
                CuPyFunction = LIFFDEPREDICTOR1Function
            elif self.method == 'euler':
                from .fde_cupy_kernels_all import LIFEulerFunction, CUPY_AVAILABLE
                CuPyFunction = LIFEulerFunction
            else:
                print(f"Warning: CuPy backend not implemented for method={self.method}, falling back to torch")
                return self._multi_step_forward_torch(x_seq)
        except ImportError as e:
            print(f"Warning: CuPy kernels not found ({e}), falling back to torch backend")
            return self._multi_step_forward_torch(x_seq)
        
        if not CUPY_AVAILABLE:
            print("Warning: CuPy not available, falling back to torch backend")
            return self._multi_step_forward_torch(x_seq)
        
        # 保存原始形状
        original_shape = x_seq.shape  # [T, N, *]
        
        # Flatten 到 2D: [T, N*features]
        T = x_seq.shape[0]
        x_seq_flat = x_seq.flatten(1)  # [T, N*features]
        N_flat = x_seq_flat.shape[1]
        
        # 初始化
        if not self._initialized:
            self.initialize(x_seq[0])
        self.v_float_to_tensor(x_seq[0])
        
        # 确保 v 也被 flatten
        v_flat = self.v.flatten(0)  # [N*features]
        
        # 重置时间步
        self.k = 0
        
        # 获取参数
        tau = self.get_tau()
        if isinstance(tau, torch.Tensor):
            tau = tau.item()
        
        beta = self.get_beta()
        if isinstance(beta, torch.Tensor):
            beta = beta.item()
        
        # 根据不同方法调用对应的CuPy kernel
        if self.method == 'gl':
            spike_seq_flat, v_seq_flat = CuPyFunction.apply(
                x_seq_flat,  # [T, N*features]
                v_flat,      # [N*features]
                self.weights,  # [T+1]
                tau,
                beta,
                self.threshold,
                self.v_reset if self.v_reset is not None else 0.0,
                self.surrogate_grad_scale
            )
        elif self.method == 'gl1':
            spike_seq_flat, v_seq_flat = CuPyFunction.apply(
                x_seq_flat,  # [T, N*features]
                v_flat,      # [N*features]
                self.weights,  # [T+1] GL1使用相同的权重
                tau,
                beta,
                self.h,
                self.threshold,
                self.v_reset if self.v_reset is not None else 0.0,
                self.surrogate_grad_scale
            )
        elif self.method == 'trap':
            spike_seq_flat, v_seq_flat = CuPyFunction.apply(
                x_seq_flat,
                v_flat,
                self.weights,  # trap_weights
                tau,
                beta,
                self.threshold,
                self.v_reset if self.v_reset is not None else 0.0,
                self.surrogate_grad_scale
            )
        elif self.method == 'trap1':
            # TRAP1: 使用h_beta缩放因子的TRAP方法
            spike_seq_flat, v_seq_flat = CuPyFunction.apply(
                x_seq_flat,
                v_flat,
                self.weights,  # trap1使用相同的trap权重
                tau,
                beta,
                self.threshold,
                self.v_reset if self.v_reset is not None else 0.0,
                self.surrogate_grad_scale
            )
        elif self.method == 'predictor':
            # 计算gamma(beta+1)和h_beta
            beta_plus_one = beta + 1.0
            gamma_beta_plus_1 = math.exp(math.lgamma(beta_plus_one))
            h_beta = self.h ** beta
            
            spike_seq_flat, v_seq_flat = CuPyFunction.apply(
                x_seq_flat,
                v_flat,
                self.weights,  # predictor_weights
                tau,
                beta,
                self.threshold,
                self.v_reset if self.v_reset is not None else 0.0,
                self.surrogate_grad_scale,
                h_beta,
                gamma_beta_plus_1
            )
        elif self.method == 'predictor1':
            # PREDICTOR1: 计算gamma(beta+1)和h_beta
            beta_plus_one = beta + 1.0
            gamma_beta_plus_1 = math.exp(math.lgamma(beta_plus_one))
            h_beta = self.h ** beta
            
            spike_seq_flat, v_seq_flat = CuPyFunction.apply(
                x_seq_flat,
                v_flat,
                self.weights,  # predictor1使用相同的predictor权重
                tau,
                beta,
                self.threshold,
                self.v_reset if self.v_reset is not None else 0.0,
                self.surrogate_grad_scale,
                h_beta,
                gamma_beta_plus_1
            )
        elif self.method == 'euler':
            # Euler方法：标准LIF，无需权重和beta参数
            spike_seq_flat, v_seq_flat = CuPyFunction.apply(
                x_seq_flat,
                v_flat,
                tau,
                self.threshold,
                self.v_reset if self.v_reset is not None else 0.0,
                self.surrogate_grad_scale
            )
        
        # 恢复原始形状
        spike_seq = spike_seq_flat.reshape(original_shape)
        v_seq = v_seq_flat.reshape(original_shape)
        
        # 存储状态
        if self.store_v_seq:
            self.v_seq = v_seq
        
        # 更新膜电位（保持原始形状）
        self.v = v_seq[-1].clone()
        self.k = T
        
        return spike_seq
    
    def _multi_step_forward_torch(self, x_seq):
        """Torch 后端的多步前向传播（向后兼容）"""
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        
        self.k = 0
        
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        
        return torch.stack(y_seq)

    def _forward_gl(self, v_mem, current_input, k):
        """
        GL方法的核心计算逻辑 - 标准Grünwald-Letnikov实现
        
        支持 decay_input 参数：
        - decay_input=True:  dv = (-V + I) / τ  (标准 LIF)
        - decay_input=False: dv = -V / τ + I    (简化模型)
        """
        tau = self.get_tau()

        # LIF drift - 支持 decay_input 参数
        if self.decay_input:
            # 标准 LIF: 输入也经过 tau 衰减
            dv_no_reset = (-v_mem + current_input) / tau
        else:
            # 简化模型: 输入不衰减
            dv_no_reset = -v_mem / tau + current_input

        # Window calculation
        if self.memory is None:
            history_len = min(k, self.T)  # 确保不超过历史记录长度
            start = 0
        else:
            start = max(0, k - self.memory)
            history_len = min(k - start, self.T)  # 确保不超过历史记录长度

        # History contribution
        if history_len > 0 and k > 0:  # 添加 k > 0 检查
            # 计算实际的历史索引，考虑循环使用
            actual_indices = [(start + i) % self.T for i in range(history_len)]
            # 获取当前batch size以确保历史数据匹配
            current_batch_size = v_mem.shape[0]
            # 关键修复：强制detach历史数据，切断梯度传播链
            hist_data = torch.stack([self.mem_hist[idx, :current_batch_size].detach() for idx in actual_indices])
            
            w = self.weights[self.T - history_len:self.T].view(history_len, 1)
            win = hist_data.reshape(history_len, -1)
            hist = torch.matmul(w.T, win).reshape(v_mem.shape)
        else:
            hist = torch.zeros_like(v_mem)

        # Update voltage - 标准GL方法：直接使用dv_no_reset减去历史项
        v_post_charge = dv_no_reset - hist

        # Fire spikes
        spike = self.surrogate_f(
            v_post_charge - self.threshold,
            self.surrogate_grad_scale
        )

        # Apply reset
        v_mem_new = v_post_charge - (spike.detach() * self.threshold)

        return v_mem_new, spike

    def _forward_glmulti(self, v_mem, current_input, k):
        """
        多项分布式GL方法的核心计算逻辑。

        公式来自 the historical distributed-order implementation 的 distributed-order GL:
        y_{k+1} = (f_k - sum_m c_tilde[m] * y_{k+1-m}) / c_tilde[0]
        """
        tau = self.get_tau()

        if self.decay_input:
            dv_no_reset = (-v_mem + current_input) / tau
        else:
            dv_no_reset = -v_mem / tau + current_input

        if self.memory is None:
            history_len = min(k, self.T)
            start = 0
        else:
            start = max(0, k - self.memory)
            history_len = min(k - start, self.T)

        weights, c_tilde_0_inv = self._get_glmulti_weights(self.T, v_mem.device, v_mem.dtype)

        if history_len > 0 and k > 0:
            actual_indices = [(start + i) % self.T for i in range(history_len)]
            current_batch_size = v_mem.shape[0]
            hist_data = torch.stack([self.mem_hist[idx, :current_batch_size].detach() for idx in actual_indices])
            w = weights[self.T - history_len:self.T].view(history_len, 1)
            win = hist_data.reshape(history_len, -1)
            hist = torch.matmul(w.T, win).reshape(v_mem.shape)
        else:
            hist = torch.zeros_like(v_mem)

        v_post_charge = c_tilde_0_inv * (dv_no_reset - hist)

        spike = self.surrogate_f(
            v_post_charge - self.threshold,
            self.surrogate_grad_scale
        )

        v_mem_new = v_post_charge - (spike.detach() * self.threshold)

        return v_mem_new, spike

    def _forward_gl1(self, v_mem, current_input, k):
        """
        GL1方法的核心计算逻辑 - 带h_beta缩放因子的GL实现
        
        支持 decay_input 参数：
        - decay_input=True:  dv = (-V + I) / τ  (标准 LIF)
        - decay_input=False: dv = -V / τ + I    (简化模型)
        """
        tau = self.get_tau()

        # LIF drift - 支持 decay_input 参数
        if self.decay_input:
            # 标准 LIF: 输入也经过 tau 衰减
            dv_no_reset = (-v_mem + current_input) / tau
        else:
            # 简化模型: 输入不衰减
            dv_no_reset = -v_mem / tau + current_input

        # Window calculation
        if self.memory is None:
            history_len = min(k, self.T)  # 确保不超过历史记录长度
            start = 0
        else:
            start = max(0, k - self.memory)
            history_len = min(k - start, self.T)  # 确保不超过历史记录长度

        # History contribution
        if history_len > 0 and k > 0:  # 添加 k > 0 检查
            # 计算实际的历史索引，考虑循环使用
            actual_indices = [(start + i) % self.T for i in range(history_len)]
            # 获取当前batch size以确保历史数据匹配
            current_batch_size = v_mem.shape[0]
            # 关键修复：强制detach历史数据，切断梯度传播链
            hist_data = torch.stack([self.mem_hist[idx, :current_batch_size].detach() for idx in actual_indices])
            
            w = self.weights[self.T - history_len:self.T].view(history_len, 1)
            win = hist_data.reshape(history_len, -1)
            hist = torch.matmul(w.T, win).reshape(v_mem.shape)
        else:
            hist = torch.zeros_like(v_mem)

        # Update voltage - GL1方法：使用h_beta缩放因子
        # v_post_charge = h^β * dv_no_reset - hist
        v_post_charge = self.h_beta * dv_no_reset - hist

        # Fire spikes
        spike = self.surrogate_f(
            v_post_charge - self.threshold,
            self.surrogate_grad_scale
        )

        # Apply reset
        v_mem_new = v_post_charge - (spike.detach() * self.threshold)

        return v_mem_new, spike

    def _forward_vogl1(self, v_mem, current_input, k, beta_override):
        """Current-order GL1 update with differentiable externally shared beta."""
        tau = self.get_tau()
        if self.decay_input:
            drift = (-v_mem + current_input) / tau
        else:
            drift = -v_mem / tau + current_input

        beta = beta_override.to(device=v_mem.device, dtype=v_mem.dtype)
        beta = beta.clamp(min=1e-6, max=1.0 - 1e-6)
        beta_view = (beta.reshape((beta.shape[0],) + (1,) * (v_mem.dim() - 1))
                     if beta.dim() else beta)
        scale = torch.pow(torch.as_tensor(self.h, device=v_mem.device, dtype=v_mem.dtype), beta_view)

        if self.memory is None:
            start = 0
            history_len = min(k, self.T)
        else:
            start = max(0, k - self.memory)
            history_len = min(k - start, self.T)
        hist = torch.zeros_like(v_mem)
        coefficient = torch.ones_like(beta)
        coefficients = []
        for lag in range(1, history_len + 1):
            coefficient = (1.0 - (1.0 + beta) / float(lag)) * coefficient
            coefficients.append(coefficient)
        for offset, state_index in enumerate(range(start, start + history_len)):
            coeff = coefficients[history_len - 1 - offset]
            coeff_view = (coeff.reshape((coeff.shape[0],) + (1,) * (v_mem.dim() - 1))
                          if coeff.dim() else coeff)
            state = self.mem_hist[state_index % self.T, :v_mem.shape[0]].detach().clone()
            hist = hist + coeff_view * state

        v_post_charge = scale * drift - hist
        spike = self.surrogate_f(v_post_charge - self.threshold, self.surrogate_grad_scale)
        v_mem_new = v_post_charge - spike.detach() * self.threshold
        return v_mem_new, spike

    def _forward_trap(self, v_mem, current_input, k):
        """
        L1-2 Trapezoidal scheme的核心计算逻辑（第二个定义）
        
        支持 decay_input 参数：
        - decay_input=True:  dv = (-V + I) / τ  (标准 LIF)
        - decay_input=False: dv = -V / τ + I    (简化模型)
        """
        tau = self.get_tau()

        # LIF drift - 支持 decay_input 参数
        if self.decay_input:
            # 标准 LIF: 输入也经过 tau 衰减
            dv_no_reset = (-v_mem + current_input) / tau
        else:
            # 简化模型: 输入不衰减
            dv_no_reset = -v_mem / tau + current_input

        # Window calculation
        if self.memory is None:
            history_len = min(k, self.T)  # 确保不超过历史记录长度
            start = 0
        else:
            start = max(0, k - self.memory)
            history_len = min(k - start, self.T)  # 确保不超过历史记录长度

        # History contribution with L1-2 trapezoidal weights
        if history_len > 0 and k > 0:  # 添加 k > 0 检查
            # 计算实际的历史索引，考虑循环使用
            actual_indices = [(start + i) % self.T for i in range(history_len)]
            # 获取当前batch size以确保历史数据匹配
            current_batch_size = v_mem.shape[0]
            # 关键修复：强制detach历史数据，切断梯度传播链
            hist_data = torch.stack([self.mem_hist[idx, :current_batch_size].detach() for idx in actual_indices])
            
            w = self.weights[self.T - history_len:self.T].view(history_len, 1)
            
            # L1-2 trapezoidal scheme 使用正确的权重（无需首尾减半）
            win = hist_data.reshape(history_len, -1)
            hist = torch.matmul(w.T, win).reshape(v_mem.shape)
        else:
            hist = torch.zeros_like(v_mem)

        # L1-2 Trapezoidal scheme voltage update
        current_beta = self.get_beta()
        
        # 计算梯形方法的步长因子
        h_alpha = torch.pow(
            torch.tensor(1.0, device=v_mem.device, dtype=v_mem.dtype),
            current_beta
        )
        
        # L1-2梯形方法的电压更新：考虑当前和历史项的加权
        # v_{n+1} = v_n + h^β * [trap_factor * dv_current + history_contribution]
        trap_factor = 1.5 * current_beta + 0.5  # 梯形方法的特征系数
        
        v_post_charge = v_mem + h_alpha * (trap_factor * dv_no_reset - hist)

        # Fire spikes
        spike = self.surrogate_f(
            v_post_charge - self.threshold,
            self.surrogate_grad_scale
        )

        # Apply reset
        v_mem_new = v_post_charge - (spike.detach() * self.threshold)

        return v_mem_new, spike

    def _forward_predictor(self, v_mem, current_input, k):
        """
        预测子方法的核心计算逻辑（第二个定义）
        
        支持 decay_input 参数：
        - decay_input=True:  dv = (-V + I) / τ  (标准 LIF)
        - decay_input=False: dv = -V / τ + I    (简化模型)
        """
        tau = self.get_tau()

        # LIF drift at current time step - 支持 decay_input 参数
        if self.decay_input:
            # 标准 LIF: 输入也经过 tau 衰减
            dv_current = (-v_mem + current_input) / tau
        else:
            # 简化模型: 输入不衰减
            dv_current = -v_mem / tau + current_input

        # Window calculation
        if self.memory is None:
            history_len = min(k, self.T)  # 确保不超过历史记录长度
            start = 0
        else:
            start = max(0, k - self.memory)
            history_len = min(k - start, self.T)  # 确保不超过历史记录长度

        # History contribution with predictor method
        if history_len > 0 and k > 0:  # 添加 k > 0 检查
            # 计算实际的历史索引，考虑循环使用
            actual_indices = [(start + i) % self.T for i in range(history_len)]
            
            # 获取历史导数值（LIF drift的历史记录）
            hist_derivatives = []
            # 获取当前batch size以确保历史数据匹配
            current_batch_size = v_mem.shape[0]
            for idx in actual_indices:
                # 关键修复：强制detach历史数据，切断梯度传播链
                hist_v = self.mem_hist[idx, :current_batch_size].detach()
                # 重新计算历史时刻的导数（这里简化处理，实际可能需要存储历史输入）
                hist_dv = -hist_v / tau  # 简化：假设历史输入为0
                hist_derivatives.append(hist_dv)
            
            hist_data = torch.stack(hist_derivatives)
            
            # 获取预测子权重
            w = self.weights[self.T - history_len:self.T].view(history_len, 1)
            
            # 预测子方法的积分计算
            win = hist_data.reshape(history_len, -1)
            hist_integral = torch.matmul(w.T, win).reshape(v_mem.shape)
        else:
            hist_integral = torch.zeros_like(v_mem)

        # 预测子方法的电压更新公式
        # v_{n+1} = v_0 + (h^β/Γ(β+1)) * [w_0 * f_{n+1} + Σ(w_j * f_{n+1-j})]
        beta_plus_one = self.get_beta() + 1.0
        if isinstance(beta_plus_one, torch.Tensor):
            gamma_input = beta_plus_one.to(device=self._device, dtype=self._dtype)
        else:
            gamma_input = torch.tensor(beta_plus_one, device=self._device, dtype=self._dtype)
        gamma_beta_plus_1 = torch.exp(torch.lgamma(gamma_input))
        
        # 当前时刻权重 w_0 = 1
        w_current = self.weights[self.T]  # 最后一个元素是 w_0
        
        # 动态计算h_beta以支持可学习beta
        h_beta = torch.pow(
            torch.tensor(self.h, device=self._device, dtype=self._dtype),
            self.get_beta()
        )
        
        v_post_charge = v_mem + (h_beta / gamma_beta_plus_1) * \
                       (w_current * dv_current + hist_integral)

        # Fire spikes
        spike = self.surrogate_f(
            v_post_charge - self.threshold,
            self.surrogate_grad_scale
        )

        # Apply reset
        v_mem_new = v_post_charge - (spike.detach() * self.threshold)

        return v_mem_new, spike

    def _forward_vopredictor(self, v_mem, current_input, k, beta_override):
        """Predictor update with a differentiable externally shared beta."""
        tau = self.get_tau()
        if self.decay_input:
            dv_current = (-v_mem + current_input) / tau
        else:
            dv_current = -v_mem / tau + current_input

        if self.memory is None:
            start = 0
            history_len = min(k, self.T)
        else:
            start = max(0, k - self.memory)
            history_len = min(k - start, self.T)

        beta = beta_override.to(device=v_mem.device, dtype=v_mem.dtype)
        beta = beta.clamp(min=1e-6, max=1.0 - 1e-6)
        beta_view = beta.reshape((beta.shape[0],) + (1,) * (v_mem.dim() - 1)) if beta.dim() else beta

        if self.vo_solver == 'l1':
            n = k
            states = [torch.full_like(v_mem, float(self.v_reset))]
            if n:
                states.extend(self.mem_hist[i, :v_mem.shape[0]].detach().clone() for i in range(n))
            one_minus = 1.0 - beta
            coeffs = [torch.pow(beta.new_tensor(float(n + 1)), one_minus) -
                      torch.pow(beta.new_tensor(float(n)), one_minus)]
            for j in range(1, n + 1):
                p = float(n + 1 - j)
                coeffs.append(2.0 * torch.pow(beta.new_tensor(p), one_minus) -
                              torch.pow(beta.new_tensor(max(p - 1.0, 0.0)), one_minus) -
                              torch.pow(beta.new_tensor(p + 1.0), one_minus))
            state_sum = torch.zeros_like(v_mem)
            for coefficient, state in zip(coeffs, states):
                view = coefficient.reshape((coefficient.shape[0],) + (1,) * (v_mem.dim() - 1)) if coefficient.dim() else coefficient
                state_sum = state_sum + view * state
            h = torch.as_tensor(self.h, device=v_mem.device, dtype=v_mem.dtype)
            current_scale = torch.exp(torch.lgamma(2.0 - beta)) * torch.pow(h, beta)
            scale_view = current_scale.reshape((current_scale.shape[0],) + (1,) * (v_mem.dim() - 1)) if current_scale.dim() else current_scale
            v_post_charge = state_sum + scale_view * dv_current
            spike = self.surrogate_f(v_post_charge - self.threshold, self.surrogate_grad_scale)
            v_mem_new = v_post_charge - spike.detach() * self.threshold
            self._last_drift = (dv_current - spike.detach() * self.threshold / tau if self.vo_reset_in_drift else dv_current).detach()
            return v_mem_new, spike
        if history_len > 0:
            indices = [(start + i) % self.T for i in range(history_len)]
            hist_data = torch.stack([
                self.drift_hist[idx, :v_mem.shape[0]].detach() for idx in indices
            ])
            exponent = beta - 1.0
            if history_len == 1:
                weights = (torch.ones(1, device=v_mem.device, dtype=v_mem.dtype)
                           if beta.dim() == 0 else
                           torch.ones((1, beta.shape[0]), device=v_mem.device, dtype=v_mem.dtype))
            else:
                lags = torch.arange(
                    history_len, 1, -1, device=v_mem.device, dtype=v_mem.dtype
                )
                if beta.dim() == 0:
                    weights = torch.cat((torch.pow(lags, exponent) - torch.pow(lags - 1.0, exponent), torch.ones(1, device=v_mem.device, dtype=v_mem.dtype)))
                else:
                    weights = torch.cat((torch.pow(lags[:, None], exponent[None]) - torch.pow((lags - 1.0)[:, None], exponent[None]), torch.ones((1, beta.shape[0]), device=v_mem.device, dtype=v_mem.dtype)), dim=0)
            hist_integral = torch.sum(
                weights.reshape((history_len,) + (() if beta.dim() == 0 else (beta.shape[0],)) + (1,) * (v_mem.dim() - (0 if beta.dim() == 0 else 1))) * hist_data,
                dim=0,
            )
        else:
            hist_integral = torch.zeros_like(v_mem)

        h = torch.as_tensor(self.h, device=v_mem.device, dtype=v_mem.dtype)
        scale = torch.pow(h, beta) / torch.exp(torch.lgamma(beta + 1.0))
        scale_view = scale.reshape((scale.shape[0],) + (1,) * (v_mem.dim() - 1)) if scale.dim() else scale
        # Equation (9): reconstruct from x0; do not add the complete history to v_t again.
        v0 = torch.full_like(v_mem, float(self.v_reset))
        integral = dv_current + hist_integral
        if self.vo_solver == 'abm_pc':
            predictor = v0 + scale_view * integral
            drift_predictor = (-predictor + current_input) / tau if self.decay_input else -predictor / tau + current_input
            integral = hist_integral + 0.5 * (dv_current + drift_predictor)
        v_post_charge = v0 + scale_view * integral + self.vo_residual_mix * (v_mem - v0)
        spike = self.surrogate_f(
            v_post_charge - self.threshold, self.surrogate_grad_scale
        )
        v_mem_new = v_post_charge - spike.detach() * self.threshold
        self._last_drift = (dv_current - spike.detach() * self.threshold / tau if self.vo_reset_in_drift else dv_current).detach()
        return v_mem_new, spike

    def _forward_trap1(self, v_mem, current_input, k):
        """
        TRAP1方法的核心计算逻辑 - 带h_beta缩放因子的TRAP实现
        
        支持 decay_input 参数：
        - decay_input=True:  dv = (-V + I) / τ  (标准 LIF)
        - decay_input=False: dv = -V / τ + I    (简化模型)
        """
        tau = self.get_tau()

        # LIF drift - 支持 decay_input 参数
        if self.decay_input:
            # 标准 LIF: 输入也经过 tau 衰减
            dv_no_reset = (-v_mem + current_input) / tau
        else:
            # 简化模型: 输入不衰减
            dv_no_reset = -v_mem / tau + current_input

        # Window calculation
        if self.memory is None:
            history_len = min(k, self.T)
            start = 0
        else:
            start = max(0, k - self.memory)
            history_len = min(k - start, self.T)

        # History contribution with L1-2 trapezoidal weights
        if history_len > 0 and k > 0:
            actual_indices = [(start + i) % self.T for i in range(history_len)]
            current_batch_size = v_mem.shape[0]
            hist_data = torch.stack([self.mem_hist[idx, :current_batch_size].detach() for idx in actual_indices])
            
            w = self.weights[self.T - history_len:self.T].view(history_len, 1)
            win = hist_data.reshape(history_len, -1)
            hist = torch.matmul(w.T, win).reshape(v_mem.shape)
        else:
            hist = torch.zeros_like(v_mem)

        # TRAP1方法：使用h_beta缩放LIF drift项（关键改进）
        # v_{n+1} = v_n + h^β * dv_no_reset - hist
        v_post_charge = v_mem + self.h_beta * dv_no_reset - hist

        # Fire spikes
        spike = self.surrogate_f(
            v_post_charge - self.threshold,
            self.surrogate_grad_scale
        )

        # Apply reset
        v_mem_new = v_post_charge - (spike.detach() * self.threshold)

        return v_mem_new, spike

    def _forward_predictor1(self, v_mem, current_input, k):
        """
        PREDICTOR1方法的核心计算逻辑 - 带h_beta缩放因子的PREDICTOR实现
        
        支持 decay_input 参数：
        - decay_input=True:  dv = (-V + I) / τ  (标准 LIF)
        - decay_input=False: dv = -V / τ + I    (简化模型)
        """
        tau = self.get_tau()

        # LIF drift at current time step - 支持 decay_input 参数
        if self.decay_input:
            # 标准 LIF: 输入也经过 tau 衰减
            dv_current = (-v_mem + current_input) / tau
        else:
            # 简化模型: 输入不衰减
            dv_current = -v_mem / tau + current_input

        # Window calculation
        if self.memory is None:
            history_len = min(k, self.T)
            start = 0
        else:
            start = max(0, k - self.memory)
            history_len = min(k - start, self.T)

        # History contribution with predictor method
        if history_len > 0 and k > 0:
            actual_indices = [(start + i) % self.T for i in range(history_len)]
            
            # 获取历史导数值（LIF drift的历史记录）
            hist_derivatives = []
            current_batch_size = v_mem.shape[0]
            for idx in actual_indices:
                hist_v = self.mem_hist[idx, :current_batch_size].detach()
                # 重新计算历史时刻的导数
                hist_dv = -hist_v / tau
                hist_derivatives.append(hist_dv)
            
            hist_data = torch.stack(hist_derivatives)
            
            # 获取预测子权重
            w = self.weights[self.T - history_len:self.T].view(history_len, 1)
            
            # 预测子方法的积分计算
            win = hist_data.reshape(history_len, -1)
            hist_integral = torch.matmul(w.T, win).reshape(v_mem.shape)
        else:
            hist_integral = torch.zeros_like(v_mem)

        # PREDICTOR1方法：使用h_beta缩放（关键改进）
        # 原PREDICTOR: v_{n+1} = v_0 + (h^β/Γ(β+1)) * [w_0 * f_{n+1} + Σ(w_j * f_{n+1-j})]
        # PREDICTOR1: 简化为与GL1类似的形式 v_{n+1} = v_n + h^β * f_n - hist
        
        # 当前时刻权重 w_0 = 1
        w_current = self.weights[self.T]
        
        # PREDICTOR1简化形式：v_{n+1} = v_n + h^β * dv_current - hist_integral
        v_post_charge = v_mem + self.h_beta * (w_current * dv_current + hist_integral)

        # Fire spikes
        spike = self.surrogate_f(
            v_post_charge - self.threshold,
            self.surrogate_grad_scale
        )

        # Apply reset
        v_mem_new = v_post_charge - (spike.detach() * self.threshold)

        return v_mem_new, spike

    def _forward_euler(self, v_mem, current_input, k):
        """
        Euler方法的核心计算逻辑
        
        实现标准LIF神经元动力学:
        - decay_input=True:  V[t] = V[t-1] * (1 - 1/tau) + I[t] / tau (标准LIF)
        - decay_input=False: V[t] = V[t-1] * (1 - 1/tau) + I[t] (简化模型)
        - 发放: spike = Sigmoid(V[t] - V_threshold, alpha=4.0)
        - 重置: 根据reset_mode选择硬重置或软重置
        """
        tau = self.get_tau()
        dt = 1.0
        
        # LIF充电动力学 - 支持 decay_input 参数
        if self.decay_input:
            # decay_input=True: 标准LIF模型，输入也经过tau衰减
            # V[t] = V[t-1] * (1 - 1/tau) + I[t] / tau
            if self.v_reset is None or self.v_reset == 0.0:
                v_charged = v_mem * (1.0 - dt / tau) + current_input / tau
            else:
                v_charged = v_mem - (v_mem - self.v_reset) * dt / tau + current_input / tau
        else:
            # decay_input=False: 简化模型，输入不衰减
            # V[t] = V[t-1] * (1 - 1/tau) + I[t]
            if self.v_reset is None or self.v_reset == 0.0:
                v_charged = v_mem * (1.0 - dt / tau) + current_input
            else:
                v_charged = v_mem - (v_mem - self.v_reset) * dt / tau + current_input
        
        spike = self.surrogate_f(v_charged - self.threshold, self.surrogate_grad_scale)
        
        # CuPy Euler kernel uses soft reset with detached reset spike.
        v_mem_new = v_charged - spike.detach() * self.threshold
        
        return v_mem_new, spike

    def extra_repr(self):
        """额外的字符串表示"""
        tau_val = self.get_tau()
        if isinstance(tau_val, torch.Tensor):
            tau_val = tau_val.item()
        beta_val = self.get_beta()
        if isinstance(beta_val, torch.Tensor):
            beta_val = beta_val.item()
        return f'method={self.method}, tau={tau_val:.3f}, beta={beta_val:.3f}, threshold={self.threshold}, v_reset={self.v_reset}, step_mode={self.step_mode}'
#
# def rl_integrate_tuple(ode_func, y0_tuple, beta, t_grid: torch.Tensor, memory=None, compile_fn=False):
#     assert isinstance(y0_tuple, tuple) and len(y0_tuple) > 0
#     T = t_grid.numel()
#     assert T >= 2
#     device = y0_tuple[0].device
#     dtype  = y0_tuple[0].dtype
#     t_grid = t_grid.to(device=device, dtype=dtype)
#
#     h = (t_grid[-1] - t_grid[0]) / (T - 1)
#     h_beta = torch.pow(h, beta)
#
#     # GL coeffs c[0..T] once, and pre-reverse to avoid flip each step
#     c = torch.empty(T + 1, dtype=dtype, device=device)
#     c[0] = 1
#     for j in range(1, T + 1):
#         c[j] = (1 - (1 + beta) / j) * c[j - 1]
#     c_rev = torch.flip(c[1:], dims=[0])   # length T, l=1..T -> reversed
#
#     outs = [torch.empty((T,) + y0.shape, device=device, dtype=dtype) for y0 in y0_tuple]
#     for i, y0 in enumerate(y0_tuple):
#         outs[i][0].copy_(y0)
#
#     y = [y0.clone() for y0 in y0_tuple]
#
#     # Pre-allocate views/maps to avoid repeated .view calls
#     # flat_meta = []
#     # for i in range(len(y0_tuple)-1):
#     #     shape = y0_tuple[i].shape
#     #     numel = y0_tuple[i].numel()
#     #     flat_meta.append((shape, numel))
#
#     for k in range(1, T):
#         tk = t_grid[k]
#         dy = ode_func(tk, tuple(y))
#         assert isinstance(dy, tuple) and len(dy) == len(y)
#
#         # window
#         if memory is None:
#             history_len = k
#             start = 0
#         else:
#             start = max(0, k - memory)
#             history_len = k - start
#
#         if history_len > 0:
#             # lag weights for this window (reversed so we can use the slice directly)
#             # c[1..history_len] reversed == c_rev[T-history_len : T]
#             w = c_rev[T - history_len:T].reshape(history_len, 1)  # (L,1)
#         # update all integrated components in one python loop, inner math is vectorized
#         for i in range(len(y)-1):
#             if history_len > 0:
#                 # take window outs[i][start:k] (time, *spatial) -> (L, S)
#                 win = outs[i][start:k].reshape(history_len, -1).contiguous()
#                 hist = torch.matmul(w.T, win)  # (1,S)
#                 hist = hist.reshape(y[i].numel()).view_as(y[i])
#             else:
#                 hist = torch.zeros_like(y[i])
#             y[i] = h_beta * dy[i] - hist
#             outs[i][k].copy_(y[i])
#
#         # pass-through output (spike)
#         y[-1] = dy[-1]
#         outs[-1][k].copy_(y[-1])
#
#     return tuple(outs)


# class LIFNeuron(BaseNeuron):
#     def forward(self, v_mem, current_input=None):
#         if current_input is None:
#             return v_mem
#
#         tau = self.get_tau()
#
#         v_predict = v_mem + 1.0*current_input
#         # 基于预测的v计算spike
#         v_scaled = v_predict - self.threshold
#         spike_out = self.surrogate_f(v_scaled, self.surrogate_grad_scale)
#         spike_out_ = self.surrogate_f(v_mem - self.threshold, self.surrogate_grad_scale)
#         # Reset基于预测的spike
#         spike_reset = spike_out.detach()
#
#         # 返回包含reset的完整导数
#         dv_dt = (-v_mem + current_input) / tau - spike_reset * self.threshold / tau
#
#         return dv_dt, spike_out





class HardResetLIFNeuron(BaseNeuron):
    def __init__(self, tau=0.5, threshold=1.0, surrogate_grad_scale=5.0,
                 reset_strength=50.0):  # Add reset strength parameter
        super(HardResetLIFNeuron, self).__init__(tau, threshold, surrogate_grad_scale)
        self.reset_strength = reset_strength  # Controls the 'hardness' of reset
    def forward(self, v_mem, current_input=None):
        if current_input is None:
            return v_mem
        tau = self.get_tau()
        # Detect spike
        v_scaled = v_mem - self.threshold
        spike_out = self.surrogate_f(v_scaled, self.surrogate_grad_scale)
        # Standard LIF differential equation part
        standard_dv_dt = (-v_mem + current_input) / tau
        # Hard reset simulation: use large negative derivative impulse
        # The larger reset_strength, the closer the reset is to discrete hard reset
        reset_dv_dt = -spike_out * self.reset_strength * v_mem / tau
        # Combine derivatives
        dv_dt = standard_dv_dt #+ reset_dv_dt
        return dv_dt, spike_out
