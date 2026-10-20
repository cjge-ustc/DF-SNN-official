import torch
import torch.nn as nn
# from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint
from .neuron import BaseNeuron, LIFNeuron, IFNeuron
from .odefunc import ODEFunc, ODEFuncFromFX
from torch.fx.interpreter import Interpreter
from .solver import euler_integrate_tuple, gl_integrate_tuple, pred_integrate_tuple, l1_integrate_tuple, trap_integrate_tuple, vopred_time_integrate_tuple #euler_integrate_any,
from .solver import gl_integrate_tuple_per_state, pred_integrate_tuple_per_state, trap_integrate_tuple_per_state
from .solver import step_dynamics
from .solver import fdeint_adjoint
from .solver import (
    gl_integrate_tuple_cupy,
    pred_integrate_tuple_cupy,
    trap_integrate_tuple_cupy,
    l1_integrate_tuple_cupy,
    forward_glmulti_cupy,
)
# from .solver import fdeint_adjoint_optimized

euler_integrate_tuple_compiled = torch.compile(euler_integrate_tuple)
gl_integrate_tuple_compiled = torch.compile(gl_integrate_tuple)
trap_integrate_tuple_compiled = torch.compile(trap_integrate_tuple)
pred_integrate_tuple_compiled = torch.compile(pred_integrate_tuple)
l1_integrate_tuple_compiled = torch.compile(l1_integrate_tuple)
# fdeint_adjoint  = torch.compile(fdeint_adjoint)



SOLVERS = {
           "gl":gl_integrate_tuple_compiled,
           "trap":trap_integrate_tuple_compiled,
            "l1":l1_integrate_tuple_compiled,
            "pred":pred_integrate_tuple_compiled,
            "vopred":vopred_time_integrate_tuple,
}

PER_STATE_SOLVERS = {
           "gl": gl_integrate_tuple_per_state,
           "trap": trap_integrate_tuple_per_state,
           "pred": pred_integrate_tuple_per_state,
}

FDEINT_METHODS = {"gl", "pred", "vopred", "trap", "l1", "glmulti"}
FDEINT_ADJOINT_METHODS = {
    "gl-f",
    "gl-o",
    "pred-f",
    "pred-o",
    "trap-f",
    "trap-o",
    "l1-f",
    "l1-o",
    "glmulti",
}

CUPY_FDEINT_METHODS = {"gl", "pred", "trap", "l1", "glmulti"}
CUPY_ADJOINT_METHODS = {
    "gl-f",
    "gl-o",
    "pred-f",
    "pred-o",
    "trap-f",
    "trap-o",
    "l1-f",
    "l1-o",
    "glmulti",
}
# euler_integrate_any_compiled = torch.compile(euler_integrate_any)

def is_activation_layer(layer):
    """
    检查一个层是否为激活函数层

    参数:
        layer: 要检查的层

    返回:
        bool: 如果是激活层则返回True，否则返回False
    """
    # 根据类名检查 - 大多数PyTorch激活函数以特定后缀结尾
    layer_class_name = layer.__class__.__name__
    activation_keywords = ['ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'GELU', 'ELU', 'SELU',
                           'Softplus', 'Softsign', 'Softmin', 'Hardtanh', 'Hardsigmoid',
                           'Hardswish', 'LogSigmoid', 'PReLU', 'RReLU', 'SiLU', 'Mish', 'Threshold']

    if any(keyword in layer_class_name for keyword in activation_keywords):
        return True

    # 从nn.modules.activation模块检查
    if hasattr(layer, '__module__') and 'torch.nn.modules.activation' in layer.__module__:
        return True

    return False


def process_ann_layer(layer, neuron_cls, tau, threshold, surrogate_grad_scale, surrogate_opt, tau_learnable):
    """
    处理ANN模型中的一个层，确定它是何种类型并返回相应的处理结果

    参数:
        layer: 要处理的层
        neuron_cls: 神经元类
        tau: 时间常数
        threshold: 阈值
        surrogate_grad_scale: 替代梯度的缩放因子
        surrogate_opt: 替代梯度的类型

    返回:
        tuple: (layer, neuron, if_set_potential) 或 None（如果是激活层）
    """
    # 如果是激活层，直接跳过
    if is_activation_layer(layer):
        return None

    passive_layers = (
        nn.Flatten, nn.Unflatten, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d, nn.AvgPool1d, nn.AvgPool2d, nn.MaxPool1d, nn.MaxPool2d,
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Identity
    )

    spiking_layers = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
    )

    # 如果是被动层，直接添加，不需要神经元
    if isinstance(layer, passive_layers):
        return layer, None, False
    # 如果是脉冲层，添加层和相应的神经元
    elif isinstance(layer, spiking_layers):
        neuron = neuron_cls(tau, threshold, surrogate_grad_scale, surrogate_opt, tau_learnable=tau_learnable)
        return layer, neuron, True
    else:
        # 后备逻辑
        has_params = any(p.requires_grad for p in layer.parameters())
        if has_params:
            print(f"Warning: treating {type(layer)} as spiking computational layer.")
            neuron = neuron_cls(tau, threshold, surrogate_grad_scale, surrogate_opt, tau_learnable=tau_learnable)
            return layer, neuron, True
        else:
            print(f"Warning: removing layer {type(layer)} (no trainable parameters, treated as activation/passive).")
            return None


def build_ann_to_snn_model(ann_model, neuron_cls, tau, threshold, surrogate_grad_scale, surrogate_opt, tau_learnable):
    """
    将ANN模型转换为SNN模型

    参数:
        ann_model: 要转换的ANN模型
        neuron_cls: 神经元类
        tau: 时间常数
        threshold: 阈值
        surrogate_grad_scale: 替代梯度的缩放因子
        surrogate_opt: 替代梯度的类型

    返回:
        tuple: (layers, neurons, if_set_layer_potential, snn_model)
    """
    layers = nn.ModuleList()
    neurons = nn.ModuleList()
    if_set_layer_potential = []
    snn_model = nn.Sequential()

    # 处理每一层
    for layer in ann_model:
        result = process_ann_layer(layer, neuron_cls, tau, threshold, surrogate_grad_scale, surrogate_opt, tau_learnable)
        if result is not None:
            layer, neuron, if_set_potential = result
            layers.append(layer)
            if_set_layer_potential.append(if_set_potential)
            if if_set_potential:
                neurons.append(neuron)

    # 检查是否设置了正确数量的神经元
    if len(if_set_layer_potential) != len(layers) or len(neurons) != sum(if_set_layer_potential):
        raise ValueError("SNN Layers and if_set_layer_potential must have the same length")

    # 构建SNN模型
    for i in range(len(layers)):
        if if_set_layer_potential[i]:
            idx = sum(if_set_layer_potential[:i])
            snn_model.append(layers[i])
            snn_model.append(neurons[idx])
        else:
            snn_model.append(layers[i])

    return layers, neurons, if_set_layer_potential, snn_model


def process_snn_layer(layer, layers, neurons, if_set_layer_potential):
    """
    处理SNN模型中的一个层

    参数:
        layer: 要处理的层
        layers: 层列表
        neurons: 神经元列表
        if_set_layer_potential: 是否设置层电位的列表
    """
    # 检查层是否是BaseNeuron类
    if isinstance(layer, BaseNeuron):
        neurons.append(layer)
        if_set_layer_potential[-1] = True
        ##因为process_snn_layer是在build_snn_model中循环调用的，所以将上一层if_set_layer_potential[-1]重设为True
    else:
        layers.append(layer)
        if_set_layer_potential.append(False)
        ##默认设置为False，如果下一层是脉冲神经元类，则在后续的处理过程中会将其设置为True

def build_snn_model(snn_model):
    """
    从现有的SNN模型构建组件

    参数:
        snn_model: 现有的SNN模型

    返回:
        tuple: (layers, neurons, if_set_layer_potential)
    """
    layers = nn.ModuleList()
    neurons = nn.ModuleList()
    if_set_layer_potential = []

    # 处理SNN模型中的每一层
    for layer in snn_model:
        process_snn_layer(layer, layers, neurons, if_set_layer_potential)

    # 检查是否设置了正确数量的神经元
    if len(if_set_layer_potential) != len(layers) or len(neurons) != sum(if_set_layer_potential):
        raise ValueError("SNN Layers and if_set_layer_potential must have the same length")

    return layers, neurons, if_set_layer_potential


class SNN(nn.Module):
    def __init__(self, model, input_type='snn', neuron_type='LIF', tau=2, threshold=1.0,
                 surrogate_grad_scale=5.0, surrogate_opt="arctan_surrogate",
                 interpolation_method='linear',
                 tau_learnable=False,
                 integrator="odeint_adjoint",
                 alpha = 0.5
                 ):
        """
        统一的SNN模型，可以处理ANN模型转换或直接SNN模型

        参数:
            model: 输入模型，可以是ANN模型或SNN模型
            input_type: 'ann'或'snn'，表示输入模型的类型
            #下面参数只有在input_type为'ann'时才会使用
            neuron_type: 'LIF'或'IF'，表示使用的神经元类型
            tau: 时间常数
            threshold: 阈值
            surrogate_grad_scale: 替代梯度的缩放因子
            surrogate_opt: 替代梯度的类型
        """
        super(SNN, self).__init__()

        if input_type == 'ann':
            # 使用ANN到SNN的转换逻辑
            neuron_cls = LIFNeuron if neuron_type == 'LIF' else IFNeuron
            self.layers, self.neurons, self.if_set_layer_potential, self.snn_model = build_ann_to_snn_model(
                model, neuron_cls, tau, threshold, surrogate_grad_scale, surrogate_opt, tau_learnable
            )
        else:
            # 直接使用SNN模型
            self.snn_model = model
            self.layers, self.neurons, self.if_set_layer_potential = build_snn_model(model)

        self.interpolation_method = interpolation_method
        if integrator == "odeint_adjoint":
            from torchdiffeq import odeint_adjoint as odeint
            self.integrator = odeint
        elif integrator == "odeint":
            from torchdiffeq import odeint
            self.integrator = odeint
        elif integrator == "fdeint_adjoint":
            from torchfde import fdeint_adjoint as fdeint
            self.integrator = fdeint
        elif integrator == "fdeint":
            from torchfde import fdeint
            self.integrator = fdeint
        self.integrator_indicator = integrator
        self.alpha = alpha
        self.fde_backend = "torch"

        self.spike_count_total = torch.zeros(sum(self.if_set_layer_potential))
        self.tensor_count_total = torch.zeros(sum(self.if_set_layer_potential))

    def __str__(self):
        return str(self.snn_model)

    def __repr__(self):
        return str(self.snn_model)

    def forward(self, x, x_time, output_time=None, method='euler', options={'step_size': 0.1}):
        time_steps, batch_size = x.shape[:2]

        v_mems = []
        layer_input = x[0, :1, :]
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                layer_input = layer(layer_input)
                shape = list(layer_input.shape)
                shape[0] = batch_size
                if self.if_set_layer_potential[i]:
                    v_mems.append(torch.zeros(shape, device=x.device))
                    #change to random initialization
                    # v_mems.append(torch.randn(shape, device=x.device))

        spike_sum_init = torch.zeros(shape, device=x.device)

        ode_func = ODEFunc(self.layers, self.neurons, self.if_set_layer_potential, x, x_time, interpolation_method=self.interpolation_method)

        if not self.training:
            ode_func.eval()

        if output_time is None:
            # set output_time constains only the first and final element in x_time
            output_time = torch.tensor([x_time[0], x_time[-1]], device=x.device)

        # check if it is fdeint or odeint
        if self.integrator_indicator == "odeint_adjoint" or self.integrator_indicator == "odeint":

            # print(f"output_time: {output_time}")
            v_mem_all_time_and_cumulated_spike = self.integrator(
                ode_func,
                (*v_mems, spike_sum_init),
                output_time,
                method=method,
                options=options
            )
            # if True:#
            if not self.training:
                self.spike_count_total += ode_func.spike_count
                self.tensor_count_total += ode_func.tensor_count
                # print('self.spike_count_total, self.tensor_count_total: ', self.spike_count_total, self.tensor_count_total)
            return v_mem_all_time_and_cumulated_spike

        elif self.integrator_indicator == "fdeint_adjoint" or self.integrator_indicator == "fdeint":
            # print('using', self.integrator_indicator, 'for integration')

            T = x_time[-1]
            step_size = x_time[-1] - x_time[-2]
            v_mem_all_time_and_cumulated_spike = self.integrator(
                ode_func,
                (*v_mems, spike_sum_init),
                torch.tensor(self.alpha),
                t=T,
                step_size=step_size,
                method=method,
                options=options
            )
            if not self.training:
                self.spike_count_total += ode_func.spike_count
                self.tensor_count_total += ode_func.tensor_count
                # print('self.spike_count_total, self.tensor_count_total: ', self.spike_count_total, self.tensor_count_total)
            # out = fdeint(self.odefunc, x, torch.tensor(args.alpha), t=args.T,
            #              step_size=args.step_size, method=args.method, options={"corrector_step": 5, })

            return (v_mem_all_time_and_cumulated_spike, )
        # print element size for all tensor in v_mem_all_time
        # for i in range(len(v_mem_all_time_and_cumulated_spike)):
        #     print(f"v_mem_all_time_and_cumulated_spike[{i}]: {v_mem_all_time_and_cumulated_spike[i].size()}")

        # total_spikes = v_mem_all_time[-1][-1]
        # return total_spikes

        # total_mem = v_mem_all_time[-2][-1]
        # return total_mem



class ShapeRecorder(Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        self.shapes = []
        self.neuron_instances = []
    def run_node(self, node):
        result = super().run_node(node)
        # whenever we hit a neuron, record its output tensor shape
        if node.op == 'call_module':
            submod = self.module.get_submodule(node.target)
            if isinstance(submod, BaseNeuron):
                # result is (dv_dt, spike_out)
                spike_out = result
                # record the shape of either (they match)
                self.shapes.append(tuple(spike_out.shape))
                self.neuron_instances.append(submod)
        return result

from functools import wraps

def requires_initialization(func):
    """装饰器：确保方法调用前已初始化"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_is_initialized') or not self._is_initialized:
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"ERROR: {self.__class__.__name__} not initialized!\n"
                f"You must call '_set_neuron_shapes(input_shape)' before calling '{func.__name__}'.\n"
                f"Example:\n"
                f"  net._set_neuron_shapes(input_shape=(1, 2, 128, 128)), # (batch=1, channels=2, H=128, W=128)\n"
                f"{'='*60}"
            )
        return func(self, *args, **kwargs)
    return wrapper


def setup_multi_term(
    module,
    alpha,
    multi_coefficient=None,
    learn_coefficient=False,
    learn_alpha=False,
    alpha_min=0.01,
    alpha_max=0.99,
):
    """
    Setup fractional order parameters for a module.
    Supports both single-term and multi-term FDE.

    Args:
        module: nn.Module to register parameters/buffers on
        alpha: float or list of floats
               - If float: single-term FDE with order alpha
               - If list: multi-term FDE with orders [alpha_1, alpha_2, ...]
        multi_coefficient: list of coefficients for multi-term FDE, e.g., [1.0, 0.5, 0.2]
                          Required if alpha is a list, ignored if alpha is a float
        learn_coefficient: if True, coefficients are learnable parameters (only for multi-term)

    Returns:
        tuple: (is_multi_term, learn_coefficient)
    """
    # Determine if single-term or multi-term
    if isinstance(alpha, (list, tuple)):
        # Multi-term FDE
        is_multi_term = True
        multi_alpha = alpha

        # Validate multi_coefficient
        if multi_coefficient is None:
            raise ValueError(
                "multi_coefficient must be provided when alpha is a list (multi-term FDE)"
            )
        if len(multi_alpha) != len(multi_coefficient):
            raise ValueError(
                f"alpha (len={len(multi_alpha)}) and multi_coefficient "
                f"(len={len(multi_coefficient)}) must have the same length"
            )

        if learn_alpha:
            raise ValueError("learn_alpha is only supported for single-term FDE alpha")

        # Register multi_alpha as buffer (not learnable, but moves with device)
        module.register_buffer('alpha', torch.tensor(multi_alpha, dtype=torch.float32))

        # Register multi_coefficient as parameter (learnable) or buffer (fixed)
        if learn_coefficient:
            module.multi_coefficient = nn.Parameter(
                torch.tensor(multi_coefficient, dtype=torch.float32)
            )
        else:
            module.register_buffer(
                'multi_coefficient',
                torch.tensor(multi_coefficient, dtype=torch.float32)
            )

        print(f"[Multi-term FDE] Enabled:")
        print(f"  - Fractional orders (alpha): {multi_alpha}")
        print(f"  - Coefficients: {multi_coefficient}")
        print(f"  - Learnable coefficients: {learn_coefficient}")

    else:
        # Single-term FDE
        is_multi_term = False
        learn_coefficient = False
        module.learn_alpha = bool(learn_alpha)
        module.alpha_init = float(alpha)
        module.alpha_min_value = float(alpha_min)
        module.alpha_max_value = float(alpha_max)
        if module.alpha_min_value >= module.alpha_max_value:
            raise ValueError(f"alpha_min must be < alpha_max, got {alpha_min} >= {alpha_max}")

        if learn_alpha:
            alpha_clamped = max(module.alpha_min_value + 1e-6, min(module.alpha_max_value - 1e-6, float(alpha)))
            normalized = (alpha_clamped - module.alpha_min_value) / (module.alpha_max_value - module.alpha_min_value)
            normalized = max(1e-6, min(1.0 - 1e-6, normalized))
            raw_alpha = torch.logit(torch.tensor(normalized, dtype=torch.float32))
            module.alpha_raw = nn.Parameter(raw_alpha)
            module.register_buffer('alpha_min', torch.tensor(module.alpha_min_value, dtype=torch.float32))
            module.register_buffer('alpha_max', torch.tensor(module.alpha_max_value, dtype=torch.float32))
        else:
            # Register alpha as buffer
            module.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        module.multi_coefficient = None

        print(f"[Single-term FDE] Enabled:")
        print(f"  - Fractional order (alpha): {alpha}")
        print(f"  - Learnable alpha: {learn_alpha}")
        if learn_alpha:
            print(f"  - Alpha range: [{module.alpha_min_value}, {module.alpha_max_value}]")

    return is_multi_term, learn_coefficient


def get_integrator(integrator_name):
    """
    Get the integrator function based on name.

    Args:
        integrator_name: str, one of 'odeint_adjoint', 'odeint', 'fdeint_adjoint', 'fdeint'

    Returns:
        integrator function
    """
    if integrator_name == "odeint_adjoint":
        from torchdiffeq import odeint_adjoint as odeint
        return odeint
    elif integrator_name == "odeint":
        from torchdiffeq import odeint
        return odeint
    elif integrator_name == "fdeint_adjoint":
        from torchfde import fdeint_adjoint as fdeint
        return fdeint
    elif integrator_name == "fdeint":
        from torchfde import fdeint
        return fdeint
    else:
        raise ValueError(f"Unknown integrator: {integrator_name}")



class SNNWrapper(nn.Module):
    def __init__(self, base,
                 integrator="odeint",
                 interpolation_method='linear',
                 alpha=0.5,
                 multi_coefficient=None,
                 learn_coefficient=False,
                 learn_alpha=False,
                 alpha_min=0.01,
                 alpha_max=0.99,
                 alpha_mode="global",
                 state_alpha=None,
                 learn_state_alpha=False,
                 time_steps=16,
                 alpha_constraint="none",
                 compile_ode=True):
        """
        Args:
            base: base neural network model
            integrator: str, integrator type
            interpolation_method: str, interpolation method for inputs
            alpha: float or list of floats
                   - If float: single-term FDE with order alpha
                   - If list: multi-term FDE with orders [alpha_1, alpha_2, ...]
            multi_coefficient: list of coefficients for multi-term FDE
                              Required if alpha is a list
            learn_coefficient: if True, coefficients are learnable (only for multi-term)
        """

        super().__init__()
        self.integrator_indicator = integrator
        self.interpolation_method = interpolation_method
        self.alpha_mode = alpha_mode
        if self.alpha_mode not in {"global", "per_state", "per_time"}:
            raise ValueError(
                f"Unsupported alpha_mode={alpha_mode}; expected 'global', 'per_state', or 'per_time'."
            )
        if self.alpha_mode in {"per_state", "per_time"} and isinstance(alpha, (list, tuple)):
            raise ValueError(f"alpha_mode='{self.alpha_mode}' expects a single initial alpha.")
        self.learn_state_alpha = bool(learn_state_alpha)
        self.state_alpha_init = None


        # Setup multi-term parameters (external function)
        self.is_multi_term, self.learn_coefficient = setup_multi_term(
            self,
            alpha,
            multi_coefficient,
            learn_coefficient,
            learn_alpha=learn_alpha if self.alpha_mode == "global" else False,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        if self.alpha_mode == "per_state":
            if self.is_multi_term:
                raise ValueError("alpha_mode='per_state' is not supported with multi-term alpha.")
            self._setup_state_alpha(state_alpha, alpha, learn_state_alpha, alpha_min, alpha_max)
        elif self.alpha_mode == "per_time":
            self._setup_time_alpha(alpha, time_steps, learn_alpha, alpha_constraint)

        # Get integrator function only when needed
        if integrator in ("odeint_adjoint", "odeint"):
            self.integrator = get_integrator(integrator)
        else:
            self.integrator = None

        # build our FX‐based ODEFunc
        self.ode_func = ODEFuncFromFX(base, interpolation_method=interpolation_method)
        self.traced_backbone = self.ode_func.traced
        self.post_neuron_module = self.ode_func.get_post_neuron_module()
        self.neuron_instances = None
        # Initialize as None to track initialization status
        self.neuron_shapes = None

        self._is_initialized = False  # 添加初始化标志
        # we must call _set_neuron_shapes before run
        if compile_ode:
            self.ode_func = torch.compile(self.ode_func)
        self.fde_backend = "torch"

    def _parse_state_alpha(self, state_alpha, fallback_alpha):
        if state_alpha is None or state_alpha == "":
            return [float(fallback_alpha)]
        if isinstance(state_alpha, str):
            values = [item.strip() for item in state_alpha.split(",") if item.strip()]
            return [float(item) for item in values]
        if isinstance(state_alpha, torch.Tensor):
            return [float(item) for item in state_alpha.detach().cpu().flatten()]
        if isinstance(state_alpha, (list, tuple)):
            return [float(item) for item in state_alpha]
        return [float(state_alpha)]

    def _setup_time_alpha(self, alpha, time_steps, learn_alpha, alpha_constraint):
        n_updates = int(time_steps)
        if n_updates < 1:
            raise ValueError(f"time_steps must be positive, got {time_steps}")
        init = torch.full((n_updates,), float(alpha), dtype=torch.float32)
        self.learn_time_alpha = bool(learn_alpha)
        self.time_steps = int(time_steps)
        self.alpha_constraint = str(alpha_constraint)
        if self.alpha_constraint not in {"none", "softmax01"}:
            raise ValueError(f"Unsupported alpha_constraint={alpha_constraint}")
        if self.alpha_constraint == "softmax01":
            if not 0.0 < float(alpha) < 1.0:
                raise ValueError("softmax01 alpha initialization must be in (0, 1)")
            raw_init = torch.logit(init)
            if self.learn_time_alpha:
                self.alpha_time_raw = nn.Parameter(raw_init)
            else:
                self.register_buffer("alpha_time_raw", raw_init)
        elif self.learn_time_alpha:
            self.alpha_time = nn.Parameter(init)
        else:
            self.register_buffer("alpha_time", init)
        print("[Per-time FDE alpha] Enabled:")
        print(f"  - Integration updates: {n_updates}")
        print(f"  - Alpha init: {float(alpha)}")
        print(f"  - Learnable: {self.learn_time_alpha}")
        print(f"  - Constraint: {self.alpha_constraint}")

    def _setup_state_alpha(self, state_alpha, fallback_alpha, learn_state_alpha, alpha_min, alpha_max):
        values = self._parse_state_alpha(state_alpha, fallback_alpha)
        self.state_alpha_init = values
        self.state_alpha_min_value = float(alpha_min)
        self.state_alpha_max_value = float(alpha_max)
        if self.state_alpha_min_value >= self.state_alpha_max_value:
            raise ValueError(
                f"alpha_min must be < alpha_max for state alpha, got {alpha_min} >= {alpha_max}"
            )
        init = torch.tensor(values, dtype=torch.float32)
        if learn_state_alpha:
            clamped = init.clamp(self.state_alpha_min_value + 1e-6, self.state_alpha_max_value - 1e-6)
            normalized = (clamped - self.state_alpha_min_value) / (self.state_alpha_max_value - self.state_alpha_min_value)
            normalized = normalized.clamp(1e-6, 1.0 - 1e-6)
            self.state_alpha_raw = nn.Parameter(torch.logit(normalized))
            self.register_buffer('state_alpha_min', torch.tensor(self.state_alpha_min_value, dtype=torch.float32))
            self.register_buffer('state_alpha_max', torch.tensor(self.state_alpha_max_value, dtype=torch.float32))
        else:
            self.register_buffer('state_alpha', init)
        print("[Per-state FDE alpha] Enabled:")
        print(f"  - State alpha init: {values}")
        print(f"  - Learnable state alpha: {learn_state_alpha}")
        if learn_state_alpha:
            print(f"  - State alpha range: [{self.state_alpha_min_value}, {self.state_alpha_max_value}]")

    def get_alpha(self):
        if self.alpha_mode == "per_time":
            if self.alpha_constraint == "softmax01":
                logits = torch.stack((self.alpha_time_raw, torch.zeros_like(self.alpha_time_raw)), dim=-1)
                return torch.softmax(logits, dim=-1)[..., 0]
            return self.alpha_time
        if getattr(self, "learn_alpha", False):
            return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_raw)
        return self.alpha

    def get_state_alpha(self):
        if self.alpha_mode != "per_state":
            return None
        if getattr(self, "learn_state_alpha", False):
            return self.state_alpha_min + (self.state_alpha_max - self.state_alpha_min) * torch.sigmoid(self.state_alpha_raw)
        return self.state_alpha

    def get_alpha_value(self):
        if self.alpha_mode == "per_state":
            return self.get_state_alpha_value()
        alpha = self.get_alpha()
        if torch.is_tensor(alpha):
            if alpha.numel() == 1:
                return float(alpha.detach().cpu())
            return [float(item) for item in alpha.detach().cpu().flatten()]
        return float(alpha)

    def get_state_alpha_value(self):
        alpha = self.get_state_alpha()
        if alpha is None:
            return None
        return [float(item) for item in alpha.detach().cpu().flatten()]

    def _build_state_alpha_tuple(self, v_mems):
        if self.alpha_mode != "per_state":
            return self.get_alpha()
        base_alpha = self.get_state_alpha()
        if base_alpha is None:
            raise RuntimeError("state alpha is not initialized.")
        alpha_tuple = []
        for neuron, state in zip(self.neuron_instances, v_mems):
            multiplier = int(getattr(neuron, "state_multiplier", 1))
            if base_alpha.numel() == 1:
                alpha_values = base_alpha.repeat(multiplier)
            elif base_alpha.numel() == multiplier:
                alpha_values = base_alpha
            else:
                names = getattr(neuron, "state_names", tuple(str(i) for i in range(multiplier)))
                raise ValueError(
                    f"state_alpha length {base_alpha.numel()} does not match "
                    f"{neuron.__class__.__name__} state_multiplier={multiplier} ({names})."
                )
            if state.shape[1] % multiplier != 0:
                raise ValueError(
                    f"State channel count {state.shape[1]} is not divisible by "
                    f"{neuron.__class__.__name__}.state_multiplier={multiplier}."
                )
            channels_per_state = state.shape[1] // multiplier
            pieces = []
            for alpha_value in alpha_values:
                alpha_piece = alpha_value.to(device=state.device, dtype=state.dtype).reshape(1)
                view_shape = [1] * state.dim()
                view_shape[1] = 1
                alpha_piece = alpha_piece.reshape(view_shape)
                repeat_shape = [1] * state.dim()
                repeat_shape[1] = channels_per_state
                pieces.append(alpha_piece.repeat(*repeat_shape))
            alpha_mask = torch.cat(pieces, dim=1)
            alpha_tuple.append(alpha_mask)
        return tuple(alpha_tuple)

    def train(self, mode=True):
        """覆盖train方法，确保所有子模块正确切换模式"""
        super().train(mode)  # 先调用父类的train方法
        self.traced_backbone.train(mode)
        self.ode_func.ode_gm.train(mode)
        if hasattr(self.ode_func, 'post_neuron_module'):
            self.post_neuron_module.train(mode)
        return self

    def eval(self):
        """覆盖eval方法，确保所有子模块正确切换到评估模式"""
        super().eval()  # 先调用父类的eval方法
        self.traced_backbone.eval()
        self.ode_func.ode_gm.eval()
        if hasattr(self.ode_func, 'post_neuron_module'):
            self.post_neuron_module.eval()
        return self

    def _set_neuron_shapes(self, input_shape = None):#[1, 2, 128, 128]):
        """Pre-compute neuron output shapes to avoid runtime overhead"""
        # Create dummy input with correct dimensions
        #dummy_input = torch.zeros(1, 2, 128, 128)  # (batch=1, channels=2, H=128, W=128)
        self._is_initialized = True
        if input_shape is None:
            #raise an exception
            raise NotImplementedError("Neuron membrane shape must be obtained using an input shape.")
        dummy_input = torch.zeros(input_shape, device=next(self.parameters()).device)
        # Set to eval mode temporarily to avoid affecting BatchNorm
        training_mode = self.training
        self.eval()

        with torch.no_grad():
            recorder = ShapeRecorder(self.traced_backbone)
            recorder.run(dummy_input)
            self.neuron_shapes = recorder.shapes
            self.neuron_instances = recorder.neuron_instances
        # print("?????????????????",self.neuron_shapes)
        # Restore training mode
        self.train(training_mode)

    def _reset_mem(self):
        for neuron in self.neuron_instances:
            neuron._reset()

    @requires_initialization
    def forward(self, x, x_time, output_time=None, method='euler', options={'step_size': 0.1}):

        time_steps, batch_size = x.shape[:2]
        if len(x_time) != time_steps:
            x_time = x_time[0:time_steps]
        self.ode_func.set_inputs(x, x_time)

        adjusted_shapes = [(batch_size, *shape[1:]) for shape in self.neuron_shapes]
        v_mems = []
        for neuron, shape in zip(self.neuron_instances, adjusted_shapes):
            if hasattr(neuron, "initial_state"):
                v_mems.append(neuron.initial_state(shape, device=x.device, dtype=x.dtype))
            else:
                v_mems.append(torch.zeros(shape, device=x.device, dtype=x.dtype))

        spike_sum_init = torch.zeros(adjusted_shapes[-1], device=x.device, dtype=x.dtype)

        # print('--init v_mems and spike_sum_init done--')

        # 2) now run the real integration
        if output_time is None:
            # set output_time constains only the first and final element in x_time
            output_time = x_time#torch.tensor([x_time[0], x_time[-1]], device=x.device)

        # output_time = x_time
        # print(output_time)

        # if self.integrator_indicator == "fdeint_adjoint" and method in FDEINT_METHODS:
        #     raise ValueError(
        #         f"Method '{method}' requires integrator 'fdeint', got '{self.integrator_indicator}'."
        #     )
        # if self.integrator_indicator == "fdeint" and method in FDEINT_ADJOINT_METHODS:
        #     raise ValueError(
        #         f"Method '{method}' requires integrator 'fdeint_adjoint', got '{self.integrator_indicator}'."
        #     )

        # check if it is fdeint or odeint
        if self.integrator_indicator == "odeint" and method == 'euler':
            v_mem_all_time_and_final_spike = euler_integrate_tuple_compiled(self.ode_func,
                                                                          (*v_mems, spike_sum_init),
                                                                          output_time)
            # finalspike_out = v_mem_all_time_and_final_spike[-1]#[1:, ...]
            finalspike_out = torch.stack(v_mem_all_time_and_final_spike, dim=0)
            # print("finalspike_out.shape", finalspike_out.shape)
            final_ouput = self.post_neuron_module(finalspike_out)
            return final_ouput

            # final_v_mem = v_mem_all_time_and_cumulated_spike[-1][1:,...]
            # print(final_v_mem.max(), final_v_mem.min(), final_v_mem.sum())
            # v_scaled = final_v_mem - self.ode_func.finalneuron_threshold
            # finalspike_out = self.ode_func.finalneuron_surrogate_f(v_scaled, self.ode_func.finalneuron_surrogate_grad_scale)
            # final_ouput = self.post_neuron_module(final_v_mem)
            # return final_ouput

        if self.integrator_indicator == "odeint_mem":
            v_mem_all_time_and_final_spike = step_dynamics(self.ode_func,
                (*v_mems, spike_sum_init),
                output_time)
            finalspike_out = torch.stack(v_mem_all_time_and_final_spike, dim=0)
            final_ouput = self.post_neuron_module(finalspike_out)
            return final_ouput


        elif self.integrator_indicator == "odeint_adjoint" or self.integrator_indicator == "odeint":
            # raise NotImplementedError("please only use odeint+euler or fdeint+gl")
            # from torchdiffeq import odeint
            # v_mem_all_time_and_cumulated_spike = odeint(
            v_mem_all_time_and_cumulated_spike = self.integrator(
                self.ode_func,
                (*v_mems, spike_sum_init),
                output_time,
                method=method,
                options=options
            )
            finalspike_out_sum = v_mem_all_time_and_cumulated_spike[-1][-1:,...]

            # v_scaled = final_v_mem - self.ode_func.finalneuron_threshold
            # finalspike_out = self.ode_func.finalneuron_surrogate_f(v_scaled, self.ode_func.finalneuron_surrogate_grad_scale)
            final_ouput = self.post_neuron_module(finalspike_out_sum)
            return final_ouput


        elif self.integrator_indicator == "fdeint":
            if options["memory"] == -1:
                memory = None# Here None means full memory
            else:
                memory = options["memory"]
            alpha = self._build_state_alpha_tuple(v_mems) if self.alpha_mode == "per_state" else self.get_alpha()

            if self.alpha_mode == "per_time" and method != "vopred":
                raise ValueError("alpha_mode='per_time' requires method='vopred'.")
            if method == "vopred" and self.alpha_mode != "per_time":
                raise ValueError("method='vopred' requires alpha_mode='per_time'.")
            if self.alpha_mode == "per_time" and getattr(self, "fde_backend", "torch") != "torch":
                raise ValueError("alpha_mode='per_time' is only implemented for the torch FDE backend.")

            use_cupy = getattr(self, "fde_backend", "torch") == "cupy" and method in CUPY_FDEINT_METHODS
            if self.alpha_mode == "per_state" and use_cupy:
                raise ValueError("alpha_mode='per_state' is only implemented for the torch FDE backend.")
            if not hasattr(self, '_cupy_log_printed'):
                print(f"[cupy] fde_backend={getattr(self, 'fde_backend', 'torch')}, method={method}, use_cupy={use_cupy}")
                self._cupy_log_printed = True
            if self.is_multi_term:
                # Multi-term solver
                if use_cupy:
                    detach_conv = options.get("glmulti_detach_conv", True)
                    if self.learn_coefficient and detach_conv:
                        detach_conv = False
                    v_mem_all_time_and_final_spike = forward_glmulti_cupy(
                        self.ode_func, (*v_mems, spike_sum_init),
                        alpha, output_time,
                        memory=memory, multi_coefficient=self.multi_coefficient,
                        history_chunk_size=options.get("glmulti_chunk"),
                        detach_conv=detach_conv
                    )
                else:
                    from .solver import glmethod_multiterm_integrate_tuple
                    # Compile the shared ODE module, not 16 copies of the full solver graph.
                    glmethod_multiterm_compiled = glmethod_multiterm_integrate_tuple

                    v_mem_all_time_and_final_spike = glmethod_multiterm_compiled(
                        self.ode_func, (*v_mems, spike_sum_init),
                        alpha, output_time,  # self.alpha is now a tensor of multiple orders
                        memory=memory, multi_coefficient=self.multi_coefficient
                    )
            else:
                # Single-term solver
                if use_cupy:
                    if method == "gl":
                        v_mem_all_time_and_final_spike = gl_integrate_tuple_cupy(
                            self.ode_func, (*v_mems, spike_sum_init),
                            alpha, output_time, memory=memory
                        )
                    elif method == "pred":
                        v_mem_all_time_and_final_spike = pred_integrate_tuple_cupy(
                            self.ode_func, (*v_mems, spike_sum_init),
                            alpha, output_time, memory=memory
                        )
                    elif method == "trap":
                        v_mem_all_time_and_final_spike = trap_integrate_tuple_cupy(
                            self.ode_func, (*v_mems, spike_sum_init),
                            alpha, output_time, memory=memory
                        )
                    elif method == "l1":
                        v_mem_all_time_and_final_spike = l1_integrate_tuple_cupy(
                            self.ode_func, (*v_mems, spike_sum_init),
                            alpha, output_time, memory=memory
                        )
                    else:
                        raise ValueError(f"CuPy backend does not support method '{method}'")
                else:
                    if self.alpha_mode == "per_state":
                        if method not in PER_STATE_SOLVERS:
                            raise ValueError(
                                f"alpha_mode='per_state' only supports torch methods "
                                f"{sorted(PER_STATE_SOLVERS)}, got {method}."
                            )
                        integrate_tuple_compiled = PER_STATE_SOLVERS[method]
                    else:
                        integrate_tuple_compiled = SOLVERS[method]
                    v_mem_all_time_and_final_spike = integrate_tuple_compiled(
                        self.ode_func, (*v_mems, spike_sum_init),
                        alpha, output_time, memory=memory  # alpha is a scalar tensor
                    )

            # finalspike_out = v_mem_all_time_and_final_spike[-1]#[1:, ...]
            finalspike_out = torch.stack(v_mem_all_time_and_final_spike[-1], dim=0)
            # print(finalspike_out.shape)
            final_ouput = self.post_neuron_module(finalspike_out)

            # print('self.multi_coefficient: ', self.multi_coefficient)
            return final_ouput

        elif self.integrator_indicator == "fdeint_mem":

            v_mem_all_time_and_final_spike = step_dynamics(self.ode_func,
                                                                      (*v_mems, spike_sum_init),
                                                                      output_time)
            # finalspike_out = v_mem_all_time_and_final_spike
            finalspike_out = torch.stack(v_mem_all_time_and_final_spike, dim=0)
            final_ouput = self.post_neuron_module(finalspike_out)
            return final_ouput

        elif self.integrator_indicator == "fdeint_adjoint":
            if self.alpha_mode == "per_state":
                raise ValueError("alpha_mode='per_state' is not implemented for fdeint_adjoint.")
            if options["memory"] == -1:
                memory = None# Here None means full memory
            else:
                memory = options["memory"]
            alpha = self.get_alpha()
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, dtype=output_time.dtype, device=output_time.device)
            method_adj = method
            if method in {"gl", "pred", "trap", "l1"}:
                method_adj = f"{method}-f"
            backend = "cupy" if (getattr(self, "fde_backend", "torch") == "cupy"
                                 and method_adj in CUPY_ADJOINT_METHODS) else "torch"
            v_mem_all_time_and_cumulated_spike = fdeint_adjoint(
                self.ode_func,
                (*v_mems, spike_sum_init),
                alpha,
                output_time,
                method=method_adj,
                memory=memory,
                backend=backend,
                multi_coefficient=self.multi_coefficient if self.is_multi_term else None
            )
            # print('v_mem_all_time_and_cumulated_spike.shape: ', v_mem_all_time_and_cumulated_spike[-1].shape)
            finalspike_out_sum = v_mem_all_time_and_cumulated_spike[-1].unsqueeze(0)
            final_ouput = self.post_neuron_module(finalspike_out_sum)
            return final_ouput

        elif False:#self.integrator_indicator == "fdeint_adjoint" or self.integrator_indicator == "fdeint":
            # print(f"output_time: {output_time}")# print('using', self.integrator_indicator, 'for integration')
            # raise NotImplementedError("please only use odeint+euler or fdeint+gl")
            T = output_time[-1]
            step_size = output_time[-1] - output_time[-2]
            v_mem_all_time_and_cumulated_spike = self.integrator(
                self.ode_func,
                (*v_mems, spike_sum_init),
                torch.tensor(self.alpha),
                t=T,
                step_size=step_size,
                method=method,
                options=options
            )
            # print(v_mem_all_time_and_cumulated_spike[-1].shape)
            finalspike_out_sum = v_mem_all_time_and_cumulated_spike[-1].unsqueeze(0)
            # v_scaled = final_v_mem - self.ode_func.finalneuron_threshold
            # finalspike_out = self.ode_func.finalneuron_surrogate_f(v_scaled, self.ode_func.finalneuron_surrogate_grad_scale)
            final_ouput = self.post_neuron_module(finalspike_out_sum)
            return final_ouput
