import numpy as np
import torch
from torch import fx, nn
from .neuron import BaseNeuron
from .layer import VotingLayer, ClassificationHead
import operator
from typing import Dict, Tuple, Any
from torch.fx import Tracer

"""
Torch-based ODEFunc with Interpolation Methods

This module provides an implementation of ODEFunc for neural ODE systems with
various interpolation methods implemented purely in PyTorch. The interpolation
is applied to input data at arbitrary time points during integration.

Supported interpolation methods:
- 'linear': Linear interpolation between adjacent points
- 'nearest': Nearest neighbor interpolation (no actual interpolation)
- 'cubic': Cubic spline interpolation using Catmull-Rom formula
- 'akima': Akima interpolation that reduces oscillations in cubic splines

All methods are implemented using PyTorch operations to maintain GPU acceleration
without requiring data transfers between devices.
"""


def linear_interpolate(x, x_time, t):
    """PyTorch linear interpolation - tensor version."""
    # 确保 t 是 tensor
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=x_time.dtype, device=x_time.device)

    # 保持 tensor 形式，不使用 [0] 索引
    idx = torch.searchsorted(x_time, t) - 1
    idx = torch.clamp(idx, 0, len(x_time) - 2)

    # 使用 gather 或索引来获取值，保持 tensor 操作
    t0 = x_time[idx]
    t1 = x_time[idx + 1]
    x0 = x[idx]
    x1 = x[idx + 1]

    # Linear interpolation
    ratio = (t - t0) / (t1 - t0)
    # 确保 ratio 的维度匹配 x 的维度
    if ratio.dim() < x0.dim():
        ratio = ratio.unsqueeze(-1)

    return x0 + ratio * (x1 - x0)


def nearest_interpolate(x, x_time, t):
    """PyTorch nearest neighbor interpolation - tensor version."""
    # 确保 t 是 tensor
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=x_time.dtype, device=x_time.device)

    # 计算距离
    distances = torch.abs(x_time - t)
    idx = torch.argmin(distances)

    return x[idx].clone()


def cubic_interpolate(x, x_time, t):
    """
    PyTorch cubic interpolation using four surrounding points - tensor version.
    Based on Catmull-Rom spline formula.
    """
    # 确保 t 是 tensor
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=x_time.dtype, device=x_time.device)

    # Find the indices, keep as tensor
    idx = torch.searchsorted(x_time, t) - 1
    idx = torch.clamp(idx, 0, len(x_time) - 2)

    # 使用条件逻辑但保持 tensor 形式
    # 创建索引 tensor
    n = len(x_time)

    # 根据 idx 的位置选择不同的索引
    # 使用 torch.where 来处理边界情况
    idx0 = torch.where(idx == 0, idx, torch.where(idx == n - 2, idx - 2, idx - 1))
    idx1 = torch.where(idx == 0, idx, torch.where(idx == n - 2, idx - 1, idx))
    idx2 = torch.where(idx == 0, idx + 1, torch.where(idx == n - 2, idx, idx + 1))
    idx3 = torch.where(idx == 0, idx + 2, torch.where(idx == n - 2, idx, idx + 2))

    # 限制索引范围
    idx0 = torch.clamp(idx0, 0, n - 1)
    idx1 = torch.clamp(idx1, 0, n - 1)
    idx2 = torch.clamp(idx2, 0, n - 1)
    idx3 = torch.clamp(idx3, 0, n - 1)

    # 获取四个点
    p0, p1, p2, p3 = x[idx0], x[idx1], x[idx2], x[idx3]
    t0, t1, t2, t3 = x_time[idx0], x_time[idx1], x_time[idx2], x_time[idx3]

    # 避免除零
    dt = t2 - t1
    dt = torch.where(dt == 0, torch.ones_like(dt), dt)

    # Normalize t to range [0, 1] within the current segment
    t_norm = (t - t1) / dt
    t_norm = torch.clamp(t_norm, 0, 1)

    # 确保维度匹配
    if t_norm.dim() < p0.dim():
        t_norm = t_norm.unsqueeze(-1)

    # Cubic interpolation coefficients
    t_norm2 = t_norm * t_norm
    t_norm3 = t_norm2 * t_norm

    # Catmull-Rom spline formula
    a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
    a2 = -0.5 * p0 + 0.5 * p2
    a3 = p1

    return a0 * t_norm3 + a1 * t_norm2 + a2 * t_norm + a3


def akima_interpolate(x, x_time, t):
    """
    PyTorch Akima interpolation - fully tensor version.
    """
    # 确保 t 是 tensor
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=x_time.dtype, device=x_time.device)

    # Find the nearest time point index, keep as tensor
    idx = torch.searchsorted(x_time, t) - 1
    idx = torch.clamp(idx, 0, len(x_time) - 2)

    n = len(x_time)

    # 使用 torch.where 处理边界情况
    # 确保有足够的点进行 Akima 插值
    idx = torch.where(idx < 2, torch.tensor(2, device=idx.device), idx)
    idx = torch.where(idx > n - 4, torch.tensor(n - 4, device=idx.device), idx)

    # 创建索引数组
    indices = torch.arange(5, device=x_time.device) + (idx - 2)
    indices = torch.clamp(indices, 0, n - 1)

    # 批量获取值
    x_vals = x[indices]
    t_vals = x_time[indices]

    # 计算斜率，避免循环
    dt = t_vals[1:] - t_vals[:-1]
    # 避免除零
    dt = torch.where(dt == 0, torch.ones_like(dt) * 1e-10, dt)
    dx = x_vals[1:] - x_vals[:-1]

    # 处理维度
    if dt.dim() < dx.dim():
        dt = dt.unsqueeze(-1)

    m = dx / dt

    # 计算权重
    dm = torch.abs(m[1:] - m[:-1])
    w1 = dm[1]  # |m2 - m1|
    w2 = dm[0]  # |m1 - m0|
    w3 = dm[2]  # |m3 - m2|

    # 添加 epsilon 避免除零
    epsilon = 1e-10

    # 计算 s1 和 s2
    denom1 = w1 + w2 + epsilon
    s1 = torch.where(
        denom1 > epsilon,
        (w1 * m[0] + w2 * m[2]) / denom1,
        m[1]
    )

    denom2 = w1 + w3 + epsilon
    s2 = torch.where(
        denom2 > epsilon,
        (w1 * m[3] + w3 * m[1]) / denom2,
        m[2]
    )

    # 标准化 t 到 [0,1] 范围
    h = t_vals[3] - t_vals[2]
    h = torch.where(h == 0, torch.ones_like(h), h)
    t_norm = (t - t_vals[2]) / h
    t_norm = torch.clamp(t_norm, 0, 1)

    # 确保维度匹配
    if t_norm.dim() < x_vals[2].dim():
        t_norm = t_norm.unsqueeze(-1)

    # 计算 Akima 系数
    a = x_vals[2]
    b = s1 * h
    c = 3 * (x_vals[3] - x_vals[2]) / h - 2 * s1 - s2
    d = s1 + s2 - 2 * (x_vals[3] - x_vals[2]) / h

    # 计算多项式
    t_norm2 = t_norm * t_norm
    t_norm3 = t_norm2 * t_norm

    return a + b * t_norm + c * t_norm2 + d * t_norm3


# def interpolate(x, x_time, t, method='linear'):
#     """
#     Interpolate input at time t using purely PyTorch operations.
#     Args:
#         x: Input tensor data
#         x_time: Time points corresponding to input data
#         t: The time point to interpolate at
#         method: Interpolation method ('linear', 'nearest', 'cubic', or 'akima')
#     Returns:
#         Interpolated input tensor at time t
#     """
#     t = torch.as_tensor(t, dtype=x_time.dtype, device=x_time.device)
#
#     # 记录原始维度
#     original_dim = t.dim()
#
#     # 如果 t 是标量 tensor，确保它的形状正确
#     if t.dim() == 0:
#         t = t.unsqueeze(0)
#
#     #print the device of t and x_time
#
#
#     # 处理边界情况 - 使用 tensor 比较
#     result = torch.where(
#         t <= x_time[0],
#         x[0].clone().unsqueeze(0) if t.dim() > 0 else x[0].clone(),
#         torch.where(
#             t >= x_time[-1],
#             x[-1].clone().unsqueeze(0) if t.dim() > 0 else x[-1].clone(),
#             torch.zeros_like(x[0]).unsqueeze(0) if t.dim() > 0 else torch.zeros_like(x[0])
#         )
#     )
#
#     # 判断是否在边界内
#     in_bounds = (t > x_time[0]) & (t < x_time[-1])
#
#     # 只有在边界内才进行插值
#     if in_bounds.any():
#         if method == 'linear':
#             interpolated = linear_interpolate(x, x_time, t)
#         elif method == 'nearest':
#             interpolated = nearest_interpolate(x, x_time, t)
#         elif method == 'cubic':
#             interpolated = cubic_interpolate(x, x_time, t)
#         elif method == 'akima':
#             interpolated = akima_interpolate(x, x_time, t)
#         else:
#             # 默认回退行为 - 使用 tensor 操作
#             t_normalized = t / 0.1
#             t_int = torch.clamp(t_normalized.long(), 0, x.size(0) - 1)
#             interpolated = x[t_int]
#
#         # 确保维度匹配
#         if interpolated.dim() < result.dim():
#             interpolated = interpolated.unsqueeze(0)
#         elif interpolated.dim() > result.dim():
#             result = result.unsqueeze(0)
#
#         # 扩展 in_bounds 以匹配数据维度
#         while in_bounds.dim() < result.dim():
#             in_bounds = in_bounds.unsqueeze(-1)
#
#         # 只在边界内的部分使用插值结果
#         result = torch.where(in_bounds, interpolated, result)
#
#     # 如果原始 t 是标量，返回相应形状的结果
#     if original_dim == 0 and result.shape[0] == 1:
#         result = result.squeeze(0)
#
#     return result


def interpolate(x, x_time, t, method='linear'):
    """
    Interpolate input at time t using purely PyTorch operations.
    Args:
        x: Input tensor data
        x_time: Time points corresponding to input data
        t: The time point to interpolate at (scalar)
        method: Interpolation method ('linear', 'nearest', 'cubic', or 'akima')
    Returns:
        Interpolated input tensor at time t
    """
    t = torch.as_tensor(t, dtype=x_time.dtype, device=x_time.device)

    # 处理边界情况 - 直接用条件判断
    if t <= x_time[0]:
        return x[0].clone()
    elif t >= x_time[-1]:
        return x[-1].clone()

    # 在边界内，执行插值
    if method == 'linear':
        return linear_interpolate(x, x_time, t)
    elif method == 'nearest':
        return nearest_interpolate(x, x_time, t)
    elif method == 'cubic':
        return cubic_interpolate(x, x_time, t)
    elif method == 'akima':
        return akima_interpolate(x, x_time, t)
    else:
        # 默认回退行为
        t_normalized = t/(x_time[1]-x_time[0])
        t_int = torch.clamp(t_normalized.long(), 0, x.size(0) - 1)
        return x[t_int]


class ODEFunc(nn.Module):
    def __init__(self, layers, neurons, if_set_layer_potential, x, x_time, interpolation_method='linear'):
        super(ODEFunc, self).__init__()
        self.layers = layers
        self.neurons = neurons  # List of neuron instances (LIFNeuron, IFNeuron, etc.)
        self.x = x
        self.x_time = x_time
        self.if_set_layer_potential = if_set_layer_potential
        self.interpolation_method = interpolation_method

        # Initialize spike count and tensor count as zeros numpy with length to be the number of True in if_set_layer_potential
        self.spike_count = torch.zeros(sum(if_set_layer_potential))#Total count of spikes (1s) in each test batch
        self.tensor_count = torch.zeros(sum(if_set_layer_potential)) # Total count of elements in tensors in each test batch

    def forward(self, t, states):
        v_mems, spike_accum = states[:-1], states[-1]

        # Get interpolated input at time t
        current_input = interpolate(self.x, self.x_time, t, self.interpolation_method)

        dv_dt_list, current_spikes = [], current_input
        v_mem_idx, neuron_idx = 0, 0
        # they are the same, but we track them separately to indicate the index of the neuron and v_mems

        for i, layer in enumerate(self.layers):
            current_spikes = layer(current_spikes)
            if self.if_set_layer_potential[i]:
                neuron = self.neurons[neuron_idx]
                dv_dt, spike_out = neuron(v_mems[v_mem_idx], current_spikes)
                dv_dt_list.append(dv_dt)

                # if t != 0:
                #     print('torch.sum(spike_out): ', v_mem_idx, torch.sum(dv_dt), torch.sum(spike_out))

                # if self.training:  # 仅在训练初期使用
                #     if torch.sum(spike_out) < 0.1:
                #         # 以一定概率强制生成脉冲
                #         random_mask = (torch.rand_like(spike_out) < 0.1).float()
                #         spike_out = spike_out + random_mask

                # 在测试阶段统计spike_out的激活比例
                # print('self.training', self.training)
                # if True:  #
                if not self.training:
                    # 计算当前批次的spike_out中1的数量和总元素数
                    spike_count = torch.sum(spike_out).item()
                    tensor_size = spike_out.numel()

                    self.spike_count[v_mem_idx] += spike_count
                    self.tensor_count[v_mem_idx] += tensor_size
                    # print('self.spike_count, self.tensor_count: ', self.spike_count, self.tensor_count)


                current_spikes = spike_out
                v_mem_idx += 1
                neuron_idx += 1


        return (*dv_dt_list, current_spikes)



class MyTracer(Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        # treat BaseNeuron subclasses as leaf
        if isinstance(m, BaseNeuron):
            return True
        if getattr(m, "_snnwrapper_leaf_module", False):
            return True
        if isinstance(m, VotingLayer):
            return True
        if isinstance(m, ClassificationHead):
            return True
        return super().is_leaf_module(m, module_qualified_name)

def symbolic_trace_leaf_neurons(module):
    tracer = MyTracer()
    graph = tracer.trace(module)
    return fx.GraphModule(module, graph)


def remove_dead_code(m: nn.Module) -> nn.Module:
    graph = fx.Tracer().trace(m)

    # 自动删除没有被使用的节点
    graph.eliminate_dead_code()

    return fx.GraphModule(m, graph)


class ODEFuncFromFX(nn.Module):
    """Wrapper that converts a spiking neural network to ODE form.

    This class uses PyTorch FX to trace the given backbone neural network and
    constructs a new computation graph that transforms the original network into
    a form suitable for ODE solvers. Specifically, it will:
    1. Extract state differential equations from neuron layers
    2. Preserve forward propagation logic of other layers
    3. Use interpolation to handle time-continuous inputs
    4. Extract post-neuron operations into a separate module

    Args:
        backbone (nn.Module): The spiking neural network model to be converted
    """

    def __init__(self, backbone: nn.Module,
                 interpolation_method='linear'):
        super().__init__()

        self.interpolation_method = interpolation_method
        self.neuron_count = 0
        self.x = None
        self.x_time = None

        # Step 1: Symbolically trace the backbone to obtain computation graph
        traced: fx.GraphModule = symbolic_trace_leaf_neurons(backbone)
        modules = dict(traced.named_modules())

        # Step 2: Create new computation graph
        new_graph = fx.Graph()

        # Node mapping table: original graph node -> new graph node
        node_map: Dict[fx.Node, fx.Node] = {}

        # Step 3: Create input placeholders and establish mappings
        t_node = new_graph.placeholder('t')
        v_mems_node = new_graph.placeholder('v_mems')
        x_node = new_graph.placeholder('x')
        x_time_node = new_graph.placeholder('x_time')

        input_placeholder = None
        for node in traced.graph.nodes:
            if node.op == 'placeholder':
                input_placeholder = node
                node_map[node] = x_node
                break

        # Step 4: Use interpolation to get input at current time
        current_input = new_graph.call_function(
            interpolate,
            args=(x_node, x_time_node, t_node, self.interpolation_method)
        )

        # Initialize variables
        current_output = current_input
        dv_dt_list = []
        neuron_index = 0

        # Find the last neuron node
        last_neuron_node = None
        for node in traced.graph.nodes:
            if node.op == 'call_module':
                submodule = modules[node.target]
                if isinstance(submodule, BaseNeuron):
                    last_neuron_node = node
                    self.finalneuron_threshold = submodule.threshold
                    self.finalneuron_surrogate_f = submodule.surrogate_f
                    self.finalneuron_surrogate_grad_scale = submodule.surrogate_grad_scale

        # Track important nodes
        final_spike_output_node = None
        final_spike_output = None
        # Track nodes strictly after the last neuron
        nodes_after_last_neuron = []
        found_last_neuron = False

        # Step 5: FIRST PASS - Process ALL nodes to build complete graph
        for node in list(traced.graph.nodes):
            if node.op == 'placeholder' and node == input_placeholder:
                node_map[node] = current_input
                continue

            if node.op == 'call_module':
                submodule = modules[node.target]

                if isinstance(submodule, BaseNeuron):
                    # Process neuron layer
                    v_mem_current = new_graph.call_function(
                        operator.getitem,
                        args=(v_mems_node, neuron_index)
                    )

                    out_node = new_graph.call_module(
                        node.target,
                        args=(v_mem_current, current_output)
                    )

                    dv_dt = new_graph.call_function(
                        operator.getitem,
                        args=(out_node, 0)
                    )

                    spike_output = new_graph.call_function(
                        operator.getitem,
                        args=(out_node, 1)
                    )

                    dv_dt_list.append(dv_dt)
                    current_output = spike_output
                    neuron_index += 1

                    if node == last_neuron_node:
                        final_spike_output_node = node
                        final_spike_output = spike_output
                        found_last_neuron = True

                else:
                    # Process non-neuron layers
                    mapped_args = self._map_arguments(node.args, node_map)
                    mapped_kwargs = self._map_arguments(node.kwargs, node_map)

                    new_node = new_graph.call_module(
                        node.target,
                        args=mapped_args,
                        kwargs=mapped_kwargs
                    )
                    current_output = new_node

                    # Track nodes strictly after the last neuron
                    if found_last_neuron:
                        nodes_after_last_neuron.append(node)

                node_map[node] = current_output

            elif node.op in ['call_function', 'call_method']:
                # Process function calls and method calls
                mapped_args = self._map_arguments(node.args, node_map)
                mapped_kwargs = self._map_arguments(node.kwargs, node_map)

                if node.op == 'call_function':
                    new_node = new_graph.call_function(
                        node.target,
                        args=mapped_args,
                        kwargs=mapped_kwargs
                    )
                else:  # call_method
                    new_node = new_graph.call_method(
                        node.target,
                        args=mapped_args,
                        kwargs=mapped_kwargs
                    )

                current_output = new_node
                node_map[node] = new_node

                # Track nodes strictly after the last neuron
                if found_last_neuron:
                    nodes_after_last_neuron.append(node)

        # Step 6: SECOND PASS - Find the cut point
        cut_point_node = None
        cut_point_output = None
        post_neuron_nodes = []

        if final_spike_output_node is not None:
            # Build set of all nodes up to (but not include) the last neuron
            nodes_before_last_neuron = set()
            for node in traced.graph.nodes:
                if node == final_spike_output_node:
                    break
                # Track nodes strictly before the last neuron
                nodes_before_last_neuron.add(node)


            # Check each node after the last neuron
            for node in nodes_after_last_neuron:
                # Check if this node depends on any nodes BEFORE the last neuron
                depends_on_earlier = self._depends_on_nodes(node, nodes_before_last_neuron)

                if depends_on_earlier:
                    # This node depends on earlier nodes, update cut point
                    cut_point_node = node
                    cut_point_output = node_map[node]

            # Determine which nodes go to post-neuron module
            if cut_point_node is not None:
                # Find all nodes strictly after the cut point
                past_cut_point = False
                for node in nodes_after_last_neuron:
                    if past_cut_point:
                        post_neuron_nodes.append(node)
                    elif node == cut_point_node:
                        past_cut_point = True
            else:
                # No cut point found, all nodes after last neuron go to post-neuron
                post_neuron_nodes = nodes_after_last_neuron


        # No post-neuron nodes, use the original graph
        if cut_point_node is not None:
            output_tuple = tuple(dv_dt_list) + (cut_point_output,)
        elif final_spike_output is not None:
            output_tuple = tuple(dv_dt_list) + (final_spike_output,)
        else:
            output_tuple = tuple(dv_dt_list) + (current_output,)

        new_graph.output(output_tuple)

        print("=" * 50)
        print("ODE Graph:")
        print(new_graph)

        # Step 8: Create new GraphModule
        self.ode_gm = fx.GraphModule(traced, new_graph)
        self.ode_gm.graph.eliminate_dead_code()

        # Step 9: Build post-neuron module
        if post_neuron_nodes:
            post_neuron_graph = fx.Graph()

            # Create input placeholder
            post_input_node = post_neuron_graph.placeholder('x')

            # Node mapping for post-neuron graph
            post_node_map = {}

            # Map the input to the appropriate node
            if cut_point_node is not None:
                post_node_map[cut_point_node] = post_input_node
            else:
                post_node_map[final_spike_output_node] = post_input_node

            current_post_output = post_input_node

            # Recreate all post-neuron operations
            for node in post_neuron_nodes:
                if node.op == 'call_module':
                    mapped_args = self._map_arguments(node.args, post_node_map)
                    mapped_kwargs = self._map_arguments(node.kwargs, post_node_map)

                    new_post_node = post_neuron_graph.call_module(
                        node.target,
                        args=mapped_args,
                        kwargs=mapped_kwargs
                    )
                    current_post_output = new_post_node
                    post_node_map[node] = new_post_node

                elif node.op in ['call_function', 'call_method']:
                    mapped_args = self._map_arguments(node.args, post_node_map)
                    mapped_kwargs = self._map_arguments(node.kwargs, post_node_map)

                    if node.op == 'call_function':
                        new_post_node = post_neuron_graph.call_function(
                            node.target,
                            args=mapped_args,
                            kwargs=mapped_kwargs
                        )
                    else:  # call_method
                        new_post_node = post_neuron_graph.call_method(
                            node.target,
                            args=mapped_args,
                            kwargs=mapped_kwargs
                        )

                    current_post_output = new_post_node
                    post_node_map[node] = new_post_node

            # Set the output
            post_neuron_graph.output(current_post_output)

            # Create the post-neuron GraphModule
            self.post_neuron_module = fx.GraphModule(traced, post_neuron_graph)

            print("-" * 50)
            print("Post-Neuron Graph:")
            print(post_neuron_graph)
        else:
            self.post_neuron_module = nn.Identity()
            print("-" * 50)
            print("No post-neuron operations found, set nn.Identity()")

        self.neuron_count = neuron_index
        self.traced = traced
        self.nfe = 0

    def _depends_on_nodes(self, node: fx.Node, target_nodes: set) -> bool:
        """Check if a node depends on any nodes in the target set.

        Args:
            node: The node to check
            target_nodes: Set of nodes to check dependency against

        Returns:
            bool: True if node depends on any target nodes
        """

        def check_args(args):
            if isinstance(args, fx.Node):
                return args in target_nodes
            elif isinstance(args, (tuple, list)):
                return any(check_args(arg) for arg in args)
            elif isinstance(args, dict):
                return any(check_args(v) for v in args.values())
            return False

        # Check both args and kwargs
        depends = check_args(node.args) or check_args(node.kwargs)
        return depends

    def _map_arguments(self, args: Any, node_map: Dict[fx.Node, fx.Node]) -> Any:
        """Recursively map node references in arguments.

        Args:
            args: Original arguments (can be tuple, dict or other types)
            node_map: Node mapping table

        Returns:
            Mapped arguments
        """
        if isinstance(args, tuple):
            return tuple(node_map.get(a, a) if isinstance(a, fx.Node) else a
                         for a in args)
        elif isinstance(args, dict):
            return {k: node_map.get(v, v) if isinstance(v, fx.Node) else v
                    for k, v in args.items()}
        elif isinstance(args, fx.Node):
            return node_map.get(args, args)
        else:
            return args

    def set_inputs(self, x, x_time):
        self.x = x
        self.x_time = x_time

    def forward(self, t: float, v_mems: Tuple) -> Tuple:
        """Forward propagation function called by ODE solver.

        Args:
            t: Current time point
            v_mems: Membrane potential states of all neurons
            x: Input data

        Returns:
            tuple: (dv_dt_1, dv_dt_2, ..., dv_dt_n, final_output)
                  Contains membrane potential derivatives of all neurons and final output
        """
        x = self.x
        x_time = self.x_time
        self.nfe += 1

        return self.ode_gm(t, v_mems, x, x_time)

    def get_post_neuron_module(self):
        """Get the module containing all operations after the cut point.

        Returns:
            nn.Module: Module with post-neuron operations
        """
        return self.post_neuron_module

    def get_ode_module(self):
        """Get the ODE graph module.

        Returns:
            fx.GraphModule: The ODE computation graph module
        """
        return self.ode_gm
