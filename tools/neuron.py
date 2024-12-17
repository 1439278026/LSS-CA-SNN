from . import base, surrogate, neuron_kernel, lava_exchange
from .auto_cuda import cfunction
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Callable
import math
import numpy as np
import cupy
# try:
#     import lava.lib.dl.slayer as slayer
# except BaseException as e:
#     cupy = None
#     neuron_kernel = None
#     cu_kernel_opt = None
def check_backend(backend: str):
    if backend == 'torch':
        return
    elif backend == 'cupy':
        assert cupy is not None, 'CuPy is not installed! You can install it from "https://github.com/cupy/cupy".'
    # elif backend == 'lava':
    #     assert slayer is not None, 'Lava-DL is not installed! You can install it from "https://github.com/lava-nc/lava-dl".'
    else:
        raise NotImplementedError(backend)
class IzhikevichNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(), u_threshold: float = 1., tau: float = 2.,
                 detach_reset: bool = True, step_mode='m', backend='torch', store_u_seq: bool = False,
                 k: float = 1, a: float = 0.002, b: float = 0.02, c: float = 0, d: float = 0.2, ur: float = -0.05,
                 learn_a: bool = False, learn_b: bool = False, learn_tau: bool = False, learn_d: bool = False):
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', 0.)  # 补充电压
        self.register_memory('u', -0.05)  # 膜电压

        self.ur = ur
        self.u_threshold = u_threshold
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_u_seq = store_u_seq

        if learn_tau:
            self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float))
        else:
            self.tau = tau

        if learn_a:
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float))
        else:
            self.a = a
        if learn_b:
            self.b = nn.Parameter(torch.tensor(b, dtype=torch.float))
        else:
            self.b = b
        self.c = c
        if learn_d:
            self.d = nn.Parameter(torch.tensor(d, dtype=torch.float))
        else:
            self.d = d
        self.k = k

    @property
    def store_u_seq(self):
        return self._store_u_seq

    @store_u_seq.setter
    def store_u_seq(self, value: bool):
        self._store_u_seq = value
        if value:
            if not hasattr(self, 'u_seq'):
                self.register_memory('u_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.u = self.u + (self.k * (self.u - self.ur) * (self.u - self.u_threshold) - self.v + x) / self.tau
        self.v = self.v + self.a * (self.b * (self.u - self.ur) - self.v) / self.tau

    def neuronal_fire(self):
        return self.surrogate_function(self.u - self.u_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.u = (1. - spike_d) * self.u + spike_d * self.c
        self.v = self.v + spike_d * self.d

    def extra_repr(self):
        return f'self.a={self.a}, self.b={self.b}, self.c={self.c}, self.d={self.d}, self.k={self.k}, self.tau={self.tau}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_u_seq:
            u_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_u_seq:
                u_seq.append(self.u)
        if self.store_u_seq:
            self.u_seq = torch.stack(u_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.u, float):
            v_init = self.v
            u_init = self.u
            self.v = torch.full_like(x.data, v_init)
            self.u = torch.full_like(x.data, u_init)



class RAFNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(), v_threshold: float = 1.,
                 detach_reset: bool = True, step_mode='m', backend='torch', store_v_seq: bool = False,
                 q: float = 0.65, w: float = 0.5, detach_convey: bool = False, input_size: tuple = (2, 128, 128)):
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', 0.)
        self.register_memory('u1', 0.)
        self.register_memory('u2', 0.)
        self.register_memory('u3', 0.)

        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.detach_convey = detach_convey
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq
        self.w = torch.tensor(w, dtype=torch.float)
        self.q = torch.tensor(q, dtype=torch.float)
        # self.w = nn.Parameter(torch.tensor(w, dtype=torch.float))
        # self.q = nn.Parameter(torch.tensor(q, dtype=torch.float))

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.w * self.v + (1 - self.w) * x - self.q * self.u3 + (self.q + self.q ** 2) * self.u2 - (self.q ** 2 + self.q ** 3) * self.u1

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike, x, u2, u3):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        if self.detach_convey:
            x_d = x.detach()
            u2_d = u2.detach()
            u3_d = u3.detach()
        else:
            x_d = x
            u2_d = u2
            u3_d = u3
        self.v = (1. - spike_d) * self.v
        self.u1 = (1. - spike_d) * u2_d
        # self.u2 = (1. - spike_d) * x_d
        self.u2 = (1. - spike_d) * u3_d
        self.u3 = (1. - spike_d) * x_d

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, self.q={self.q}, self.w={self.w}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike, x, self.u2, self.u3)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)
            self.u1 = torch.full_like(x.data, v_init)
            self.u2 = torch.full_like(x.data, v_init)
            self.u3 = torch.full_like(x.data, v_init)


class IFNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(), v_threshold: float = 1.,
                 detach_reset: bool = False, step_mode='m', backend='torch', store_v_seq: bool = False,
                 ):
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', 0.)
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = (1. - spike_d) * self.v

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)
class MultiStepIFNode(IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, backend='torch', lava_s_cale=1 << 6):
        """
        * :ref:`API in English <MultiStepIFNode.__init__-en>`

        .. _MultiStepIFNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param backend: 使用哪种计算后端，可以为 ``'torch'`` 或 ``'cupy'``。``'cupy'`` 速度更快，但仅支持GPU。
        :type backend: str

        多步版本的 :class:`spikingjelly.clock_driven.neuron.IFNode`。

        .. tip::

            对于多步神经元，输入 ``x_seq.shape = [T, *]``，不仅可以使用 ``.v`` 和 ``.spike`` 获取 ``t = T - 1`` 时刻的电压和脉冲，还能够
            使用 ``.v_seq`` 和 ``.spike_seq`` 获取完整的 ``T`` 个时刻的电压和脉冲。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepIFNode.__init__-cn>`

        .. _MultiStepIFNode.__init__-en:

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param backend: use which backend, ``'torch'`` or ``'cupy'``. ``'cupy'`` is faster but only supports GPU
        :type backend: str

        The multi-step version of :class:`spikingjelly.clock_driven.neuron.IFNode`.

        .. admonition:: Tip
            :class: tip

            The input for multi-step neurons are ``x_seq.shape = [T, *]``. We can get membrane potential and spike at
            time-step ``t = T - 1`` by ``.v`` and ``.spike``. We can also get membrane potential and spike at all ``T``
            time-steps by ``.v_seq`` and ``.spike_seq``.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.

        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

        self.register_memory('v_seq', None)

        check_backend(backend)

        self.backend = backend

        self.lava_s_cale = lava_s_cale

        if backend == 'lava':
            self.lava_neuron = self.to_lava()
        else:
            self.lava_neuron = None


    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        elif self.backend == 'lava':
            if self.lava_neuron is None:
                self.lava_neuron = self.to_lava()

            spike, self.v = lava_exchange.lava_neuron_forward(self.lava_neuron, x_seq, self.v)

            return spike

        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

    def to_lava(self):
        return lava_exchange.to_lava_neuron(self)

    def reset(self):
        super().reset()
        if self.lava_neuron is not None:
            self.lava_neuron.current_state.zero_()
            self.lava_neuron.voltage_state.zero_()

class LIFNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(), v_threshold: float = 1.,
                 detach_reset: bool = True, step_mode='m', backend='torch', store_v_seq: bool = False,
                 w: float = 0.5, train: bool = False):
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', 0.)

        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq
        if train:
            self.w = nn.Parameter(torch.tensor(w, dtype=torch.float))
        else:
            self.w = w

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.w * self.v + (1 - self.w) * x

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = (1. - spike_d) * self.v

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, self.w={self.w}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)

class MultiStepLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch', lava_s_cale=1 << 6):
        """
        * :ref:`API in English <MultiStepLIFNode.__init__-en>`

        .. _MultiStepLIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param decay_input: 输入是否会衰减
        :type decay_input: bool

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param backend: 使用哪种计算后端，可以为 ``'torch'`` 或 ``'cupy'``。``'cupy'`` 速度更快，但仅支持GPU。
        :type backend: str

        多步版本的 :class:`spikingjelly.clock_driven.neuron.LIFNode`。

        .. tip::

            对于多步神经元，输入 ``x_seq.shape = [T, *]``，不仅可以使用 ``.v`` 和 ``.spike`` 获取 ``t = T - 1`` 时刻的电压和脉冲，还能够
            使用 ``.v_seq`` 和 ``.spike_seq`` 获取完整的 ``T`` 个时刻的电压和脉冲。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepLIFNode.__init__-cn>`

        .. _MultiStepLIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param backend: use which backend, ``'torch'`` or ``'cupy'``. ``'cupy'`` is faster but only supports GPU
        :type backend: str

        The multi-step version of :class:`spikingjelly.clock_driven.neuron.LIFNode`.

        .. admonition:: Tip
            :class: tip

            The input for multi-step neurons are ``x_seq.shape = [T, *]``. We can get membrane potential and spike at
            time-step ``t = T - 1`` by ``.v`` and ``.spike``. We can also get membrane potential and spike at all ``T``
            time-steps by ``.v_seq`` and ``.spike_seq``.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.

        """
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)

        check_backend(backend)

        self.backend = backend

        self.lava_s_cale = lava_s_cale

        if backend == 'lava':
            self.lava_neuron = self.to_lava()
        else:
            self.lava_neuron = None

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.decay_input, self.tau, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        elif self.backend == 'lava':
            if self.lava_neuron is None:
                self.lava_neuron = self.to_lava()

            spike, self.v = lava_exchange.lava_neuron_forward(self.lava_neuron, x_seq, self.v)

            return spike

        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

    def to_lava(self):
        return lava_exchange.to_lava_neuron(self)

    def reset(self):
        super().reset()
        if self.lava_neuron is not None:
            self.lava_neuron.current_state.zero_()
            self.lava_neuron.voltage_state.zero_()

class Cuba_LIFNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(), v_threshold: float = 1.,
                 detach_reset: bool = False, step_mode='m', backend='torch',
                 w: float = 0.5, alpha: float = 0.5, train: bool = False, channels: int=22):
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', 0.)
        self.register_memory('i', 0.)
        self.register_memory('spike', 0.)

        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        if train:
            self.w = nn.Parameter(torch.tensor(w, dtype=torch.float))
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float))
        else:
            self.w = w
            self.alpha = alpha
        self.fc = nn.Sequential(nn.Linear(channels, channels))  # N, C

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.i = self.alpha * self.i + x + self.fc(self.spike)
        self.v = self.w * self.v + (1 - self.w) * self.i

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = (1. - spike_d) * self.v

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, self.w={self.w}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            self.spike = self.single_step_forward(x_seq[t])
            y_seq.append(self.spike)
        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)
            self.i = torch.full_like(x.data, 0)
            self.spike = torch.full_like(x.data, 0)


class MANode(base.MemoryModule):
    def __init__(self, batchsize, time_step, channel, surrogate_function: Callable = surrogate.Sigmoid(), v_threshold: float = 1.,
                 detach_reset: bool = True, step_mode='m', backend='torch', store_v_seq: bool = False, train: bool = True
                 ):
        assert isinstance(detach_reset, bool)
        super().__init__()
        self.register_memory('v', 0.)
        self.N = batchsize
        self.T = time_step
        self.C = channel
        self.avg_pool_TA = nn.AdaptiveAvgPool3d(1)
        self.max_pool_TA = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP_TA = nn.Sequential(
            nn.Conv3d(self.T, self.T // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(self.T // 16, self.T, 1, bias=False),
        )
        self.avg_pool_CA = nn.AdaptiveAvgPool2d(1)
        self.max_pool_CA = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP_CA = nn.Sequential(
            nn.Conv2d(self.C, self.C // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.C // 16, self.C, 1, bias=False),
        )
        self.conv_SA = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_TA = nn.Sigmoid()
        self.sigmoid_CA = nn.Sigmoid()
        self.sigmoid_SA = nn.Sigmoid()
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend
        if train:
            self.w = nn.Parameter(torch.tensor(0, dtype=torch.float))
        else:
            self.w = torch.tensor(0, dtype=torch.float)
        self.store_v_seq = store_v_seq

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x - self.v) * self.w.sigmoid()

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = (1. - spike_d) * self.v

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}'

    def single_step_forward(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)  # [N, C, H, W]
        self.neuronal_charge(x)
        self.v = self.v * self.channel_attention(self.v)
        self.v = self.v * self.spatio_attention(self.v)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        self.N = x_seq.shape[1]
        spikes = []
        x_seq = self.time_attention(x_seq)
        for t in range(T):
            spike = self.single_step_forward(x_seq[t])
            spikes.append(spike)
        return torch.stack(spikes)

    def time_attention(self, x_seq):
        x_seq = x_seq.transpose(0, 1)  # N, T, C, H, W
        avg_max_out = self.sharedMLP_TA(torch.cat([self.avg_pool_TA(x_seq), self.max_pool_TA(x_seq)], dim=0))  # N, T, 1, 1, 1 -> 2N, T, 1, 1, 1
        x_seq_TA = x_seq * self.sigmoid_TA(avg_max_out[:self.N] + avg_max_out[self.N:])  # 2N, T, 1, 1, 1 -> 两个 N, T, 1, 1, 1
        return x_seq_TA.transpose(0, 1)

    def channel_attention(self, v):  # N, C, H, W
        avg_max_out = self.sharedMLP_CA(torch.cat([self.avg_pool_CA(v), self.max_pool_CA(v)], dim=0))  # N, C, 1, 1 -> 2N, C, 1, 1
        CA = self.sigmoid_CA(avg_max_out[:self.N] + avg_max_out[self.N:])  # 2N, C, 1, 1 -> 两个 N, C, 1, 1
        return CA

    def spatio_attention(self, v):  # N, C, H, W
        avgout = torch.mean(v, dim=1, keepdim=True)  # N, 1, H, W
        maxout, _ = torch.max(v, dim=1, keepdim=True)
        SA = self.sigmoid_SA(self.conv_SA(torch.cat([avgout, maxout], dim=1)))
        return SA


class MyLIAFNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(), v_threshold: float = 1., channels: int = 22,
                 detach_reset: bool = False, step_mode='m', backend='torch', w: float = 0.5, train: bool = False):
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', 0.)

        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend
        self.fc = nn.Linear(channels, channels)
        if train:
            self.w = nn.Parameter(torch.tensor(w, dtype=torch.float))
        else:
            self.w = w

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.w * self.v + (1-self.w) * x

    def neuronal_fire(self):
        return self.heaviside(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        spike_d = spike.detach()
        self.v = (1. - spike_d) * self.v

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, self.w={self.w}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        v_seq = []
        for t in range(T):
            self.single_step_forward(x_seq[t])
            v_seq.append(self.fc(self.v))
        return torch.stack(v_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)

    def heaviside(self, x: torch.Tensor):
        return (x >= 0).to(x)


class AttentionNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(), v_threshold: float = 1.,
                 detach_reset: bool = True, step_mode='m', backend='torch', store_v_seq: bool = False,
                 train: bool = False, timestep: int = 50, batchsize: int = 32):
        assert isinstance(detach_reset, bool)
        super().__init__()
        self.register_memory('v', 0.)
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend
        if train:
            self.w = nn.Parameter(torch.tensor(0, dtype=torch.float))
        else:
            self.w = torch.tensor(0, dtype=torch.float)
        self.store_v_seq = store_v_seq
        self.T = timestep
        self.N = batchsize
        self.lin = nn.Linear(self.T, self.T, bias=False)

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x - self.v) * self.w.sigmoid()

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = (1. - spike_d) * self.v

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}'

    def single_step_forward(self, x: torch.Tensor, t):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x[0].squeeze().data, v_init)  # [N, C]
        if t != 0:
            input = []
            mean = torch.mean(x[:t+1], dim=-1).reshape(self.N, -1)  # [t+1, N, C] -> [t+1, N, 1] -> [N, t+1]
            if t != self.T - 1:
                mean = torch.cat((mean, torch.zeros((self.N, self.T - t - 1)).cuda()), dim=1)  # [N, t+1] + [N, T-t-1] -> [N, T]
            attention = torch.softmax(self.lin(mean), dim=1)  # [N, T] -> [N, T]
            for i in range(x.shape[1]):  # 遍历batch
                input.append(torch.mm(attention[i, :].unsqueeze(0), x[:, i, :]).squeeze())  # [1, T] * [T, C] -> [1, C]
            input = torch.stack(input)
        else:
            input = x[0].squeeze()
        self.neuronal_charge(input)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        spikes = []
        for t in range(T):
            spike = self.single_step_forward(x_seq, t)
            spikes.append(spike)
        return torch.stack(spikes)


class ParametricLIFNode(base.BaseNode):
    def __init__(self, init_tau: float = 2.0, decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = True, step_mode='s', backend='torch', store_v_seq: bool = False):
        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.decay_input = decay_input
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - self.w.sigmoid()) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * self.w.sigmoid() + x

    def multi_step_forward(self, x_seq: torch.Tensor):
        return super().multi_step_forward(x_seq)

class MultiStepParametricLIFNode(ParametricLIFNode):
    def __init__(self, init_tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch'):
        """
        * :ref:`API in English <MultiStepParametricLIFNode.__init__-en>`

        .. _MultiStepParametricLIFNode.__init__-cn:

        :param init_tau: 膜电位时间常数的初始值
        :type init_tau: float

        :param decay_input: 输入是否会衰减
        :type decay_input: bool

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        多步版本的 `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_
        提出的 Parametric Leaky Integrate-and-Fire (PLIF)神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})

        其中 :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`，:math:`w` 是可学习的参数。

        .. tip::

            对于多步神经元，输入 ``x_seq.shape = [T, *]``，不仅可以使用 ``.v`` 和 ``.spike`` 获取 ``t = T - 1`` 时刻的电压和脉冲，还能够
            使用 ``.v_seq`` 和 ``.spike_seq`` 获取完整的 ``T`` 个时刻的电压和脉冲。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepParametricLIFNode.__init__-cn>`

        .. _MultiStepParametricLIFNode.__init__-en:

        :param init_tau: the initial value of membrane time constant
        :type init_tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param backend: use which backend, ``'torch'`` or ``'cupy'``. ``'cupy'`` is faster but only supports GPU
        :type backend: str

        The multi-step Parametric Leaky Integrate-and-Fire (PLIF) neuron, which is proposed by `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_ and can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})

        where :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`, :math:`w` is a learnable parameter.

        .. admonition:: Tip
            :class: tip

            The input for multi-step neurons are ``x_seq.shape = [T, *]``. We can get membrane potential and spike at
            time-step ``t = T - 1`` by ``.v`` and ``.spike``. We can also get membrane potential and spike at all ``T``
            time-steps by ``.v_seq`` and ``.spike_seq``.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.
        """
        super().__init__(init_tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)

        check_backend(backend)

        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)


            spike_seq, self.v_seq = neuron_kernel.MultiStepParametricLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.w.sigmoid(), self.decay_input, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq
        else:
            raise NotImplementedError

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'
class QIFNode(base.BaseNode):
    def __init__(self, tau: float = 2., v_c: float = 0.8, a0: float = 1., v_threshold: float = 1., v_rest: float = 0.,
                 v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = True, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.tau = tau
        self.v_c = v_c
        self.v_rest = v_rest
        self.a0 = a0

    # @property
    # def store_v_seq(self):
    #     return self._store_v_seq

    # @store_v_seq.setter
    # def store_v_seq(self, value: bool):
    #     self._store_v_seq = value
    #     if value:
    #         if not hasattr(self, 'v_seq'):
    #             self.register_memory('v_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c)) / self.tau

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = (1. - spike_d) * self.v

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, v_c={self.v_c}, a0={self.a0}, v_rest={self.v_rest}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        # y_seq = []
        y_seq = torch.zeros_like(x_seq).cuda()
        # if self.store_v_seq:
        #     v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            # y_seq.append(y)
            y_seq[t] = y
            # if self.store_v_seq:
            #     v_seq.append(self.v)
        # if self.store_v_seq:
        #     self.v_seq = torch.stack(v_seq)
        # return torch.stack(y_seq)
        return y_seq

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class EIFNode(base.BaseNode):
    def __init__(self, tau: float = 2., delta_T: float = 1., theta_rh: float = .8, v_threshold: float = 1.,
                 v_rest: float = 0., v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert delta_T > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.tau = tau
        self.delta_T = delta_T
        self.v_rest = v_rest
        self.theta_rh = theta_rh

    # @property
    # def store_v_seq(self):
    #     return self._store_v_seq

    # @store_v_seq.setter
    # def store_v_seq(self, value: bool):
    #     self._store_v_seq = value
    #     if value:
    #         if not hasattr(self, 'v_seq'):
    #             self.register_memory('v_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.v_rest - self.v + self.delta_T * torch.exp((self.v - self.theta_rh) / self.delta_T)) / self.tau

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = (1. - spike_d) * self.v

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, delta_T={self.delta_T}, theta_rh={self.theta_rh}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        # if self.store_v_seq:
        #     v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
        #     if self.store_v_seq:
        #         v_seq.append(self.v)
        # if self.store_v_seq:
        #     self.v_seq = torch.stack(v_seq)
        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)