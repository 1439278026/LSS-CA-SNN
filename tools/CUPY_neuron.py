import numpy as np
import torch.nn as nn

from . import base, surrogate
import torch
from typing import Callable
from .auto_cuda import neuron_kernel, cfunction, cuda_utils, configure
import math
from .auto_cuda.base import CKernel2D
import cupy
import torch.nn.functional as F
class IFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):
    def neuronal_charge(self) -> str:
        return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)
class IFNodeBPTTKernel(neuron_kernel.NeuronBPTTKernel):
    def grad_h_next_to_v(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_h_next_to_v', x=1., dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
class IFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None,
                forward_kernel: IFNodeFPTTKernel, backward_kernel: IFNodeBPTTKernel):
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset
        }
        requires_grad, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_forward(py_dict)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        neuron_kernel.NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel)


        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        backward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None


class LIFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} &', cname='decay')

    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(z=f'{self.dtype} LIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        else:
            codes = f'{self.dtype} LIFNodeFPTTKernel_temp_var = v_v_seq[t];'

        if self.decay_input:
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)

        codes += cfunction.add(z='h_seq[t]', x='LIFNodeFPTTKernel_temp_var', y='v_v_seq[t]', dtype=self.dtype)

        return codes
class LIFNodeBPTTKernel(neuron_kernel.NeuronBPTTKernel):
    def __init__(self, decay_input: bool, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} &', cname='decay')

    def grad_h_next_to_v(self) -> str:
        return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_v', x=cfunction.constant(None, x=1., dtype=self.dtype), y='decay', dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        if not self.decay_input:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
        else:
            return f'const {self.dtype} grad_h_to_x = decay;'
class LIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None, decay: float,
                forward_kernel: LIFNodeFPTTKernel, backward_kernel: LIFNodeBPTTKernel):
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'decay': decay,
        }
        requires_grad, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_forward(py_dict)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        neuron_kernel.NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel, decay=py_dict['decay'])


        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        py_dict['decay'] = ctx.decay

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')


        backward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None


        return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None, None

### PLIF ###
class ParametricLIFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} *', cname='decay')

    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(z=f'{self.dtype} LIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        else:
            codes = f'{self.dtype} LIFNodeFPTTKernel_temp_var = v_v_seq[t];'
        if self.decay_input:
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay[0]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay[0]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)

        codes += cfunction.add(z='h_seq[t]', x='LIFNodeFPTTKernel_temp_var', y='v_v_seq[t]', dtype=self.dtype)

        return codes


class ParametricLIFNodeBPTTKernel(neuron_kernel.NeuronBPTTKernel):
    def __init__(self, decay_input: bool, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} *', cname='decay')
        self.add_param(ctype=f'float *', cname='grad_decay')
        # float to avoid overflow
        self.add_param(ctype=f'const {dtype} *', cname='v_v_seq')

    def grad_h_next_to_v(self) -> str:
        return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_v', x=cfunction.constant(None, x=1., dtype=self.dtype), y='decay[0]', dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        if not self.decay_input:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
        else:
            return f'const {self.dtype} grad_h_to_x = decay[0];'

    @property
    def head(self):
        cuda_threads = 512
        # override
        codes = '''
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
        '''
        codes += fr'''
            
            __shared__ float sdata[{cuda_threads}];
        '''
        # __shared__ float sdata[{configure.cuda_threads}];\
        codes += '''
            if (index < N)
            {
                const int dt = N;
        '''

        codes += self.pre_core

        if self.reverse:
            codes += '''
                for(int t = numel - N + index; t >= 0; t -= dt)
                {
            '''
        else:
            codes += '''
                for(int t = index; t < numel; t += dt)
                {
            '''
        return codes


    @property
    def pre_core(self):
        codes = base.CodeTyper(16)
        # use float to avoid overflow
        codes.append('sdata[threadIdx.x] = 0.0f;')
        return super().pre_core + '\n' + codes.codes

    @property
    def core(self):
        core_codes = base.CodeTyper(18)
        with base.CodeBlock(core_codes):
            if self.decay_input:

                core_codes.append(cfunction.sub(z=f'{self.dtype} temp_var', x='h_seq[t]', y='v_v_seq[t]', dtype=self.dtype))
                core_codes.append(cfunction.mul(z='temp_var', x='temp_var', y='grad_h', dtype=self.dtype))
                core_codes.append(cfunction.div(z='temp_var', x='temp_var', y='decay[0]', dtype=self.dtype))

            else:
                if self.hard_reset:
                    core_codes.append(
                        cfunction.sub(z=f'{self.dtype} temp_var', x='v_reset', y='v_v_seq[t]', dtype=self.dtype))
                    core_codes.append(cfunction.mul(z='temp_var', x='temp_var', y='grad_h', dtype=self.dtype))
                else:
                    core_codes.append(
                        cfunction.mul(z=f'{self.dtype} temp_var', x='grad_h', y='v_v_seq[t]', dtype=self.dtype))
                    core_codes.append(cfunction.neg(y='temp_var', x='temp_var', dtype=self.dtype))


            if self.dtype == 'float':
                core_codes.append('sdata[threadIdx.x] += temp_var;')
            elif self.dtype == 'half2':
                core_codes.append('sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_var), __high2half(temp_var)));')
            else:
                raise NotImplementedError(self.dtype)

        return super().core + '\n' + core_codes.codes

    @property
    def tail(self):
        codes = '''
                }
        '''
        codes += self.post_core
        codes += '''
            }
            else
            {
                sdata[threadIdx.x] = 0.0f;
            }
            int threadx = blockDim.x;
            #pragma unroll
            for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
            {
            // Synchronize all thread before next loop
            __syncthreads();
            if (threadIdx.x < stride)
            {
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
            atomicAdd(grad_decay, sdata[0]);
            }
        }
        '''
        return codes


class ParametricLIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None, decay: torch.Tensor, forward_kernel: ParametricLIFNodeFPTTKernel, backward_kernel: ParametricLIFNodeBPTTKernel):
        if x_seq.dtype == torch.float16 and v_init.numel() % 2 != 0:
            raise ValueError('When using the the PLIF neuron with half2 cupy backend, the numer of neurons should be even to avoid the wrong gradient of tau caused by padding!')
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'decay': decay,
        }
        requires_grad, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_forward(py_dict)
        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')
        forward_kernel((blocks,), (threads,), py_dict)
        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None
        neuron_kernel.NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], py_dict['v_v_seq'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel, decay=py_dict['decay'])
        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward_kernel, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        py_dict['decay'] = ctx.decay
        py_dict['grad_decay'] = torch.zeros_like(ctx.decay, dtype=torch.float)
        py_dict['v_v_seq'] = ctx.saved_tensors[1]
        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')
        backward_kernel((blocks,), (threads,), py_dict)
        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None
        return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None,  py_dict['grad_decay'], None, None

class CUPYIFNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(),
                 v_threshold: float = 1., v_reset: float = 0.,
                 detach_reset: bool = True, decay_input: bool = True,
                 backend='torch', init_tau: float = 2.0):
        super().__init__()
        self.register_memory('v', v_reset)
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.decay_input = decay_input
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.step_mode = 'm'
        self.backend = backend
        init_w = - math.log(init_tau - 1.)  # tau = e^(-w) + 1
        self.w = torch.as_tensor(init_w).sigmoid().cuda()

    def multi_step_forward(self, x_seq):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x_seq[0])
        if x_seq.dtype == torch.float:
            dtype = 'float'
        elif x_seq.dtype == torch.half:
            dtype = 'half2'
        else:
            raise ValueError('x_seq.dtype should be float or half2')
        forward_kernel = IFNodeFPTTKernel(hard_reset=True, dtype=dtype)
        backward_kernel = IFNodeBPTTKernel(surrogate_function=self.surrogate_function.cuda_codes,
                                    hard_reset=True, detach_reset=self.detach_reset, dtype=dtype)
        # All tensors wil be regard as 2D or 1D. Thus, we use flatten
        spike_seq, v_seq = IFNodeATGF.apply(x_seq.flatten(1),
                                    self.v.flatten(), self.v_threshold, self.v_reset,
                                     forward_kernel, backward_kernel)
        spike_seq = spike_seq.view(x_seq.shape)
        self.v = v_seq[-1].view(x_seq.shape[1:])
        return spike_seq
class CUPYLIFNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(),
                 v_threshold: float = 1., v_reset: float = 0.,
                 detach_reset: bool = True, decay_input: bool = True,
                 backend='torch', init_tau: float = 2.0):
        super().__init__()
        self.register_memory('v', v_reset)
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.decay_input = decay_input
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.step_mode = 'm'
        self.backend = backend
        init_w = - math.log(init_tau - 1.)  # tau = e^(-w) + 1
        self.w = torch.as_tensor(init_w).sigmoid().cuda()

    def multi_step_forward(self, x_seq):
        if x_seq.ndimension()  == 3:
            x_seq = x_seq.permute(2, 0, 1)
        elif x_seq.ndimension()  == 4:
            x_seq = x_seq.permute(3, 0, 1, 2)
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x_seq[0])
        if x_seq.dtype == torch.float:
            dtype = 'float'
        elif x_seq.dtype == torch.half:
            dtype = 'half2'
        else:
            raise ValueError('x_seq.dtype should be float or half2')
        forward_kernel = LIFNodeFPTTKernel(decay_input=self.decay_input,
                                                     hard_reset=True, dtype=dtype)
        backward_kernel = LIFNodeBPTTKernel(decay_input=self.decay_input,
                                    surrogate_function=self.surrogate_function.cuda_codes,
                                    hard_reset=True, detach_reset=self.detach_reset, dtype=dtype)
        # All tensors wil be regard as 2D or 1D. Thus, we use flatten
        spike_seq, v_seq = LIFNodeATGF.apply(x_seq.flatten(1),
                                    self.v.flatten(), self.v_threshold, self.v_reset,
                                    self.w, forward_kernel, backward_kernel)
        spike_seq = spike_seq.view(x_seq.shape)
        self.v = v_seq[-1].view(x_seq.shape[1:])
        if x_seq.ndimension()  == 3:
            spike_seq = spike_seq.permute(1,2,0)
        elif  x_seq.ndimension()  == 4:
            spike_seq = spike_seq.permute(1,2,3,0)
        return spike_seq
class CUPYPLIFNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(),
                 v_threshold: float = 1., v_reset: float = 0.,
                 detach_reset: bool = True, decay_input: bool = True,
                 backend='torch', init_tau: float = 2.0):
        super().__init__()
        self.register_memory('v', v_reset)
        self.detach_reset = detach_reset
        self.decay_input = decay_input
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.step_mode = 'm'
        self.backend = backend
        init_w = - math.log(init_tau - 1.)  # tau = e^(-w) + 1
        self.w = torch.as_tensor(init_w).sigmoid().cuda()

        self.v_threshold = v_threshold
    def multi_step_forward(self, x_seq):
        if x_seq.ndimension()  == 3:
            x_seq = x_seq.permute(2, 0, 1)
        elif x_seq.ndimension()  == 4:
            x_seq = x_seq.permute(3, 0, 1, 2)
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x_seq[0])
        if x_seq.dtype == torch.float:
            dtype = 'float'
        elif x_seq.dtype == torch.half:
            dtype = 'half2'
        else:
            raise ValueError('x_seq.dtype should be float or half2')
        forward_kernel = ParametricLIFNodeFPTTKernel(decay_input=self.decay_input,
                                                     hard_reset=True, dtype=dtype)
        backward_kernel = ParametricLIFNodeBPTTKernel(decay_input=self.decay_input,
                                    surrogate_function=self.surrogate_function.cuda_codes,
                                    hard_reset=True, detach_reset=self.detach_reset, dtype=dtype)
        # All tensors wil be regard as 2D or 1D. Thus, we use flatten
        spike_seq, v_seq = ParametricLIFNodeATGF.apply(x_seq.flatten(1),
                                    self.v.flatten(), self.v_threshold, self.v_reset,
                                    self.w, forward_kernel, backward_kernel)
        spike_seq = spike_seq.view(x_seq.shape)
        self.v = v_seq[-1].view(x_seq.shape[1:])
        if x_seq.ndimension()  == 3:
            spike_seq = spike_seq.permute(1,2,0)
        elif  x_seq.ndimension()  == 4:
            spike_seq = spike_seq.permute(1,2,3,0)
        return spike_seq

class BNAndPadLayer(torch.nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = torch.nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps
class RepConv(torch.nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = torch.nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            torch.nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            torch.nn.BatchNorm2d(out_channel),
        )

        self.body = torch.nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)
# class MS_Attention_RepConv_qkv_id(torch.nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads=8,
#         qkv_bias=False,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         sr_ratio=1,
#     ):
#         super().__init__()
#         assert (
#             dim % num_heads == 0
#         ), f"dim {dim} should be divided by num_heads {num_heads}."
#         self.dim = dim
#         self.num_heads = num_heads
#         self.scale = 0.125
#
#         self.head_lif = CUPYPLIFNode()
#
#         self.q_conv = torch.nn.Sequential(RepConv(dim, dim, bias=False), torch.nn.BatchNorm2d(dim))
#
#         self.k_conv = torch.nn.Sequential(RepConv(dim, dim, bias=False), torch.nn.BatchNorm2d(dim))
#
#         self.v_conv = torch.nn.Sequential(RepConv(dim, dim, bias=False), torch.nn.BatchNorm2d(dim))
#
#         self.q_lif = CUPYPLIFNode()
#
#         self.k_lif = CUPYPLIFNode()
#
#         self.v_lif = CUPYPLIFNode()
#
#         self.attn_lif = CUPYPLIFNode(v_threshold=0.5)
#
#         self.proj_conv = torch.nn.Sequential(
#             RepConv(dim, dim, bias=False), torch.nn.BatchNorm2d(dim)
#         )
#
#     def forward(self, x):
#         # T, B, C, H, W = x.shape
#         # N = H * W
#         B, C, _, T = x.shape
#         # x = self.head_lif(x)
#
#         q = self.q_conv(x)  # 形状为 (B, C, 1, T)
#         k = self.k_conv(x)  # 形状为 (B, C, 1, T)
#         v = self.v_conv(x)  # 形状为 (B, C, 1, T)
#         q = self.q_lif(q).permute(3,0,1,2)   # 形状为 (T, B, C, 1)
#         k = self.k_lif(k).permute(3,0,1,2)
#         v = self.v_lif(v).permute(3,0,1,2)
#         q = (
#             q.transpose(-1, -2)
#             .reshape(T, B, 1, self.num_heads, C // self.num_heads)
#             .permute(0, 1, 3, 2, 4)
#             .contiguous()
#         )
#         k = (
#             k.transpose(-1, -2)
#             .reshape(T, B, 1, self.num_heads, C // self.num_heads)
#             .permute(0, 1, 3, 2, 4)
#             .contiguous()
#         )
#
#         v = (
#             v.transpose(-1, -2)
#             .reshape(T, B, 1, self.num_heads, C // self.num_heads)
#             .permute(0, 1, 3, 2, 4)
#             .contiguous()
#         )
#
#         x = k.transpose(-2, -1) @ v
#         x = (q @ x) * self.scale
#
#         x = x.transpose(3, 4).reshape(B, C, 1, T).contiguous()
#         x = self.attn_lif(x)
#         x = self.proj_conv(x)
#
#         # q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
#         # k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
#         # v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
#
#         # q = self.q_lif(q).flatten(3)
#         # q = (
#         #     q.transpose(-1, -2)
#         #     .reshape(T, B, N, self.num_heads, C // self.num_heads)
#         #     .permute(0, 1, 3, 2, 4)
#         #     .contiguous()
#         # )
#         #
#         # k = self.k_lif(k).flatten(3)
#         # k = (
#         #     k.transpose(-1, -2)
#         #     .reshape(T, B, N, self.num_heads, C // self.num_heads)
#         #     .permute(0, 1, 3, 2, 4)
#         #     .contiguous()
#         # )
#         #
#         # v = self.v_lif(v).flatten(3)
#         # v = (
#         #     v.transpose(-1, -2)
#         #     .reshape(T, B, N, self.num_heads, C // self.num_heads)
#         #     .permute(0, 1, 3, 2, 4)
#         #     .contiguous()
#         # )
#         #
#         # x = k.transpose(-2, -1) @ v
#         # x = (q @ x) * self.scale
#         #
#         # x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
#         # x = self.attn_lif(x).reshape(T, B, C, H, W)
#         # x = x.reshape(T, B, C, H, W)
#         # x = x.flatten(0, 1)
#         # x = self.proj_conv(x).reshape(T, B, C, H, W)
#
#         return x


from spikingjelly.clock_driven.neuron import MultiStepLIFNode
class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.0625

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
        )

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):

        # T, B, C, H, W = x.shape
        B,C,_,T = x.shape
        x = x.unsqueeze(2).permute(4,0,1,2,3)

        N,H,W = 1,1,1

        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)
        x = x.squeeze(-1).permute(1,2,3,0)
        return x