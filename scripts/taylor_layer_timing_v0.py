#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from time import time
from collections import OrderedDict
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributed import ReduceOp
from colossalai.communication import broadcast
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.kernel import LayerNorm
from colossalai.nn import init as init
from colossalai.registry import LAYERS
from colossalai.communication.collective import (all_gather, all_reduce, reduce, reduce_scatter)
from colossalai.utils.checkpointing import (broadcast_state_dict, gather_tensor_parallel_state_dict,
                                            partition_tensor_parallel_state_dict)
from colossalai.utils.cuda import get_current_device
from colossalai.utils import print_rank_0, MultiTimer
from torch import Tensor
from torch.nn.parameter import Parameter
from ..vanilla import VanillaPatchEmbedding, VanillaLayerNorm

from ..base_layer import ParallelLayer
from ..colossalai_layer._utils import ColossalaiModule
from ..utils import divide, set_tensor_parallel_attribute_by_partition
from ._utils import (gather_forward_split_backward, get_parallel_input, reduce_grad, _gather, reduce_input, set_parallel_input,
                     split_forward_gather_backward)
import torch.autograd.profiler as profiler


@LAYERS.register_module
class TaylorAttn(ParallelLayer):
    r""" Linear layer with row parallelism

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
    """

    def __init__(self,
                #  in_seq_length: int, # n
                #  out_features: int,
                 hidden_dim: int, # d (width of Q, K, V)
                 num_heads: int,
                 qkv_bias: bool = False,
                 dtype: torch.dtype = None,
                 parallel_input: bool = False,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()

        assert hidden_dim % num_heads == 0, 'dim should be divisible by num_heads'
        # Keep input parameters
        # self.in_seq_length = in_seq_length
        # self.out_features = out_features
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        head_dim = hidden_dim // num_heads
        self.scale = head_dim ** -0.5
        self.parallel_input = parallel_input
        
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}

        # Divide the weight matrix along the last dimension.
        # self.input_size_per_partition = divide(in_seq_length, gpc.tensor_parallel_size)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=qkv_bias, device=factory_kwargs['device'])

        # Parameters.
        # Initialize weight.
        # self.weight = Parameter(torch.empty(self.out_features, self.input_size_per_partition, **factory_kwargs))

        # if bias:
        #     self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        # else:
        #     self.bias = None
        # with seed(ParallelMode.TENSOR):
        #     self.reset_parameters(weight_initializer, bias_initializer)
        # self._set_tensor_parallel_attributes()
        set_parallel_input(False)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.out_features
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)
            broadcast(self.bias, gpc.get_ranks_in_group(ParallelMode.PARALLEL_1D)[0], ParallelMode.PARALLEL_1D)

    def _set_tensor_parallel_attributes(self):
        num_partition = gpc.get_world_size(ParallelMode.TENSOR)
        set_tensor_parallel_attribute_by_partition(self.weight, num_partition)

    def forward(self, input_: Tensor) -> Tensor:
        # Set up backprop all-reduce.
        
        DEBUG = True
        comp_times = []
        comm_times = []
        if DEBUG: print_rank_0(f"forward step(-1), input_.shape: {input_.shape}")
        
        if self.parallel_input:
            assert input_.shape[-1] == self.weight.shape[-1], \
                'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1])
            input_ = input_
        else:
            # assert divide(input_.shape[-2], gpc.tensor_parallel_size) == self.weight.shape[-1], \
            #     'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
            #     input_.shape, self.weight.shape, self.weight.shape[-1] * gpc.tensor_parallel_size)
            input_ = split_forward_gather_backward(input_, ParallelMode.PARALLEL_1D, dim=-2) # -2 is the seq length(n) dimension

        if DEBUG: print_rank_0(f"forward step(0), input_.shape: {input_.shape}")
        start = time()
        B, N, C = input_.shape # Batch, Seq_len, C
        qkv = self.qkv(input_).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   
        comp_times.append(time() - start)
        
        # Step 1: Mean-centering keys
        start = time()
        k_mean = torch.mean(k, dim = 2).unsqueeze(2)
        comp_times.append(time() - start) # step 1a
        
        start = time()
        k_mean = all_reduce(k_mean, ParallelMode.PARALLEL_1D, op=ReduceOp.SUM)
        comm_times.append(time() - start) # reduce 1 
        
        start = time()
        k_hat = k - k_mean  
        comp_times[-1] += time() - start # step 1 total time
        
        if DEBUG: print_rank_0(f"forward step(1), k_hat.shape: {k_hat.shape}")
        
        # Step 2: Global context matrix
        start = time()
        g = k_hat.transpose(-2, -1) @ v
        comp_times.append(time() - start) # step 2 
        if DEBUG: print_rank_0(f"forward step(2), g.shape: {g.shape}")
        
        
        # Step 3: Column sum of keys, values
        start = time()
        k_sum = torch.sum(k_hat, dim=2, keepdim=True) # 3a ks, vs
        v_sum = torch.sum(v, dim=2, keepdim=True)
        comp_times.append(time() - start) # step 3
        
        start = time()
        g_ks_vs = torch.cat((g, k_sum, v_sum), -2) # 3b: concat, all-reduce
        g_ks_vs = all_reduce(g_ks_vs, ParallelMode.PARALLEL_1D, op=ReduceOp.SUM) # 3b all-reduce
        g, k_sum, v_sum = g_ks_vs[:,:,:-2,:], g_ks_vs[:,:,-2,:].unsqueeze(-2), g_ks_vs[:,:,-1,:].unsqueeze(-2)
        comm_times.append(time() - start) # reduce2 (cat, reduce, unpack)
        if DEBUG: print_rank_0(f"forward step(3), g: {g.shape}, k_sum:{k_sum.shape}, v_sum:{v_sum.shape}")
        
        # Step 4: Compute Taylor denominator
        start = time()
        q_ksum_T = q @ k_sum.transpose(-2, -1)
        t_D = torch.ones_like(q_ksum_T, device=q.device) * N * self.scale + q_ksum_T
        comp_times.append(time() - start) # step 4 time
        if DEBUG: print_rank_0(f"forward step(4), t_D.shape: {t_D.shape}")
        
        # Step 5: Compute Taylor numerator
        start = time()
        T_N = torch.ones((B, q.shape[1], N, 1), device=q.device) * self.scale + q @ g
        comp_times.append(time() - start) # step 5 time
        if DEBUG: print_rank_0(f"forward step(5), T_N.shape: {T_N.shape}")
        
        
        # Step 6: Taylor Attention score
        start = time()
        z = (1 / t_D) * T_N
        comp_times.append(time() - start) # step 6 time
        
        start = time()
        z = _gather(z, ParallelMode.PARALLEL_1D, dim=-2)
        comm_times.append(time() - start) # gather time

        if DEBUG: print_rank_0(f"forward step(6), z_.shape: {z.shape}")
    
        if DEBUG: print_rank_0('\n'.join([f'step{i} {comp_times[i]}' for i in range(7)]))
        


        # return (input_ + z)
        return z, comp_times, comm_times