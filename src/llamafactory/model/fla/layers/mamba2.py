"""
Linear attention classes
"""
import sys
from typing import List, Tuple, Optional
import copy
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from einops import rearrange, repeat
import torch.nn.functional as F
import math


from transformers.cache_utils import Cache  # starting at Transformers v4.36

# Causal linear attention dot product CUDA kernel from fast-transformers

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.modules.activations import ACT2FN

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). 
    The hidden states go from: 
       (batch, num_key_value_heads, seqlen, head_dim) to 
       (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# ---------------------
# Simple Mamba2 layer class
# ---------------------

class Mamba2(nn.Module):
    def __init__(self,
                 hidden_size: int = 1024,
                 num_heads: int = 32,
                 head_dim: int = 128,
                 num_key_value_heads: Optional[int] = None,
                 num_key_value_groups: Optional[int] = None,
                 layer_idx: Optional[int] = None,
                 conv_size: int = 2,
                 norm_eps: float = 1e-5,
                 elementwise_affine: Optional[bool] = True,
                 use_short_conv: Optional[bool] = False,
                 use_gnorm: Optional[bool] = True,
                 use_A: Optional[bool] = True,
                 use_D: Optional[bool] = False,
                 mimic_init: Optional[bool] = True,
                 **kwargs: any) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups

        self.layer_idx = layer_idx

        self.use_short_conv = use_short_conv
        self.use_D = use_D
        self.use_gnorm = use_gnorm
        self.use_A = use_A

        self.mimic_init = mimic_init
        self.bias = False
        self.chunk_size = 128
        conv_bias = True
        self.conv_bias = conv_bias
        self.conv_size = conv_size
        self.activation="silu"

        self.attention_dropout = 0  # We don't use dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.device = self.q_proj.weight.device
        self.dtype = self.q_proj.weight.dtype

        self.q_shape = [self.num_heads, self.head_dim]
        self.k_shape = [self.num_key_value_heads, self.head_dim]
        self.v_shape = [self.num_key_value_heads, self.head_dim]
        
        if self.use_short_conv:
            conv_dim = self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim
            
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=self.conv_bias,
                kernel_size=self.conv_size,
                groups=conv_dim,
                padding=self.conv_size - 1,
                device=self.device, 
                dtype=self.dtype
            )
            if self.mimic_init:
                with torch.no_grad():
                    self.conv1d.weight.zero_()  
                    self.conv1d.weight[:, 0, 1] = 1 
                    self.conv1d.bias.zero_()  

        # Activation after conv
        if self.activation == "identity":
            self.act = nn.Identity()
        elif self.activation in ["silu", "swish"]:
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation {self.activation}")

        self.in_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.num_heads,
            bias=self.bias,
        ).to(self.dtype).to(self.device)
        if self.mimic_init: 
            nn.init.zeros_(self.in_proj.weight)

        if self.use_gnorm:
            self.g_norm = RMSNorm(hidden_size=self.head_dim, elementwise_affine=elementwise_affine, eps=norm_eps).to(self.dtype).to(self.device)
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, device=self.device, dtype=self.dtype)
            # self.gate_fn = ACT2FN["swish"]
            self.gate_fn = ACT2FN["swish"]
            nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        #nn.init.zeros_(self.g_proj.weight)
        #nn.init.constant_(self.g_proj.bias, 1.28)

        dt = torch.exp(
            torch.rand(self.num_heads, dtype=self.dtype, device=self.device) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        dt = torch.clamp(dt, min=0.001)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        if self.mimic_init:
            self.dt_bias = nn.Parameter(inv_dt)
        else:
            self.dt_bias = nn.Parameter(torch.zeros_like(inv_dt))
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if self.use_A:
            if self.mimic_init:
                A_log = torch.ones(self.num_heads, dtype=self.dtype, device=self.device)
                self.A_log_bias = nn.Parameter(A_log)
                self.A_log_bias._no_weight_decay = True
            else:
                A_init_range = (1, 16)
                A = torch.empty(self.num_heads, dtype=torch.float32, device=self.device).uniform_(*A_init_range)
                A_log = torch.log(A).to(dtype=self.dtype)
                self.A_log_bias = nn.Parameter(A_log)
                self.A_log_bias._no_weight_decay = True


        if self.use_D:
            self.D = nn.Parameter(torch.ones(self.num_heads, device=self.device, dtype=self.dtype))
            self.D._optim = {"weight_decay": 0.0}
        



    def forward(self,
                hidden_states: torch.Tensor,
                vision_patch_indices: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None,  # "legacy" cache approach
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass modified from transformers.models.mistral.modeling_mistral (v4.36)
        - Consistent with HuggingFace Transformers for easy use with their pretrained models
        """
        b, l, _ = hidden_states.size()
        
        hidden_states = hidden_states.to(self.dtype)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(b, l, *self.q_shape).transpose(1, 2)
        k = k.view(b, l, *self.k_shape).transpose(1, 2)
        v = v.view(b, l, *self.v_shape).transpose(1, 2)



            
        u = hidden_states
        batch, seqlen, dim = u.shape
        v = rearrange(v, "b h l n -> b l h n", h=self.num_key_value_heads)
        q = rearrange(q, "b h l n -> b l h n", h=self.num_heads)
        k = rearrange(k, "b h l n -> b l h n", h=self.num_key_value_heads)

        if self.use_short_conv:
            v_flattened = rearrange(v, "b l h n -> b l (h n)")
            k_flattened = rearrange(k, "b l h n -> b l (h n)")
            q_flattened = rearrange(q, "b l h n -> b l (h n)")

            # 2. 在维度 2 上拼接三个张量，得到 (b, l, 3*(h*d))
            vkq = torch.cat([v_flattened, k_flattened, q_flattened], dim=2)

            vkq = self.convolutional_forward(vkq)

            v, k, q = torch.split(
                vkq,
                [
                    self.num_key_value_heads * self.head_dim,
                    self.num_key_value_heads * self.head_dim,
                    self.num_heads * self.head_dim,
                ],
                dim=-1,
            )

            v = rearrange(v, "b l (h n) -> b h l n", h=self.num_key_value_heads)
            k = rearrange(k, "b l (h n) -> b h l n", h=self.num_key_value_heads)
            q = rearrange(q, "b l (h n) -> b l h n", h=self.num_heads)
            k = repeat_kv(k, self.num_key_value_groups).transpose(1, 2)
            v = repeat_kv(v, self.num_key_value_groups).transpose(1, 2)
        else:
            q, k ,v = self.act(q), self.act(k), self.act(v)
            
        dt = self.in_proj(u)
        if self.use_A:
            A = -torch.exp(self.A_log_bias.float())
        else:
            A = -torch.ones(self.num_heads, device=dt.device)
        y = mamba_chunk_scan_combined(
            x = v,
            #x = v / F.softplus(A_log).to(v.dtype).unsqueeze(-1),
            dt=dt,
            dt_softplus=True,
            A=A,
            B=k,
            C=q,
            chunk_size=self.chunk_size,
            dt_bias=self.dt_bias,
            # initial_states=(state["ssm"] if state is not None else None), # currently not supported by mamba_ssm.utils.generation
            return_final_states=False,
        )

        if self.use_D:
            Du = torch.einsum("h,blhp->blhp", self.D, v)
            y = y + Du
        if self.use_gnorm:
            o = self.g_norm(y)
            g = self.g_proj(hidden_states)
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
            o = o * self.gate_fn(g) 
        else:
            o = y
        o = rearrange(o, 'b l h d -> b h l d')

        # Concatenate heads and apply output projection
        o = o.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
        o = self.o_proj(o)

        past_key_values = None

        return o, None, past_key_values
    
    def convolutional_forward(self, xBC):
        xBC = causal_conv1d_fn(
            xBC.transpose(1, 2),
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            activation=None if self.activation == "identity" else self.activation,
        ).transpose(1, 2)
        return xBC
