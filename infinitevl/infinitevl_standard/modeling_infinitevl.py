# coding=utf-8
# Copyright 2025 The HustVL Team.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen2.5-VL, which is derived from EleutherAI's GPT-NeoX library
# and the GPT-NeoX and OPT implementations. It has been modified to create InfiniteVL,
# adapting the architecture to accommodate [mention your specific changes briefly, e.g., long-context handling, etc.].
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as InfiniteVLRMSNorm

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from .configuration_infinitevl import InfiniteVLConfig, InfiniteVLTextConfig, InfiniteVLVisionConfig

logger = logging.get_logger(__name__)


def _get_decoder_cfg(config):
    if hasattr(config, "get_text_config"):
        return config.get_text_config(decoder=True)
    return config

class StaticSlidingWindowLayerPrealloc(CacheLayerMixin):
    is_sliding = True

    def __init__(
        self,
        *,
        config,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,  # True: init with zeros; False: empty (faster)
    ):
        super().__init__()
        cfg = _get_decoder_cfg(config)

        # Dimensions
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads")))
        head_dim = int(getattr(cfg, "head_dim"))
        W = (
            getattr(cfg, "sliding_window", None)
            or getattr(cfg, "attention_chunk_size", None)
            or int(getattr(cfg, "max_position_embeddings"))
        )
        if W is None or int(W) <= 0:
            raise ValueError("SWA requires valid sliding_window / attention_chunk_size / max_position_embeddings")
        W = int(W)
        self.sliding_window = W
        self.capacity = max(W - 1, 0)

        # State
        self.is_initialized = True
        self.dtype = dtype
        self.device = device
        self.batch_size = int(batch_size)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.size = 0
        self.cumulative_length = 0

        # Pre-allocation
        if self.capacity > 0:
            shape = (self.batch_size, self.num_kv_heads, self.capacity, self.head_dim)
            alloc = torch.zeros if zero_init else torch.empty
            self._buf_keys = alloc(shape, dtype=self.dtype, device=self.device)
            self._buf_values = alloc(shape, dtype=self.dtype, device=self.device)
            self.keys = self._buf_keys[:, :, :0, :]
            self.values = self._buf_values[:, :, :0, :]
        else:
            empty = torch.empty(
                (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self._buf_keys = self._buf_values = None
            self.keys = self.values = empty

    # —— Read-only view (<= capacity)
    def _prev_cache(self):
        return self.keys, self.values

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        conv_state: Optional[tuple] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shape/Batch consistency check
        assert key_states.shape == value_states.shape, "K/V shapes must match"
        B, H, Tq, D = key_states.shape
        if B != self.batch_size:
            raise ValueError(f"SWA pre-allocated batch_size={self.batch_size}, but got B={B}")
        if H != self.num_kv_heads or D != self.head_dim:
            raise ValueError(f"SWA head dim mismatch: got H={H},D={D}, expect H={self.num_kv_heads},D={self.head_dim}")

        prev_k, prev_v = self._prev_cache()
        full_k = torch.cat([prev_k, key_states], dim=-2)
        full_v = torch.cat([prev_v, value_states], dim=-2)

        # Generate new tail (length new_size)
        new_size = min(self.capacity, self.size + Tq)
        need_from_prev = max(0, new_size - Tq)
        if need_from_prev > 0:
            pk_tail = prev_k[:, :, self.size - need_from_prev :, :]
            pv_tail = prev_v[:, :, self.size - need_from_prev :, :]
        else:
            pk_tail = key_states[:, :, :0, :]
            pv_tail = value_states[:, :, :0, :]

        take_from_new = new_size - need_from_prev
        if take_from_new > 0:
            nk_tail = key_states[:, :, Tq - take_from_new :, :]
            nv_tail = value_states[:, :, Tq - take_from_new :, :]
            k_tail = torch.cat([pk_tail, nk_tail], dim=-2)
            v_tail = torch.cat([pv_tail, nv_tail], dim=-2)
        else:
            k_tail, v_tail = pk_tail, pv_tail

        # Write back to fixed buffer
        if self.capacity > 0 and new_size > 0:
            self._buf_keys[:, :, :new_size, :].copy_(k_tail)
            self._buf_values[:, :, :new_size, :].copy_(v_tail)
            self.keys = self._buf_keys[:, :, :new_size, :]
            self.values = self._buf_values[:, :, :new_size, :]
        self.size = int(new_size)
        self.cumulative_length += Tq
        return full_k, full_v

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        q_len = int(cache_position.shape[0])
        # cumulative_length includes q_len after update(); we need the length of 'past' before update
        pre_cum = max(int(self.cumulative_length) - q_len, 0)
        kv_offset = max(pre_cum - self.sliding_window + 1, 0)
        if pre_cum >= self.sliding_window:
            kv_len = (self.sliding_window - 1) + q_len  # Window full: tail (W-1) + current
        else:
            kv_len = pre_cum + q_len  # Window not full: existing past + current
        return kv_len, kv_offset

    def get_seq_length(self) -> int:
        return int(self.cumulative_length)

    def get_max_cache_shape(self) -> int:
        return int(self.sliding_window)

    def crop(self, max_length: int) -> None:
        if self.get_seq_length() >= self.sliding_window:
            raise ValueError("Cropping is forbidden after filling SWA window (to avoid state loss)")
        if max_length < 0:
            new_size = max(0, self.size - abs(max_length))
        else:
            new_size = min(self.size, max_length)
        if self.capacity > 0:
            if new_size == 0:
                self.keys = self._buf_keys[:, :, :0, :]
                self.values = self._buf_values[:, :, :0, :]
            else:
                self._buf_keys[:, :, :new_size, :].copy_(
                    self._buf_keys[:, :, self.size - new_size : self.size, :]
                )
                self._buf_values[:, :, :new_size, :].copy_(
                    self._buf_values[:, :, self.size - new_size : self.size, :]
                )
                self.keys = self._buf_keys[:, :, :new_size, :]
                self.values = self._buf_values[:, :, :new_size, :]
        self.size = int(new_size)
        self.cumulative_length = int(self.size)

    # Batch operations (Strictly static: changing batch_size is not allowed)
    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats != 1:
            raise RuntimeError("Static cache forbids changing batch size (repeat_interleave)")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if indices.numel() != self.batch_size:
            raise RuntimeError("Static cache forbids changing batch size (select_indices)")

    def lazy_initialization(self, *args, **kwargs):
        # Pre-allocated layer is fully initialized in __init__, do nothing here.
        # Interface preserved for HF abstract base class requirements.
        return

class StaticLinearLayerPrealloc(CacheLayerMixin):
    is_sliding = False

    def __init__(
        self,
        *,
        config,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
        recurrent_state_shape: Optional[Tuple[int, ...]] = None,  # To override default shape
    ):
        super().__init__()
        cfg = _get_decoder_cfg(config)

        # Dimensions
        self.num_linear_heads = int(getattr(cfg, "num_linear_heads", getattr(cfg, "num_attention_heads")))
        self.num_linear_kv_heads = int(getattr(cfg, "num_linear_key_value_heads", self.num_linear_heads))
        self.linear_head_dim = int(getattr(cfg, "linear_head_dim", getattr(cfg, "head_dim")))
        self.conv_size = int(getattr(cfg, "conv_size", 1))
        self.use_short_conv = bool(getattr(cfg, "use_short_conv", True))
        expand_v = float(getattr(cfg, "expand_v", 1.0))
        self.v_head_dim = int(round(self.linear_head_dim * expand_v))

        # State
        self.is_initialized = True
        self.dtype = dtype
        self.device = device
        self.batch_size = int(batch_size)
        self.seq_len = 0
        self.start = False

        alloc = torch.zeros if zero_init else torch.empty
        B = self.batch_size
        Hq = self.num_linear_heads
        Hk = self.num_linear_kv_heads
        C = self.linear_head_dim
        Cv = self.v_head_dim
        K = self.conv_size

        # Pre-allocate conv state
        if self.use_short_conv:
            self.conv_state_q = alloc((B, Hq * C, K), dtype=self.dtype, device=self.device)
            self.conv_state_k = alloc((B, Hk * C, K), dtype=self.dtype, device=self.device)
            self.conv_state_v = alloc((B, Hk * Cv, K), dtype=self.dtype, device=self.device)
        else:
            self.conv_state_q = self.conv_state_k = self.conv_state_v = None

        # Pre-allocate recurrent state (Default shape, can be overridden by recurrent_state_shape)
        if recurrent_state_shape is None:
            recurrent_state_shape = (B, Hq, C, Cv)
        else:
            # If user provides full shape: check batch dimension matches B
            assert recurrent_state_shape[0] == B, "recurrent_state_shape batch dim must match pre-allocated batch_size"
        self.recurrent_state = alloc(recurrent_state_shape, dtype=self.dtype, device=self.device)

    def update(
        self,
        key_states: Optional[torch.Tensor] = None,  # Compatible, not used
        value_states: Optional[torch.Tensor] = None,  # Compatible, not used
        conv_state: Optional[tuple] = None,  # (cq, ck, cv) or None
        recurrent_state: Optional[torch.Tensor] = None,  # If passed, must match pre-allocated shape
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple:
        if cache_kwargs is None:
            cache_kwargs = {}
        op = cache_kwargs.get("op", "get" if (conv_state is None and recurrent_state is None) else "set")

        if self.start is False:
            self.start = True
            return (None, None, None), None

        if op == "get":
            return (self.conv_state_q, self.conv_state_k, self.conv_state_v), self.recurrent_state

        # set: In-place copy only, shape/batch change forbidden
        if conv_state is not None and self.use_short_conv:
            assert isinstance(conv_state, (tuple, list)), "conv_state must be (cq, ck, cv)"
            cq, ck, cv = (conv_state + (None, None, None))[:3]
            if cq is not None:
                if tuple(cq.shape) != tuple(self.conv_state_q.shape):
                    raise RuntimeError(
                        f"conv_q shape changed: got {tuple(cq.shape)} vs prealloc {tuple(self.conv_state_q.shape)}"
                    )
                self.conv_state_q.copy_(cq)
            if ck is not None:
                if tuple(ck.shape) != tuple(self.conv_state_k.shape):
                    raise RuntimeError(
                        f"conv_k shape changed: got {tuple(ck.shape)} vs prealloc {tuple(self.conv_state_k.shape)}"
                    )
                self.conv_state_k.copy_(ck)
            if cv is not None:
                if tuple(cv.shape) != tuple(self.conv_state_v.shape):
                    raise RuntimeError(
                        f"conv_v shape changed: got {tuple(cv.shape)} vs prealloc {tuple(self.conv_state_v.shape)}"
                    )
                self.conv_state_v.copy_(cv)
        elif conv_state is not None and not self.use_short_conv:
            raise RuntimeError("config.use_short_conv=False, but conv_state was passed")

        if recurrent_state is not None:
            if tuple(recurrent_state.shape) != tuple(self.recurrent_state.shape):
                raise RuntimeError(
                    f"recurrent_state shape changed: got {tuple(recurrent_state.shape)} vs prealloc {tuple(self.recurrent_state.shape)}"
                )
            self.recurrent_state.copy_(recurrent_state)

        self.seq_len += int(cache_kwargs.get("delta_len", 0))
        return (self.conv_state_q, self.conv_state_k, self.conv_state_v), self.recurrent_state

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        qlen = cache_position.shape[0] if cache_position is not None else 0
        return self.get_seq_length() + qlen, 0

    def get_seq_length(self) -> int:
        return int(self.seq_len)

    def get_max_cache_shape(self) -> int:
        return -1

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = max(0, self.get_seq_length() - abs(max_length))
        self.seq_len = min(self.get_seq_length(), max_length)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats != 1:
            raise RuntimeError("Static cache forbids changing batch size (repeat_interleave)")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if indices.numel() != self.batch_size:
            raise RuntimeError("Static cache forbids changing batch size (select_indices)")

    def lazy_initialization(self, *args, **kwargs):
        return

class StaticCachePrealloc(Cache):
    """
    Pre-allocates memory for all layers in __init__; update() at runtime performs no new allocations.
    """

    def __init__(
        self,
        *,
        config,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
        recurrent_state_shape: Optional[Tuple[int, ...]] = None,  # Can unify override for linear recurrent state
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        layers = []
        cfg = _get_decoder_cfg(config)

        layer_types = getattr(cfg, "layer_types", None)
        if layer_types is None:
            # Default: all linear_attention
            layer_types = ["linear_attention"] * int(getattr(cfg, "num_hidden_layers"))

        # Shared KV layer pruning (if any)
        if hasattr(cfg, "num_kv_shared_layers"):
            layer_types = layer_types[: -int(getattr(cfg, "num_kv_shared_layers"))]

        for lt in layer_types:
            if lt in ("sliding_attention", "chunked_attention"):
                layers.append(
                    StaticSlidingWindowLayerPrealloc(
                        config=cfg,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                        zero_init=zero_init,
                    )
                )
            elif lt in ("linear_attention", "delta_net", "retnet", "state_space"):
                layers.append(
                    StaticLinearLayerPrealloc(
                        config=cfg,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                        zero_init=zero_init,
                        recurrent_state_shape=recurrent_state_shape,
                    )
                )
            else:
                # Full attention layers (can also write a pre-alloc version if needed);
                # currently keeping the original DynamicLayer concept or similar placeholder
                # (Note: Original code had DynamicLayer which was not provided in context, assuming user handles this)
                pass

        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor = None,
        value_states: torch.Tensor = None,
        conv_state: Optional[Tuple[torch.Tensor]] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        # No allocation, just forward
        return self.layers[layer_idx].update(key_states, value_states, conv_state, recurrent_state, cache_kwargs)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        legacy_cache = ()
        for layer in self.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            legacy_cache += ((k, v),)
        return legacy_cache


# ================= Vision: InfiniteVL Front-end =================
class InfiniteVLVisionMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class InfiniteVLVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class InfiniteVLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class InfiniteVLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = InfiniteVLRMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class InfiniteVLVisionAttention(nn.Module):
    def __init__(self, config: InfiniteVLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class InfiniteVLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = InfiniteVLRMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = InfiniteVLRMSNorm(config.hidden_size, eps=1e-6)
        self.attn = InfiniteVLVisionAttention(config=config)
        self.mlp = InfiniteVLVisionMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@auto_docstring
class InfiniteVLPreTrainedModel(PreTrainedModel):
    config: InfiniteVLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InfiniteVLDecoderLayer", "InfiniteVLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True


class InfiniteVLVisionTransformerPretrainedModel(InfiniteVLPreTrainedModel):
    config: InfiniteVLVisionConfig
    _no_split_modules = ["InfiniteVLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = InfiniteVLVisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = InfiniteVLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([InfiniteVLVisionBlock(config) for _ in range(config.depth)])
        self.merger = InfiniteVLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for InfiniteVL outputs, with hidden states and attentions.
    """
)
class InfiniteVLModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class InfiniteVLRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: InfiniteVLTextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # InfiniteVL uses 3D grid positions (temporal / height / width)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class InfiniteVLTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos and
            sin so that they can be properly broadcasted to the dimensions of q and k.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InfiniteVLSelfAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention.
    """

    def __init__(self, config: InfiniteVLTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Enable window only if the layer is sliding window/chunk
        self.sliding_window = (
            config.sliding_window if config.layer_types[self.layer_idx] == "sliding_attention" else None
        )
        self.config._attn_implementation = "flash_attention_2"
        self.rotary_emb = InfiniteVLRotaryEmbedding(config=config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # 1) Linear projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [B, T, H*D] -> [B, H, T, D]
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # 2) RoPE (only for the new tokens in this step)
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.rope_scaling["mrope_section"],
        )

        # 3) Adapt to Static Cache: write and retrieve visible KV; crop mask to same visible range
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            # First, uniformly write current step K/V into cache (for both Full Attention / Sliding Window)
            key_states, value_states = past_key_values.update(
                layer_idx=self.layer_idx,
                key_states=key_states,
                value_states=value_states,
                conv_state=None,
                recurrent_state=None,
                cache_kwargs=cache_kwargs,
            )

            # Only sliding window layers need mask cropping
            if self.sliding_window is not None:
                kv_len, kv_offset = past_key_values.layers[self.layer_idx].get_mask_sizes(cache_position)
                if kv_offset != 0:
                    attention_mask = None
                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        attention_mask = attention_mask[:, :, :, kv_offset : kv_offset + kv_len]
                    elif attention_mask.dim() == 2:
                        attention_mask = attention_mask[:, kv_offset : kv_offset + kv_len]

        # 4) Choose attention backend
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 5) Forward pass
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )

        # 6) Output projection
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GatedDeltaNet(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).

    This is used as the linear/delta branch in InfiniteVL.
    """

    def __init__(self, config: InfiniteVLTextConfig, layer_idx: int):
        super().__init__()

        self.mode = config.mode

        self.hidden_size = config.hidden_size
        self.expand_v = config.expand_v
        self.norm_eps = config.norm_eps

        self.use_gate = config.use_gate
        self.use_short_conv = config.use_short_conv
        self.conv_size = config.conv_size
        self.conv_bias = config.conv_bias

        self.num_heads = config.num_linear_heads
        self.num_key_value_heads = config.num_linear_key_value_heads

        self.head_dim = getattr(config, "linear_head_dim", config.hidden_size // config.num_attention_heads)

        self.key_dim = int(self.num_key_value_heads * self.head_dim)
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.key_dim * self.expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.key_dim * self.expand_v}, which is invalid for nn.Linear."
            )
        if not math.isclose(self.head_dim * self.expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value when multiplied by head_dim={self.head_dim}. "
                f"Resulting head_v_dim would be {self.head_dim * self.expand_v}, which is invalid for FusedRMSNormGated."
            )
        assert self.mode in ["chunk", "fused_recurrent"], f"Not suppoerted mode `{self.mode}`."

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if self.use_short_conv:
            self.conv_size = config.conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.num_heads * self.head_dim,
                kernel_size=self.conv_size,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_size,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=self.conv_size,
                activation="silu",
            )
        else:
            raise UserWarning(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )

        if self.use_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_v_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=self.norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=self.norm_eps)
        self.o_proj = nn.Linear(self.num_heads * self.head_v_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[Dict],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_mask = None
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len]."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if q_len <= 64 else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # === Read Cache: Linear layer conv/recurrent state ===
        prev_conv_bundle = (None, None, None)
        recurrent_state = None
        use_cache = False

        if past_key_values is not None:
            use_cache = True
            # First time: get, do not modify cache
            prev_conv_bundle, recurrent_state = past_key_values.update(
                layer_idx=self.layer_idx,
                key_states=None,
                value_states=None,
                conv_state=None,
                recurrent_state=None,
                cache_kwargs={"op": "get", "cache_position": cache_position},
            )

        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."),
                indices,
            ).unsqueeze(0)

        # === Short Convolution (if enabled) ===
        if self.use_short_conv:
            prev_q, prev_k, prev_v = prev_conv_bundle
            q, new_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=prev_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, new_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=prev_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, new_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=prev_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            next_conv_bundle = (new_state_q, new_state_k, new_state_v)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))
            next_conv_bundle = None  # No cache needed if short conv is not used

        # === Shape adjustments ===
        q = rearrange(q, "b t (h d) -> b t h d", d=self.head_dim)
        k = rearrange(k, "b t (h d) -> b t h d", d=self.head_k_dim)
        v = rearrange(v, "b t (h d) -> b t h d", d=self.head_v_dim)

        beta = self.b_proj(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        # === Recurrent Kernel ===
        if mode == "chunk":
            o, next_recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            o, next_recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # === Write Cache: Store new conv/recurrent state ===
        if past_key_values is not None:
            past_key_values.update(
                layer_idx=self.layer_idx,
                key_states=None,
                value_states=None,
                conv_state=next_conv_bundle,
                recurrent_state=next_recurrent_state,
                cache_kwargs={"op": "set", "delta_len": q_len, "cache_position": cache_position},
            )

        # === Output Projection ===
        if self.use_gate:
            g_gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g_gate)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None


class InfiniteVLDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: InfiniteVLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.self_attn = GatedDeltaNet(config, layer_idx)
        elif self.layer_type in ("full_attention", "sliding_attention"):
            self.self_attn = InfiniteVLSelfAttention(config, layer_idx)

        self.mlp = InfiniteVLTextMLP(config)
        self.input_layernorm = InfiniteVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = InfiniteVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention / Gated Delta
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class InfiniteVLTextModel(InfiniteVLPreTrainedModel):
    config: InfiniteVLTextConfig

    def __init__(self, config: InfiniteVLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [InfiniteVLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = InfiniteVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = InfiniteVLRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if (
            use_cache
            and (past_key_values is None or not isinstance(past_key_values, StaticCachePrealloc))
            and not torch.jit.is_tracing()
        ):
            # Allocate static cache on the first forward pass
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            past_key_values = StaticCachePrealloc(
                config=self.config,
                batch_size=inputs_embeds.shape[0],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: packed FA2 case uses 4D position_ids (text + 3D vision)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping["full_attention"],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@auto_docstring
class InfiniteVLModel(InfiniteVLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: InfiniteVLConfig
    _no_split_modules = ["InfiniteVLDecoderLayer", "InfiniteVLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = InfiniteVLVisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = InfiniteVLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    # normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t,
                        dtype=range_tensor.dtype,
                        device=range_tensor.device,
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                if attention_mask is not None:
                    position_ids[..., i, attention_mask[i]] = llm_positions.to(position_ids.device)
                else:
                    position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, InfiniteVLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids + delta.to(position_ids.device)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = InfiniteVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for InfiniteVL causal language model (or autoregressive) outputs.
    """
)
class InfiniteVLCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class InfiniteVLQwen2_5_VLForConditionalGeneration(InfiniteVLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = InfiniteVLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, InfiniteVLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        return InfiniteVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # InfiniteVL position_ids are prepared with rope_deltas
        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            if cache_position[0] == 0 or self.model.rope_deltas is None:
                vision_positions, rope_deltas = self.model.get_rope_index(
                    model_inputs.get("input_ids", None),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            elif "position_ids" in model_inputs:
                batch_size, seq_length = model_inputs["position_ids"].shape
                device = model_inputs["position_ids"].device
                position_ids = torch.arange(seq_length, device=device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                delta = cache_position[0] + self.model.rope_deltas
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                vision_positions = position_ids + delta.expand_as(position_ids)

            # Concatenate "text + vision" positions into [4, bs, seq-len]
            text_positions = model_inputs["position_ids"][None, ...]
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        """

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids,
                inputs_embeds=model_kwargs.get("inputs_embeds", None),
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key],
                        lengths=lengths,
                        repeat_times=expand_size,
                    )
                elif key == "image_grid_thw":
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key],
                        lengths=lengths,
                        repeat_times=expand_size,
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key],
                        lengths=lengths,
                        repeat_times=expand_size,
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key],
                        lengths=lengths,
                        repeat_times=expand_size,
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key],
                        lengths=list(video_nums),
                        repeat_times=expand_size,
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def allocate_inference_cache(self, batch_size):
        return StaticCachePrealloc(
            config=self.config.text_config,
            batch_size=batch_size,
            dtype=self.model.dtype,
            device=self.model.device,
        )


__all__ = [
    "InfiniteVLQwen2_5_VLForConditionalGeneration",
    "InfiniteVLModel",
    "InfiniteVLPreTrainedModel",
    "InfiniteVLTextModel",
]
