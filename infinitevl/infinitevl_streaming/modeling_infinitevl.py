# coding=utf-8
# Copyright 2025 The HustVL Team.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen2.5-VL, which is derived from EleutherAI's GPT-NeoX library
# and the GPT-NeoX and OPT implementations. It has been modified to create InfiniteVL,
# adapting the architecture to accommodate long-context handling, static cache preallocation, etc.
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

from configuration_infinitevl import InfiniteVLConfig, InfiniteVLTextConfig, InfiniteVLVisionConfig

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

logger = logging.get_logger(__name__)


def _get_decoder_cfg(config):
    if hasattr(config, "get_text_config"):
        return config.get_text_config(decoder=True)
    return config


# ================= Dynamic KV cache (full attention fallback) =================
class DynamicLayer(CacheLayerMixin):
    """
    Dynamic cache layer for full attention.

    This cache grows as more tokens are generated and stores key/value tensors of shape
    `[batch_size, num_heads, seq_len, head_dim]`. It is used as a fallback when no
    statically preallocated cache is provided (e.g., for full attention layers).
    """

    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        conv_state: Optional[tuple] = None,             # compatibility only, unused
        recurrent_state: Optional[torch.Tensor] = None, # compatibility only, unused
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key/value states to the dynamic cache.

        Args:
            key_states (`torch.Tensor`):
                New key states to cache.
            value_states (`torch.Tensor`):
                New value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache (unused).

        Returns:
            `tuple(torch.Tensor, torch.Tensor)`:
                Concatenated key and value states.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """
        Return `(length, offset)` for mask construction.

        For the dynamic cache, the whole sequence is always visible, so the offset is 0.
        """
        kv_offset = 0
        kv_length = self.get_seq_length()
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Return the number of cached tokens."""
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        """Dynamic cache has no fixed maximum length."""
        return -1

    def crop(self, max_length: int) -> None:
        """
        Crop the cache to `max_length` tokens (or remove `abs(max_length)` tokens if negative).
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the batch dimension `repeats` times."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Index the batch dimension with `indices`."""
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]


# ================= Static SWA (sliding / chunked) =================
class StaticSlidingWindowLayerPrealloc(CacheLayerMixin):
    """
    Static preallocated cache for sliding-window / chunked attention.

    All memory is allocated in `__init__`; `update` does not allocate new tensors and
    only updates views into the preallocated buffers.
    """

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

    # Read-only view (≤ capacity)
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
        # Shape / batch checks
        assert key_states.shape == value_states.shape, "Key/value shapes must match"
        B, H, Tq, D = key_states.shape
        if B != self.batch_size:
            raise ValueError(f"SWA pre-allocated batch_size={self.batch_size}, but got B={B}")
        if H != self.num_kv_heads or D != self.head_dim:
            raise ValueError(
                f"SWA head dim mismatch: got H={H},D={D}, expected H={self.num_kv_heads},D={self.head_dim}"
            )

        prev_k, prev_v = self._prev_cache()
        full_k = torch.cat([prev_k, key_states], dim=-2)
        full_v = torch.cat([prev_v, value_states], dim=-2)

        # Compute new tail (length new_size)
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

        # Write back into fixed buffers
        if self.capacity > 0 and new_size > 0:
            self._buf_keys[:, :, :new_size, :].copy_(k_tail)
            self._buf_values[:, :, :new_size, :].copy_(v_tail)
            self.keys = self._buf_keys[:, :, :new_size, :]
            self.values = self._buf_values[:, :, :new_size, :]
        self.size = int(new_size)
        self.cumulative_length += Tq
        return full_k, full_v

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """
        Compute visible KV length and offset for the sliding window.

        `cumulative_length` includes the current query length after `update()`, so we
        subtract `q_len` to get the length of the previous context.
        """
        q_len = int(cache_position.shape[0]) if cache_position is not None else 0
        pre_cum = max(int(self.cumulative_length) - q_len, 0)
        kv_offset = max(pre_cum - self.sliding_window + 1, 0)
        if pre_cum >= self.sliding_window:
            kv_len = (self.sliding_window - 1) + q_len  # Full window: tail (W-1) + current
        else:
            kv_len = pre_cum + q_len                     # Not full: all past + current
        return kv_len, kv_offset

    def get_seq_length(self) -> int:
        return int(self.cumulative_length)

    def get_max_cache_shape(self) -> int:
        return int(self.sliding_window)

    def crop(self, max_length: int) -> None:
        """
        Crop internal buffers to `max_length`. Disallowed after the window has been filled
        (to avoid losing required sliding-window state).
        """
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

    # Batch operations (batch size is strictly static)
    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats != 1:
            raise RuntimeError("Static cache forbids changing batch size (repeat_interleave)")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if indices.numel() != self.batch_size:
            raise RuntimeError("Static cache forbids changing batch size (select_indices)")

    def lazy_initialization(self, *args, **kwargs):
        # Pre-allocated in __init__, nothing to do here.
        # Interface is kept for HF CacheLayerMixin compatibility.
        return


# ================= Static linear (Delta / state-space branch) =================
class StaticLinearLayerPrealloc(CacheLayerMixin):
    """
    Static preallocated cache for linear / Delta / state-space layers.

    Holds short-convolution state and recurrent state, updated in-place without
    additional allocations at generation time.
    """

    is_sliding = False

    def __init__(
        self,
        *,
        config,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
        recurrent_state_shape: Optional[Tuple[int, ...]] = None,  # Optional override for recurrent state shape
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

        # Pre-allocate recurrent state (default shape can be overridden)
        if recurrent_state_shape is None:
            recurrent_state_shape = (B, Hq, C, Cv)
        else:
            # Custom shape: only the batch dimension must match
            assert recurrent_state_shape[0] == B, "recurrent_state_shape batch dim must match pre-allocated batch_size"
        self.recurrent_state = alloc(recurrent_state_shape, dtype=self.dtype, device=self.device)

    def update(
        self,
        key_states: Optional[torch.Tensor] = None,      # compatibility only, not used
        value_states: Optional[torch.Tensor] = None,    # compatibility only, not used
        conv_state: Optional[tuple] = None,             # (cq, ck, cv) or None
        recurrent_state: Optional[torch.Tensor] = None, # if passed, must match preallocated shape
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

        # op == "set": overwrite in-place, no shape/batch changes allowed
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
                    f"recurrent_state shape changed: got {tuple(recurrent_state.shape)} vs "
                    f"prealloc {tuple(self.recurrent_state.shape)}"
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
        # Already fully initialized in __init__.
        return


# ================= Aggregate static cache =================
class StaticCachePrealloc(Cache):
    """
    Pre-allocates memory for all layers in `__init__`.

    At runtime, `update()` does not allocate new tensors and simply forwards to
    the per-layer cache objects.
    """

    def __init__(
        self,
        *,
        config,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
        recurrent_state_shape: Optional[Tuple[int, ...]] = None,  # unify override for linear recurrent state
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        cfg = _get_decoder_cfg(config)
        layers = []

        layer_types = getattr(cfg, "layer_types", None)
        if layer_types is None:
            # Default: all linear_attention
            layer_types = ["linear_attention"] * int(getattr(cfg, "num_hidden_layers"))

        # Handle shared KV layers (if any)
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
                # Full attention layers: keep a dynamic cache as in the original implementation
                layers.append(DynamicLayer())

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
        # No allocation here, just forward to the correct layer
        return self.layers[layer_idx].update(key_states, value_states, conv_state, recurrent_state, cache_kwargs)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        legacy_cache = ()
        for layer in self.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            legacy_cache += ((k, v),)
        return legacy_cache


# ================= Vision: InfiniteVL front-end (graph-optimized variant) =================
class InfiniteVLVisionMLP(nn.Module):
    def __init__(self, config: InfiniteVLVisionConfig, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions."""
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
    Equivalent to `torch.repeat_interleave(x, dim=1, repeats=n_rep)` for KV heads.

    (batch, num_kv_heads, seqlen, head_dim) -> (batch, num_attention_heads, seqlen, head_dim)
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

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
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
            # Flash Attention 2: use cu_seqlens for variable-length attention
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
            # Other implementations: process each window separately
            win_lengths_list = kwargs.get("win_lengths_list", None)
            if win_lengths_list is None:
                lengths = cu_seqlens[..., 1:] - cu_seqlens[..., :-1]
                win_lengths_list = [int(x) for x in lengths.detach().cpu().tolist()]

            splits = [
                torch.split(tensor, win_lengths_list, dim=2)
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
    def __init__(self, config: InfiniteVLVisionConfig, attn_implementation: str = "sdpa") -> None:
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
    """
    Vision transformer front-end for InfiniteVL, with precomputed window layouts for
    CUDA graph capture and efficient FlashAttention integration.
    """

    config: InfiniteVLVisionConfig
    _no_split_modules = ["InfiniteVLVisionBlock"]

    def __init__(self, config: InfiniteVLVisionConfig, *inputs, **kwargs) -> None:
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

    def set_graph_bucket(self, grid_thw: torch.Tensor):
        """
        Set the `(t, h, w)` bucket used for CUDA graph capture.

        All entries in a bucket must share the same temporal/height/width sizes so
        that window layouts and cu_seqlens can be precomputed as Python constants.
        """
        g = grid_thw.detach().cpu()
        t = int(g[0, 0])
        h = int(g[0, 1])
        w = int(g[0, 2])
        assert (g[:, 0] == t).all() and (g[:, 1] == h).all() and (g[:, 2] == w).all(), "Bucket entries must share t/h/w"
        self._bucket_t = t
        self._bucket_h = h
        self._bucket_w = w
        self._bucket_llm_h = h // self.spatial_merge_size
        self._bucket_llm_w = w // self.spatial_merge_size
        self._vit_merger_ws = self.window_size // self.spatial_merge_size // self.patch_size

    def precompute_full_cu_seqlens(self):
        """
        Precompute full-attention cu_seqlens for a single entry (time-wise full attention).

        This is done outside of graph capture so that the resulting length list can be stored
        as a pure Python list and reused as a constant.
        """
        assert hasattr(self, "_bucket_t"), "Call set_graph_bucket() before precompute_full_cu_seqlens()"
        device = self.rotary_pos_emb.inv_freq.device
        t, h, w = self._bucket_t, self._bucket_h, self._bucket_w

        step = h * w
        lens = torch.full((t,), step, device=device, dtype=torch.int32)
        cu = torch.cumsum(lens, dim=0, dtype=torch.int32)
        self._cu_full_1 = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.int32), cu],
            dim=0,
        )  # [t + 1]

        # Length list (Python) for each time slice, used by non-flash attention
        self._full_lengths_list_1 = [int(x) for x in lens.detach().cpu().tolist()]

    def precompute_window_buffers(self):
        """
        Precompute window indices and cu_seqlens for a single entry.

        This is performed outside CUDA graph capture; the resulting index tensor and
        cu_seqlens can then be replicated and shifted per-entry in forward().
        """
        if not hasattr(self, "_bucket_t"):
            raise AssertionError("Call set_graph_bucket() before precompute_window_buffers()")

        device = self.rotary_pos_emb.inv_freq.device
        t = self._bucket_t
        H = self._bucket_llm_h
        W = self._bucket_llm_w
        ws = self._vit_merger_ws
        unit = self.spatial_merge_size**2

        base = torch.arange(t * H * W, device=device, dtype=torch.int32).view(t, H, W)

        num_h = (H + ws - 1) // ws
        num_w = (W + ws - 1) // ws

        chunks = []
        lens = []
        for tt in range(t):
            for wh in range(num_h):
                h0 = wh * ws
                h1 = min(h0 + ws, H)
                for ww in range(num_w):
                    w0 = ww * ws
                    w1 = min(w0 + ws, W)
                    chunk = base[tt, h0:h1, w0:w1].reshape(-1)
                    chunks.append(chunk)
                    lens.append((h1 - h0) * (w1 - w0))

        self._win_index_1 = torch.cat(chunks, dim=0)  # [K]
        lens = torch.tensor(lens, device=device, dtype=torch.int32) * unit
        self._cu_win_seqlens_1 = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.int32), lens.cumsum(0)],
            dim=0,
        )

        # Python list of per-window lengths for non-flash attention
        self._win_lengths_list_1 = [int(x) for x in lens.detach().cpu().tolist()]

    # GPU-friendly rotary embedding
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings for the vision grid entirely on GPU.

        This avoids Python-side loops and CPU round-trips during graph capture.
        """
        device = self.rotary_pos_emb.inv_freq.device
        grid_thw = grid_thw.to(device)

        if torch.cuda.is_current_stream_capturing():
            assert hasattr(self, "_bucket_t"), "Call set_graph_bucket() before graph capture"
            t = self._bucket_t
            h = self._bucket_h
            w = self._bucket_w
            llm_h = self._bucket_llm_h
            llm_w = self._bucket_llm_w
        else:
            t = int(grid_thw[0, 0].item())
            h = int(grid_thw[0, 1].item())
            w = int(grid_thw[0, 2].item())
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

        s = self.spatial_merge_size

        h_grid = torch.arange(h, device=device).unsqueeze(1).expand(h, w)
        w_grid = torch.arange(w, device=device).unsqueeze(0).expand(h, w)

        hpos = h_grid.reshape(llm_h, s, llm_w, s).permute(0, 2, 1, 3).reshape(h * w)
        wpos = w_grid.reshape(llm_h, s, llm_w, s).permute(0, 2, 1, 3).reshape(h * w)

        pos_ids = torch.stack([hpos, wpos], dim=-1).repeat(t, 1)  # [t*h*w, 2]

        max_grid = max(h, w)
        rotary_table = self.rotary_pos_emb(max_grid)              # [max_grid, dim]
        rotary = rotary_table[pos_ids].flatten(1)                 # [t*h*w, 2*dim]
        return rotary

    def get_window_index(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get window-reordered indices and cu_seqlens for the given batch of grids.

        The single-entry buffers computed in `precompute_window_buffers` are replicated
        and shifted according to the number of vision entries in the batch.
        """
        if not hasattr(self, "_win_index_1"):
            raise RuntimeError("Call set_graph_bucket() and precompute_window_buffers() first")

        device = self.rotary_pos_emb.inv_freq.device
        grid_thw = grid_thw.to(device)

        idx1 = self._win_index_1
        cu1 = self._cu_win_seqlens_1
        Bv = grid_thw.size(0)

        if Bv == 1:
            return idx1, cu1

        K = idx1.numel()
        offset = torch.arange(Bv, device=device, dtype=idx1.dtype) * K
        window_index = idx1.repeat(Bv) + offset.repeat_interleave(K)

        step = cu1[-1]
        shifts = torch.arange(Bv, device=device, dtype=cu1.dtype) * step
        cu = cu1.repeat(Bv) + shifts.repeat_interleave(cu1.numel())
        return window_index, cu

    def get_full_cu_seqlens(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Get full-attention cu_seqlens for the given batch of grids.
        """
        device = self.rotary_pos_emb.inv_freq.device
        grid_thw = grid_thw.to(device)
        cu1 = self._cu_full_1
        Bv = grid_thw.size(0)

        if Bv == 1:
            return cu1

        step = cu1[-1]
        shifts = torch.arange(Bv, device=device, dtype=cu1.dtype) * step
        return cu1.repeat(Bv) + shifts.repeat_interleave(cu1.numel())

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                Flattened visual tokens from the tokenizer front-end.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                Temporal, height and width of each visual grid in the LLM space.
        """
        grid_thw = grid_thw.to(hidden_states.device)

        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens_full = self.get_full_cu_seqlens(grid_thw)
        cu_seqlens_full = F.pad(cu_seqlens_full, (1, 0), value=0)

        Bv = grid_thw.size(0)

        win_lengths_list_win = self._win_lengths_list_1 * int(Bv)

        if hasattr(self, "_full_lengths_list_1"):
            win_lengths_list_full = self._full_lengths_list_1 * int(Bv)
        else:
            lengths = cu_seqlens_full[1:] - cu_seqlens_full[:-1]
            base_list = [int(x) for x in lengths.detach().cpu().tolist()]
            self._full_lengths_list_1 = base_list
            win_lengths_list_full = base_list * int(Bv)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens_full
                win_lengths_list_now = win_lengths_list_full
            else:
                cu_seqlens_now = cu_window_seqlens
                win_lengths_list_now = win_lengths_list_win

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                win_lengths_list=win_lengths_list_now,
                **kwargs,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


# ================= Text: outputs / rotary / blocks =================
@dataclass
@auto_docstring(
    custom_intro="""
    Base class for InfiniteVL outputs, with hidden states and attentions.
    """
)
class InfiniteVLModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        A [`~cache_utils.Cache`] instance. For more details, see the
        KV cache guide in the Transformers documentation.

        Contains pre-computed hidden-states (keys and values in the self-attention blocks) that can be used
        (see `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The RoPE index difference between sequence length and multimodal rope.
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
    @dynamic_rope_update  # used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # InfiniteVL uses 3D grid positions (temporal / height / width)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class InfiniteVLTextMLP(nn.Module):
    def __init__(self, config: InfiniteVLTextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding with multimodal sections to the query and key tensors.

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
        mrope_section (`List[int]`):
            Channel sections for temporal, height and width in RoPE calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            Dimension along which to unsqueeze cos and sin so that they can be broadcast to `q` and `k`.

    Returns:
        `tuple(torch.Tensor, torch.Tensor)`:
            Query and key tensors after applying multimodal Rotary Position Embedding.
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
    Multi-headed attention (from "Attention Is All You Need").

    This variant integrates sliding-window attention and static cache preallocation.
    """

    def __init__(self, config: InfiniteVLTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
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

        # Enable window only if this layer is a sliding-attention layer
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

        # 1) Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [B, T, H*D] -> [B, H, T, D]
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # 2) RoPE (applied to new tokens)
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.rope_scaling["mrope_section"],
        )

        # 3) Static cache integration: write and retrieve visible KV; crop mask accordingly
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            # Write current step K/V into cache
            key_states, value_states = past_key_values.update(
                layer_idx=self.layer_idx,
                key_states=key_states,
                value_states=value_states,
                conv_state=None,
                recurrent_state=None,
                cache_kwargs=cache_kwargs,
            )

            # Sliding-window layers need attention mask cropping
            if self.sliding_window is not None:
                kv_len, kv_offset = past_key_values.layers[self.layer_idx].get_mask_sizes(cache_position)
                if kv_offset != 0:
                    attention_mask = None
                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        attention_mask = attention_mask[:, :, :, kv_offset : kv_offset + kv_len]
                    elif attention_mask.dim() == 2:
                        attention_mask = attention_mask[:, kv_offset : kv_offset + kv_len]

        # 4) Select attention backend
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 5) Forward attention
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )

        # 6) Output projection
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GatedDeltaNet(nn.Module):
    """
    Gated Delta Networks layer implementation for InfiniteVL.

    This is the linear / Delta branch used in InfiniteVL, following
    "Gated Delta Networks: Improving Mamba2 with Delta Rule" (https://arxiv.org/abs/2412.06464).
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

        # Consistency checks
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
        assert self.mode in ["chunk", "fused_recurrent"], f"Not supported mode `{self.mode}`."

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
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
                "Do not turn it off (use_short_conv=False) unless you know what you are doing."
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

        # Read cache: conv and recurrent state
        prev_conv_bundle = (None, None, None)
        recurrent_state = None
        use_cache = False

        if past_key_values is not None:
            use_cache = True
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

        # Short convolution (if enabled)
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
            next_conv_bundle = None

        # Reshape to [b, t, h, d]
        q = rearrange(q, "b t (h d) -> b t h d", d=self.head_dim)
        k = rearrange(k, "b t (h d) -> b t h d", d=self.head_k_dim)
        v = rearrange(v, "b t (h d) -> b t h d", d=self.head_v_dim)

        beta = self.b_proj(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        # Recurrent kernel
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

        # Write cache: store updated conv / recurrent state
        if past_key_values is not None:
            past_key_values.update(
                layer_idx=self.layer_idx,
                key_states=None,
                value_states=None,
                conv_state=next_conv_bundle,
                recurrent_state=next_recurrent_state,
                cache_kwargs={"op": "set", "delta_len": q_len, "cache_position": cache_position},
            )

        # Output projection
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
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`, *optional*):
                Attention mask of shape `(batch, sequence_length)` where 0 indicates padding.
            output_attentions (`bool`, *optional*):
                Whether to return the attention weights.
            use_cache (`bool`, *optional*):
                Whether to return key/value states in `past_key_values` for faster decoding.
            past_key_values (`Cache`, *optional*):
                Cached key/value projection states.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Positions of the input tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple of cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`.
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

        # MLP block
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

        # torch.jit.trace() does not support cache objects in outputs
        if (
            use_cache
            and (past_key_values is None or not isinstance(past_key_values, StaticCachePrealloc))
            and not torch.jit.is_tracing()
        ):
            # Allocate static cache on first forward
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

        # The hard-coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Packed case: `[4, bs, seq-len]` = text + 3D vision positions
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # Prepare causal masks (may have already been prepared in `generate`)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # Shared position embeddings for all decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

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


# ================= Multimodal InfiniteVL base model =================
@auto_docstring
class InfiniteVLModel(InfiniteVLPreTrainedModel):
    """
    Multimodal InfiniteVL base model: vision transformer + language model.

    This variant matches the naming and docstring style of the primary InfiniteVL implementation
    while keeping the graph-optimized / static-cache logic intact.
    """

    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: InfiniteVLConfig
    _no_split_modules = ["InfiniteVLDecoderLayer", "InfiniteVLVisionBlock"]

    def __init__(self, config: InfiniteVLConfig):
        super().__init__(config)
        self.visual = InfiniteVLVisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = InfiniteVLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

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
        GPU-friendly computation of 3D RoPE indices for text + vision.

        This avoids Python-side list ops (`tolist`, etc.) and keeps heavy work on CUDA.
        Only a few `.item()` calls are used to form scalar sizes for `torch.arange`.
        """
        device = (
            input_ids.device
            if input_ids is not None
            else (attention_mask.device if attention_mask is not None else torch.device("cuda"))
        )
        dtype_long = torch.long

        # --- config & safety ---
        spatial_merge_size = int(getattr(self.config.vision_config, "spatial_merge_size", 1) or 1)
        if spatial_merge_size < 1:
            spatial_merge_size = 1
        image_token_id = int(self.config.image_token_id)
        video_token_id = int(self.config.video_token_id)
        vision_start_token_id = int(self.config.vision_start_token_id)

        # Move optional inputs to the same device / dtype
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device=device, dtype=torch.long, non_blocking=True)
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(device=device, dtype=torch.long, non_blocking=True)
        if second_per_grid_ts is not None:
            second_per_grid_ts = torch.as_tensor(second_per_grid_ts, device=device)

        mrope_position_deltas_list: List[torch.Tensor] = []

        # ===== Text + Vision (multimodal) case =====
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids.to(device=device, dtype=dtype_long, non_blocking=True)

            attn_bool = None
            if attention_mask is not None:
                attn_bool = attention_mask.to(device=device, non_blocking=True) == 1

            B, L = total_input_ids.shape
            position_ids = torch.ones((3, B, L), dtype=dtype_long, device=device)

            img_global_idx = 0
            vid_global_idx = 0

            for b in range(B):
                seq = total_input_ids[b]
                if attn_bool is not None:
                    seq = seq[attn_bool[b]]

                # find all vision segments and their type tokens
                vs_pos = torch.nonzero(seq == vision_start_token_id, as_tuple=False).squeeze(1)  # [K]
                type_pos = vs_pos + 1
                type_pos = type_pos[type_pos < seq.size(0)]  # avoid out of range

                if type_pos.numel() > 0:
                    types = seq.index_select(0, type_pos)
                else:
                    types = type_pos

                img_pos = type_pos[types == image_token_id]
                vid_pos = type_pos[types == video_token_id]

                p_img = 0
                p_vid = 0
                n_img = img_pos.numel()
                n_vid = vid_pos.numel()

                llm_pos_ids_chunks: List[torch.Tensor] = []
                st = 0  # current text start (Python int, only used for index arithmetic)

                while (p_img < n_img) or (p_vid < n_vid):
                    # Pick the earliest of img/video
                    take_img = False
                    next_img = img_pos[p_img] if p_img < n_img else None
                    next_vid = vid_pos[p_vid] if p_vid < n_vid else None

                    if next_vid is None:
                        take_img = True
                    elif next_img is None:
                        take_img = False
                    else:
                        take_img = bool(next_img <= next_vid)

                    if take_img:
                        ed = int(img_pos[p_img].item())
                        t, h, w = image_grid_thw[img_global_idx]
                        sec_per_grid_t = 0.0
                        img_global_idx += 1
                    else:
                        ed = int(vid_pos[p_vid].item())
                        t, h, w = video_grid_thw[vid_global_idx]
                        if second_per_grid_ts is not None:
                            sec_per_grid_t = float(second_per_grid_ts[vid_global_idx].item())
                        else:
                            sec_per_grid_t = 1.0
                        vid_global_idx += 1

                    # 1) text chunk [st, ed)
                    text_len = max(0, ed - st)
                    if text_len > 0:
                        st_idx = (
                            llm_pos_ids_chunks[-1].amax() + 1
                            if len(llm_pos_ids_chunks) > 0
                            else torch.tensor(0, device=device, dtype=dtype_long)
                        )
                        text_range = torch.arange(text_len, device=device, dtype=dtype_long)
                        llm_pos_ids_chunks.append(text_range.view(1, -1).expand(3, -1) + st_idx)

                    # 2) vision grid (t, h, w) after spatial merge
                    T = int(t.item())
                    H = int((h // spatial_merge_size).item())
                    W = int((w // spatial_merge_size).item())
                    num_tokens = T * H * W

                    st_idx = (
                        llm_pos_ids_chunks[-1].amax() + 1
                        if len(llm_pos_ids_chunks) > 0
                        else torch.tensor(0, device=device, dtype=dtype_long)
                    )

                    if num_tokens > 0:
                        r = torch.arange(num_tokens, device=device, dtype=dtype_long)
                        t_index = r // (H * W)
                        h_index = (r % (H * W)) // W
                        w_index = r % W

                        if sec_per_grid_t == 0.0:
                            # image: temporal dim is 0 (or simple index)
                            t_index_scaled = t_index
                        else:
                            # video: scale temporal by seconds
                            tps = getattr(self.config.vision_config, "tokens_per_second", 25)
                            t_index_scaled = (
                                t_index.to(torch.float32) * float(sec_per_grid_t) * float(tps)
                            ).to(dtype_long)

                        llm_pos_ids_chunks.append(
                            torch.stack([t_index_scaled, h_index, w_index], dim=0)
                            + (ed - st)
                            + st_idx
                        )

                    # Move to next segment: skip the special type token + vision tokens
                    st = ed + num_tokens

                    if take_img:
                        p_img += 1
                    else:
                        p_vid += 1

                # tail pure text
                if st < seq.size(0):
                    tail_len = int(seq.size(0) - st)
                    st_idx = (
                        llm_pos_ids_chunks[-1].amax() + 1
                        if len(llm_pos_ids_chunks) > 0
                        else torch.tensor(0, device=device, dtype=dtype_long)
                    )
                    tail_range = torch.arange(tail_len, device=device, dtype=dtype_long)
                    llm_pos_ids_chunks.append(tail_range.view(1, -1).expand(3, -1) + st_idx)

                if len(llm_pos_ids_chunks) > 0:
                    llm_positions = torch.cat(llm_pos_ids_chunks, dim=1).reshape(3, -1)
                else:
                    llm_positions = torch.zeros((3, seq.size(0)), device=device, dtype=dtype_long)

                if attn_bool is not None:
                    position_ids[..., b, attn_bool[b]] = llm_positions
                else:
                    position_ids[..., b, :] = llm_positions

                # delta = max_pos + 1 - seq_len (keep original semantics)
                seq_len_b = total_input_ids[b].size(0) if attn_bool is None else int(attn_bool[b].sum().item())
                mrope_position_deltas_list.append(llm_positions.max() + 1 - seq_len_b)

            mrope_position_deltas = torch.stack(mrope_position_deltas_list, dim=0).unsqueeze(1).to(
                device=device, dtype=dtype_long
            )
            return position_ids, mrope_position_deltas

        # ===== Text-only path =====
        else:
            if attention_mask is not None:
                attn = attention_mask.to(device=device).long()
                position_ids = attn.cumsum(-1) - 1
                position_ids.masked_fill_(attn == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = (max_position_ids + 1 - attn.shape[-1]).to(dtype_long)
            else:
                B, L = input_ids.shape
                position_ids = torch.arange(L, device=device, dtype=dtype_long).view(1, 1, -1).expand(3, B, -1)
                mrope_position_deltas = torch.zeros((B, 1), device=device, dtype=dtype_long)

            return position_ids, mrope_position_deltas

    # ===== vision feature helpers =====
    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        pixel_values_videos = pixel_values_videos.to(dtype=self.visual.dtype)
        # Return flattened visual tokens directly
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds  # shape: (sum_toks, hidden)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values = pixel_values.to(dtype=self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds  # shape: (sum_toks, hidden)

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

        # Image mask
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)

        # Length check only when not under CUDA graph capture
        if image_features is not None and not torch.cuda.is_current_stream_capturing():
            expected_elems = int(n_image_tokens.item()) * inputs_embeds.shape[-1]
            actual_elems = image_features.numel()
            if expected_elems != actual_elems:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens.item()}, "
                    f"features {image_features.shape[0]}"
                )

        # Video mask
        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)

        if video_features is not None and not torch.cuda.is_current_stream_capturing():
            expected_elems = int(n_video_tokens.item()) * inputs_embeds.shape[-1]
            actual_elems = video_features.numel()
            if expected_elems != actual_elems:
                raise ValueError(
                    f"Videos features and video tokens do not match: tokens: {n_video_tokens.item()}, "
                    f"features {video_features.shape[0]}"
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

        # Inject image tokens
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Inject video tokens
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Position ids & RoPE deltas
        if position_ids is not None:
            position_ids = position_ids.to(device=inputs_embeds.device, dtype=torch.long)
        else:
            # Only check prefill via past_kv; avoid reading GPU scalars directly (for CUDA graphs)
            is_prefill = (past_key_values is None) or (past_key_values.get_seq_length() == 0)
            if is_prefill or self.rope_deltas is None:
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
                base = torch.arange(seq_length, device=inputs_embeds.device).view(1, 1, -1).expand(
                    3, batch_size, -1
                )
                position_ids = base  # reuse base sequence indices; RoPE deltas tracked separately

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


# ================= Causal LM wrapper =================
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
        It is a [`~cache_utils.Cache`] instance. For more details, see our KV cache guide.

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

    def __init__(self, config: InfiniteVLConfig):
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
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
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
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
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
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do assisted decoding
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
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, )`)
            video_nums (`torch.LongTensor` of shape `(batch_size, )`)
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
        """
        Expand inputs (including multimodal vision features) for generation, in a GPU-only,
        block-wise manner that avoids Python lists and `.item()` for large tensors.
        """
        if expand_size == 1:
            return input_ids, model_kwargs

        device = None
        if input_ids is not None:
            device = input_ids.device
            B = input_ids.shape[0]
        else:
            # try infer from model_kwargs
            for v in model_kwargs.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
            for k in ("attention_mask", "position_ids", "inputs_embeds"):
                if isinstance(model_kwargs.get(k, None), torch.Tensor):
                    B = model_kwargs[k].shape[0]
                    break
            else:
                raise ValueError("Cannot infer batch size for expansion.")

        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ]
        sms = self.visual.spatial_merge_size

        # Helper: construct a block-wise repeat index on GPU
        def _block_repeat_index(lengths_1d: torch.Tensor, repeat_times: int) -> torch.Tensor:
            if lengths_1d is None:
                return None
            lengths = lengths_1d.to(device=device, dtype=torch.long)
            if lengths.numel() == 0:
                return torch.empty(0, dtype=torch.long, device=device)

            starts = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.long),
                    lengths.cumsum(0)[:-1],
                ],
                dim=0,
            )  # (B,)
            starts_rep = starts.repeat_interleave(repeat_times)
            lengths_rep = lengths.repeat_interleave(repeat_times)
            max_len = torch.clamp(lengths.max(), min=0)
            if max_len == 0:
                return torch.empty(0, dtype=torch.long, device=device)

            base = torch.arange(max_len, device=device, dtype=torch.long).view(1, -1)
            base = base.expand(starts_rep.shape[0], -1)
            mask = base < lengths_rep.view(-1, 1)
            idx = (base + starts_rep.view(-1, 1)).masked_select(mask)
            return idx

        def _repeat_by_sample_blocks(x: torch.Tensor, lengths_per_sample: torch.Tensor, repeat_times: int):
            if x is None:
                return None
            idx = _block_repeat_index(lengths_per_sample, repeat_times)
            if idx is None or idx.numel() == 0:
                return x.new_empty((0, *x.shape[1:]))
            return x.index_select(0, idx)

        # Visual-specific expansions
        image_grid_thw = model_kwargs.get("image_grid_thw", None)
        video_grid_thw = model_kwargs.get("video_grid_thw", None)

        image_nums, video_nums = self._get_image_nums_and_video_nums(
            input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
        )
        image_nums = image_nums.to(device=device, dtype=torch.long) if isinstance(image_nums, torch.Tensor) else None
        video_nums = video_nums.to(device=device, dtype=torch.long) if isinstance(video_nums, torch.Tensor) else None

        # pixel_values (images)
        if ("pixel_values" in model_kwargs) and (model_kwargs["pixel_values"] is not None) and (image_grid_thw is not None):
            per_img_tokens = (
                image_grid_thw.to(device=device, dtype=torch.long).prod(dim=1) // (sms * sms)
            ).to(torch.long)  # (N_img,)
            img_sample_id = torch.repeat_interleave(
                torch.arange(B, device=device, dtype=torch.long), image_nums
            )  # (N_img,)
            tokens_per_sample_img = torch.zeros(B, device=device, dtype=torch.long)
            tokens_per_sample_img.index_add_(0, img_sample_id, per_img_tokens)

            model_kwargs["pixel_values"] = _repeat_by_sample_blocks(
                model_kwargs["pixel_values"], tokens_per_sample_img, expand_size
            )

        # image_grid_thw
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = _repeat_by_sample_blocks(image_grid_thw, image_nums, expand_size)

        # pixel_values_videos
        if (
            ("pixel_values_videos" in model_kwargs)
            and (model_kwargs["pixel_values_videos"] is not None)
            and (video_grid_thw is not None)
        ):
            per_vid_tokens = (
                video_grid_thw.to(device=device, dtype=torch.long).prod(dim=1) // (sms * sms)
            ).to(torch.long)
            vid_sample_id = torch.repeat_interleave(
                torch.arange(B, device=device, dtype=torch.long), video_nums
            )
            tokens_per_sample_vid = torch.zeros(B, device=device, dtype=torch.long)
            tokens_per_sample_vid.index_add_(0, vid_sample_id, per_vid_tokens)

            model_kwargs["pixel_values_videos"] = _repeat_by_sample_blocks(
                model_kwargs["pixel_values_videos"], tokens_per_sample_vid, expand_size
            )

        # video_grid_thw
        if video_grid_thw is not None:
            model_kwargs["video_grid_thw"] = _repeat_by_sample_blocks(video_grid_thw, video_nums, expand_size)

        # second_per_grid_ts
        if (
            ("second_per_grid_ts" in model_kwargs)
            and (model_kwargs["second_per_grid_ts"] is not None)
            and (video_nums is not None)
        ):
            model_kwargs["second_per_grid_ts"] = _repeat_by_sample_blocks(
                model_kwargs["second_per_grid_ts"], video_nums, expand_size
            )

        # Non-visual tensors: simple repeat on batch dimension
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

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def allocate_inference_cache(self, batch_size: int):
        # Allocate static cache for text decoder
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
#InfiniteVLQwen2_5_VLForConditionalGeneration