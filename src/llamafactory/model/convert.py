from functools import partial
from tqdm import tqdm
import torch.nn as nn
from .fla import layers
from ..extras import logging
import importlib
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from .fla.models.utils import Cache

logger = logging.get_logger(__name__)

def convert_attention(model: nn.Module, 
                      finetuning_args: dict):
    """
    Call to convert all attention layers
    """
    all_layers = traverse_layers(model)

    for layer_idx, layer in enumerate(tqdm(all_layers, desc='Converting attentions...')):
        if layer_idx not in finetuning_args.softmax_attention:
            try:
                layer.self_attn = convert_quadratic_to_linear(layer, finetuning_args.mixer, **finetuning_args.mixer_config)
                layer.self_attn.converted = True
            except:
                layer.attention = convert_quadratic_to_linear(layer, finetuning_args.mixer, **finetuning_args.mixer_config)
                layer.attention.converted = True

        else:  # Freeze any preserved softmax attention layers
            for p in layer.parameters():
                p.requires_grad = False
                    
    return model

def traverse_layers(model: nn.Module, verbose: bool = False):
    """
    Return list of model layers
    """
    try:
        all_layers = model.model.layers
        if verbose:
            print('-> Loading from model.model.layers')
    except AttributeError as e: # if base model
        if verbose:
            print(e)
        try:
            all_layers = model.model.language_model.layers
            if verbose:
                print('-> Loading from model.model.language_model.layers')
        except AttributeError as e1:  # If we make a PEFT model
            if verbose:
                print(e1)
            try:
                all_layers = model.base_model.model.model.layers
                if verbose:
                    print('-> Loading from model.base_model.model.model.layers')
            except AttributeError as e2:
                if verbose:
                    print(e2)
                all_layers = model.language_model.model.layers
                if verbose:
                    print('-> Loading from model.language_model.model.layers')

    return all_layers

def convert_quadratic_to_linear(layer: nn.Module, mixer: str, **kwargs):
    """
    Replaces a specific attention layer with a linear attention mixer (Mamba2_new or GatedDeltaNet).

    Args:
        layer (nn.Module): The transformer layer containing the original self-attention.
        mixer (str): The name of the target linear attention class (e.g., "Mamba2_new").
        **kwargs: Additional arguments passed to the mixer's constructor.
    """
    
    # Dynamically retrieve the parent class based on the mixer string
    ParentClass = getattr(layers, mixer, None)
    if ParentClass is None:
        raise ValueError(f"Mixer class '{mixer}' not found in 'layers' module.")

    class MixerWrapper(ParentClass):
        def __init__(self, layer, mixer, **kwargs):
            src = layer.self_attn  # Source attention layer reference
            
            # ------------------------------------------------------------------
            # Logic for Mamba2_new
            # ------------------------------------------------------------------
            if mixer == "Mamba2_new":
                super().__init__(
                    hidden_size=src.hidden_size,
                    num_heads=src.num_heads,
                    head_dim=src.head_dim,
                    layer_idx=src.layer_idx,
                    **kwargs
                )
                
                # Copy Query projection
                self.q_proj = src.q_proj
                
                # Handle Key and Value projections (Handling Grouped Query Attention)
                # We need to expand KV heads to match the number of Q heads for this specific implementation.
                k_weight = src.k_proj.weight.data
                v_weight = src.v_proj.weight.data

                # 1. Reshape to separate heads: (num_kv_heads, head_dim, hidden_size)
                k_weight = k_weight.view(src.num_key_value_heads, self.head_dim, self.hidden_size)
                v_weight = v_weight.view(src.num_key_value_heads, self.head_dim, self.hidden_size)

                # 2. Repeat along the num_heads dimension to match Query heads
                # (num_heads, head_dim, hidden_size)
                k_weight = k_weight.repeat(src.num_key_value_groups, 1, 1)
                v_weight = v_weight.repeat(src.num_key_value_groups, 1, 1)

                # 3. Flatten back to linear layer format: (num_heads * head_dim, hidden_size)
                k_weight = k_weight.reshape(self.num_heads * self.head_dim, self.hidden_size)
                v_weight = v_weight.reshape(self.num_heads * self.head_dim, self.hidden_size)

                # Assign processed weights
                self.k_proj.weight.data = k_weight
                self.v_proj.weight.data = v_weight

                # Handle Biases if they exist
                if src.k_proj.bias is not None:
                    k_bias = src.k_proj.bias.data
                    v_bias = src.v_proj.bias.data
                    
                    # Reshape -> Repeat -> Flatten (Same logic as weights)
                    k_bias = k_bias.view(src.num_key_value_heads, self.head_dim)
                    v_bias = v_bias.view(src.num_key_value_heads, self.head_dim)
                    k_bias = k_bias.repeat(src.num_key_value_groups, 1)
                    v_bias = v_bias.repeat(src.num_key_value_groups, 1)
                    k_bias = k_bias.reshape(self.num_heads * self.head_dim)
                    v_bias = v_bias.reshape(self.num_heads * self.head_dim)
                    
                    self.k_proj.bias.data = k_bias
                    self.v_proj.bias.data = v_bias

                self.o_proj = src.o_proj

            # ------------------------------------------------------------------
            # Logic for GatedDeltaNet
            # ------------------------------------------------------------------
            elif mixer == "GatedDeltaNet":
                super().__init__(
                    hidden_size=src.hidden_size,
                    num_heads=src.num_heads,
                    head_dim=src.head_dim,
                    expand_v=2,  # Specific expansion factor for GatedDeltaNet
                    layer_idx=src.layer_idx,
                    **kwargs
                )
                
                # Note: GatedDeltaNet logic provided did not explicitly copy o_proj.
                # Ensure this is intended behavior for your architecture.
            
            else:
                raise ValueError(f"Initialization logic for {mixer} is not defined.")

            # ------------------------------------------------------------------
            # Common Finalization: Device and Dtype casting
            # ------------------------------------------------------------------
            # Determine target device/dtype from the original query projection
            target_device = src.q_proj.weight.device
            target_dtype = src.q_proj.weight.dtype

            self.device = target_device
            self.dtype = target_dtype

            # Move all parameters of the new mixer to the correct device and dtype
            for param in self.parameters():
                param.data = param.data.to(device=target_device, dtype=target_dtype)

    return MixerWrapper(layer, mixer, **kwargs)




