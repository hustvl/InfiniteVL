# coding=utf-8
# Copyright 2025 The HustVL Team.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen2.5-VL, which is derived from EleutherAI's GPT-NeoX library
# and the GPT-NeoX and OPT implementations. It has been modified to create InfiniteVL.
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

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation


class InfiniteVLVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InfiniteVLVisionModel`].

    Args:
        depth (`int`, *optional*, defaults to 32):
            The number of layers in the vision transformer.
        hidden_size (`int`, *optional*, defaults to 3584):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        intermediate_size (`int`, *optional*, defaults to 3420):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The scaling factor for spatial merging of patches.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The size of patches along the temporal dimension.
        tokens_per_second (`int`, *optional*, defaults to 4):
            Number of tokens processed per second for video inputs.
        window_size (`int`, *optional*, defaults to 112):
            The window size for windowed attention mechanisms.
        out_hidden_size (`int`, *optional*, defaults to 3584):
            Dimensionality of the output hidden states.
        fullatt_block_indexes (`list`, *optional*):
            Indices of blocks that use full attention instead of windowed attention.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "infinite_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if fullatt_block_indexes is None:
            fullatt_block_indexes = [7, 15, 23, 31]

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range


class InfiniteVLTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InfiniteVLTextModel`]. It is used to instantiate an
    InfiniteVL model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the InfiniteVL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`InfiniteVLModel`]
        hidden_size (`int`, *optional*, defaults to 8192):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 29568):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 32768):
            Sliding window attention (SWA) window size.
        max_window_layers (`int`, *optional*, defaults to 80):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
        expand_v (`float`, *optional*, defaults to 2):
            Expansion factor for the value dimension in the linear attention/DeltaNet layer.
        mode (`str`, *optional*, defaults to `"chunk"`):
            Execution mode for the linear attention layer (e.g., "chunk" or "fused_recurrent").
        use_gate (`bool`, *optional*, defaults to `True`):
            Whether to use the gating mechanism in the DeltaNet layer.
        use_short_conv (`bool`, *optional*, defaults to `True`):
            Whether to use short convolution in the linear attention layer.
        conv_size (`int`, *optional*, defaults to 4):
            Kernel size for the short convolution.
        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the short convolution.
        num_linear_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key/value heads used in the linear attention layers.
        num_linear_heads (`int`, *optional*, defaults to 16):
            Number of query heads used in the linear attention layers.
        linear_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each head in the linear attention layers.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon value for normalization layers in the linear attention branch.

    ```python
    >>> from transformers import InfiniteVLTextModel, InfiniteVLConfig

    >>> # Initializing an InfiniteVL style configuration
    >>> configuration = InfiniteVLConfig()

    >>> # Initializing a model from the InfiniteVL style configuration
    >>> model = InfiniteVLTextModel(configuration.text_config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "infinite_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `InfiniteVL`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=80,
        layer_types=None,
        attention_dropout=0.0,
        rope_scaling=None,
        expand_v: float = 2,
        mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_linear_key_value_heads: int = 16,
        num_linear_heads: int = 16,
        linear_head_dim: int = 128,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        # DeltaNet / linear branch
        self.expand_v = expand_v
        self.mode = mode
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.num_linear_key_value_heads = num_linear_key_value_heads
        self.num_linear_heads = num_linear_heads
        self.linear_head_dim = linear_head_dim
        self.norm_eps = norm_eps

        self.layer_types = layer_types
        if self.layer_types is None:
            # Default: one sliding_attention layer followed by three linear_attention layers (period = 4)
            self.layer_types = [
                "linear_attention" if bool(i % 4) else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]

        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # Validate the correctness of rotary position embeddings parameters
        # Backward Compatibility: if there is a 'type' field, move it to 'rope_type'.
        # Also change type from 'mrope' to 'default' because `mrope` uses default RoPE calculations in this architecture.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        rope_config_validation(self, ignore_keys={"mrope_section"})
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class InfiniteVLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InfiniteVLModel`]. It is used to instantiate an
    InfiniteVL model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `InfiniteVLTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `InfiniteVLVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the video prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The token index to denote start of vision input.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The token index to denote end of vision input.

    ```python
    >>> from transformers import InfiniteVLQwen2_5_VLForConditionalGeneration, InfiniteVLConfig

    >>> # Initializing an InfiniteVL style configuration
    >>> configuration = InfiniteVLConfig()

    >>> # Initializing a model from the InfiniteVL style configuration
    >>> model = InfiniteVLQwen2_5_VLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "infinite_vl"
    sub_configs = {"vision_config": InfiniteVLVisionConfig, "text_config": InfiniteVLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        **kwargs,
    ):
        # We need to init super() here so that it does not reset values
        # that are in text config to the BaseClass defaults. The Base
        # config has many text related defaults and not all defaults are same as for `InfiniteVLTextConfig`
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # Attention implementation to use. It sets it recursively on sub-configs so we call it again in the end
        self._attn_implementation = kwargs.pop("attn_implementation", None)

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in ["dtype", "_attn_implementation_internal"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "dtype",
            "_attn_implementation_internal",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)


__all__ = ["InfiniteVLConfig", "InfiniteVLTextConfig", "InfiniteVLVisionConfig"]