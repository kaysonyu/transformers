# coding=utf-8
# Copyright 2025 OpenMOSS and HuggingFace Inc. teams. All rights reserved.
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
"""XY-Tokenizer model configuration"""

from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class XYTokenizerEncoderConfig(PretrainedConfig):
    r"""
    Configuration class for XYTokenizer encoder (used by both semantic and acoustic encoders).

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel filterbanks.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Audio sampling rate in Hz.
        hop_length (`int`, *optional*, defaults to 160):
            STFT hop length.
        stride_size (`int`, *optional*, defaults to 2):
            Convolutional stride for downsampling.
        kernel_size (`int`, *optional*, defaults to 3):
            Convolutional kernel size.
        d_model (`int`, *optional*, defaults to 1280):
            Hidden dimension size.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Whether to scale embeddings by sqrt(d_model).
        max_audio_seconds (`int`, *optional*, defaults to 30):
            Maximum audio duration in seconds.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of transformer encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads in encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Feed-forward dimension in encoder.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function name.
        attn_type (`str`, *optional*, defaults to `"varlen"`):
            Attention type.
    """

    model_type = "xy_tokenizer_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        stride_size: int = 2,
        kernel_size: int = 3,
        d_model: int = 1280,
        scale_embedding: bool = True,
        max_audio_seconds: int = 30,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        activation_function: str = "gelu",
        attn_type: str = "varlen",
        **kwargs,
    ):
        self.num_mel_bins = num_mel_bins
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.stride_size = stride_size
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.scale_embedding = scale_embedding
        self.max_audio_seconds = max_audio_seconds
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.activation_function = activation_function
        self.attn_type = attn_type
        super().__init__(**kwargs)


class XYTokenizerDecoderConfig(PretrainedConfig):
    r"""
    Configuration class for XYTokenizer decoder.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel filterbanks.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Audio sampling rate in Hz.
        hop_length (`int`, *optional*, defaults to 160):
            STFT hop length.
        stride_size (`int`, *optional*, defaults to 2):
            Convolutional stride for downsampling.
        kernel_size (`int`, *optional*, defaults to 3):
            Convolutional kernel size.
        d_model (`int`, *optional*, defaults to 1280):
            Hidden dimension size.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Whether to scale embeddings by sqrt(d_model).
        max_audio_seconds (`int`, *optional*, defaults to 30):
            Maximum audio duration in seconds.
        decoder_layers (`int`, *optional*, defaults to 32):
            Number of transformer decoder layers.
        decoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads in decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Feed-forward dimension in decoder.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function name.
        attn_type (`str`, *optional*, defaults to `"varlen"`):
            Attention type.
    """

    model_type = "xy_tokenizer_decoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        stride_size: int = 2,
        kernel_size: int = 3,
        d_model: int = 1280,
        scale_embedding: bool = True,
        max_audio_seconds: int = 30,
        decoder_layers: int = 32,
        decoder_attention_heads: int = 20,
        decoder_ffn_dim: int = 5120,
        activation_function: str = "gelu",
        attn_type: str = "varlen",
        **kwargs,
    ):
        self.num_mel_bins = num_mel_bins
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.stride_size = stride_size
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.scale_embedding = scale_embedding
        self.max_audio_seconds = max_audio_seconds
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.activation_function = activation_function
        self.attn_type = attn_type
        super().__init__(**kwargs)


class XYTokenizerTransformerConfig(PretrainedConfig):
    r"""
    Configuration class for XYTokenizer transformer adapter modules.

    Args:
        input_dim (`int`, *optional*, defaults to 1280):
            Input feature dimension.
        d_model (`int`, *optional*, defaults to 1280):
            Hidden dimension size.
        output_dim (`int`, *optional*, defaults to 1280):
            Output feature dimension.
        max_source_positions (`int`, *optional*, defaults to 1500):
            Maximum sequence length.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of transformer layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Feed-forward dimension.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function name.
        attn_type (`str`, *optional*, defaults to `"varlen"`):
            Attention type.
    """

    model_type = "xy_tokenizer_transformer"

    def __init__(
        self,
        input_dim: int = 1280,
        d_model: int = 1280,
        output_dim: int = 1280,
        max_source_positions: int = 1500,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        activation_function: str = "gelu",
        attn_type: str = "varlen",
        **kwargs,
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.max_source_positions = max_source_positions
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.activation_function = activation_function
        self.attn_type = attn_type
        super().__init__(**kwargs)


class XYTokenizerResidualVQConfig(PretrainedConfig):
    r"""
    Configuration class for XYTokenizer residual vector quantizer.

    Args:
        input_dim (`int`, *optional*, defaults to 1280):
            Input dimension.
        rvq_dim (`int`, *optional*):
            Internal RVQ dimension. Defaults to `input_dim` if not specified.
        output_dim (`int`, *optional*):
            Output dimension. Defaults to `input_dim` if not specified.
        num_quantizers (`int`, *optional*, defaults to 32):
            Number of quantizer layers in the residual VQ.
        codebook_size (`int`, *optional*, defaults to 1024):
            Number of entries in each codebook.
        codebook_dim (`int`, *optional*, defaults to 8):
            Dimension of each codebook vector.
        quantizer_dropout (`float`, *optional*, defaults to 0.5):
            Dropout ratio for quantizers during training.
        skip_rvq_ratio (`float`, *optional*, defaults to 0.0):
            Ratio for skipping RVQ during training.
        commitment (`float`, *optional*, defaults to 1.0):
            Commitment loss weight for vector quantization.
        decay (`float`, *optional*, defaults to 0.99):
            Decay rate for exponential moving average of codebook.
        epsilon (`float`, *optional*, defaults to 1e-5):
            Small constant for numerical stability.
        threshold_ema_dead (`int`, *optional*, defaults to 2):
            Threshold for detecting dead codebook entries.
        kmeans_init (`bool`, *optional*, defaults to `True`):
            Whether to use k-means initialization for codebook.
        kmeans_iters (`int`, *optional*, defaults to 10):
            Number of k-means iterations for initialization.
    """

    model_type = "xy_tokenizer_residual_vq"

    def __init__(
        self,
        input_dim: int = 1280,
        rvq_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        skip_rvq_ratio: float = 0.0,
        commitment: float = 1.0,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead: int = 2,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.rvq_dim = rvq_dim
        self.output_dim = output_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.skip_rvq_ratio = skip_rvq_ratio
        # VectorQuantizer parameters (flattened into this config)
        self.commitment = commitment
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead = threshold_ema_dead
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        super().__init__(**kwargs)


class XYTokenizerConvolutionConfig(PretrainedConfig):
    r"""
    Configuration class for XYTokenizer convolution-based components (downsample and upsample).

    Args:
        d_model (`int`, *optional*, defaults to 1280):
            Model dimension.
        downsample_avg_pooler (`int`, *optional*, defaults to 4):
            Downsampling pooling factor for ResidualDownConv.
        upsample_stride (`int`, *optional*, defaults to 4):
            Upsampling stride for UpConv.
    """

    model_type = "xy_tokenizer_convolution"

    def __init__(
        self,
        d_model: int = 1280,
        downsample_avg_pooler: int = 4,
        upsample_stride: int = 4,
        **kwargs,
    ):
        self.d_model = d_model
        self.downsample_avg_pooler = downsample_avg_pooler
        self.upsample_stride = upsample_stride
        super().__init__(**kwargs)


class XYTokenizerVocosConfig(PretrainedConfig):
    r"""
    Configuration class for XYTokenizer Vocos vocoder.

    Args:
        input_channels (`int`, *optional*, defaults to 128):
            Number of input channels.
        dim (`int`, *optional*, defaults to 512):
            Model dimension.
        intermediate_dim (`int`, *optional*, defaults to 4096):
            Feed-forward intermediate dimension.
        num_layers (`int`, *optional*, defaults to 30):
            Number of ConvNeXt layers in the backbone.
        n_fft (`int`, *optional*, defaults to 640):
            FFT size for ISTFT.
        hop_size (`int`, *optional*, defaults to 160):
            STFT hop size.
        padding (`str`, *optional*, defaults to `"same"`):
            Padding mode for convolutions.
    """

    model_type = "xy_tokenizer_vocos"

    def __init__(
        self,
        input_channels: int = 128,
        dim: int = 512,
        intermediate_dim: int = 4096,
        num_layers: int = 30,
        n_fft: int = 640,
        hop_size: int = 160,
        padding: str = "same",
        **kwargs,
    ):
        self.input_channels = input_channels
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.padding = padding
        super().__init__(**kwargs)


class XYTokenizerFeatureExtractorConfig(PretrainedConfig):
    r"""
    Configuration class for XYTokenizer feature extractor.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            Number of mel filterbanks.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Audio sampling rate in Hz.
        hop_length (`int`, *optional*, defaults to 160):
            STFT hop length.
        chunk_length (`int`, *optional*, defaults to 30):
            Chunk length in seconds for processing.
        n_fft (`int`, *optional*, defaults to 400):
            FFT size for STFT.
        n_samples (`int`, *optional*, defaults to 480000):
            Number of samples (chunk_length * sampling_rate).
        nb_max_frames (`int`, *optional*, defaults to 3000):
            Maximum number of frames.
        padding_side (`str`, *optional*, defaults to `"right"`):
            Side to pad sequences.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used for padding.
        dither (`float`, *optional*, defaults to 0.0):
            Dither amount for audio preprocessing.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return attention masks.
        max_frequency (`float`, *optional*):
            Maximum frequency for mel filterbank. Defaults to sampling_rate / 2.
        batch_size (`int`, *optional*, defaults to 8):
            Batch size for streaming processing.
        overlap_side (`str`, *optional*, defaults to `"both"`):
            Side for overlap in chunked processing.
    """

    model_type = "xy_tokenizer_feature_extractor"

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        n_samples: int = 480000,
        nb_max_frames: int = 3000,
        padding_side: str = "right",
        padding_value: float = 0.0,
        dither: float = 0.0,
        return_attention_mask: bool = False,
        max_frequency: Optional[float] = None,
        batch_size: int = 8,
        overlap_side: str = "both",
        **kwargs,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_fft = n_fft
        self.n_samples = n_samples
        self.nb_max_frames = nb_max_frames
        self.padding_side = padding_side
        self.padding_value = padding_value
        self.dither = dither
        self.return_attention_mask = return_attention_mask
        self.max_frequency = max_frequency
        self.batch_size = batch_size
        self.overlap_side = overlap_side
        super().__init__(**kwargs)


class XYTokenizerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XYTokenizer`]. It is used to instantiate a
    XY-Tokenizer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the XY-Tokenizer
    [fnlp/XY_Tokenizer_TTSD_V0_hf](https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0_hf) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        input_sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate of the input audio.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate of the output audio.
        encoder_downsample_rate (`int`, *optional*, defaults to 1280):
            The total downsampling factor of the encoder part.
        decoder_upsample_rate (`int`, *optional*, defaults to 1920):
            The total upsampling factor of the decoder part.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for weight initialization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        semantic_encoder_config (`Union[dict, XYTokenizerEncoderConfig]`, *optional*):
            Configuration for the semantic encoder. If not provided, uses default XYTokenizerEncoderConfig.
        acoustic_encoder_config (`Union[dict, XYTokenizerEncoderConfig]`, *optional*):
            Configuration for the acoustic encoder. If not provided, uses default XYTokenizerEncoderConfig.
        semantic_encoder_adapter_config (`Union[dict, XYTokenizerTransformerConfig]`, *optional*):
            Configuration for the semantic encoder adapter. If not provided, uses default XYTokenizerTransformerConfig.
        pre_rvq_adapter_config (`Union[dict, XYTokenizerTransformerConfig]`, *optional*):
            Configuration for the pre-RVQ adapter. If not provided, uses default XYTokenizerTransformerConfig.
        post_rvq_adapter_config (`Union[dict, XYTokenizerTransformerConfig]`, *optional*):
            Configuration for the post-RVQ adapter. If not provided, uses default XYTokenizerTransformerConfig.
        acoustic_decoder_config (`Union[dict, XYTokenizerDecoderConfig]`, *optional*):
            Configuration for the acoustic decoder. If not provided, uses default XYTokenizerDecoderConfig.
        quantizer_config (`Union[dict, XYTokenizerResidualVQConfig]`, *optional*):
            Configuration for the residual vector quantizer. If not provided, uses default XYTokenizerResidualVQConfig.
        convolution_config (`Union[dict, XYTokenizerConvolutionConfig]`, *optional*):
            Configuration for convolution components (downsample and upsample). If not provided, uses default
            XYTokenizerConvolutionConfig.
        vocos_config (`Union[dict, XYTokenizerVocosConfig]`, *optional*):
            Configuration for the Vocos vocoder. If not provided, uses default XYTokenizerVocosConfig.
        feature_extractor_config (`Union[dict, XYTokenizerFeatureExtractorConfig]`, *optional*):
            Configuration for the feature extractor. If not provided, uses default XYTokenizerFeatureExtractorConfig.

    Example:

    ```python
    >>> from transformers import XYTokenizerConfig, XYTokenizer

    >>> # Initializing a XY-Tokenizer configuration
    >>> configuration = XYTokenizerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = XYTokenizer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "xy_tokenizer"

    sub_configs = {
        "semantic_encoder_config": XYTokenizerEncoderConfig,
        "acoustic_encoder_config": XYTokenizerEncoderConfig,
        "semantic_encoder_adapter_config": XYTokenizerTransformerConfig,
        "pre_rvq_adapter_config": XYTokenizerTransformerConfig,
        "post_rvq_adapter_config": XYTokenizerTransformerConfig,
        "acoustic_decoder_config": XYTokenizerDecoderConfig,
        "quantizer_config": XYTokenizerResidualVQConfig,
        "convolution_config": XYTokenizerConvolutionConfig,
        "vocos_config": XYTokenizerVocosConfig,
        "feature_extractor_config": XYTokenizerFeatureExtractorConfig,
    }

    def __init__(
        self,
        input_sampling_rate: int = 16000,
        sampling_rate: int = 16000,
        encoder_downsample_rate: int = 1280,
        decoder_upsample_rate: int = 1920,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        semantic_encoder_config: Optional[Union[dict, XYTokenizerEncoderConfig]] = None,
        acoustic_encoder_config: Optional[Union[dict, XYTokenizerEncoderConfig]] = None,
        semantic_encoder_adapter_config: Optional[Union[dict, XYTokenizerTransformerConfig]] = None,
        pre_rvq_adapter_config: Optional[Union[dict, XYTokenizerTransformerConfig]] = None,
        post_rvq_adapter_config: Optional[Union[dict, XYTokenizerTransformerConfig]] = None,
        acoustic_decoder_config: Optional[Union[dict, XYTokenizerDecoderConfig]] = None,
        quantizer_config: Optional[Union[dict, XYTokenizerResidualVQConfig]] = None,
        convolution_config: Optional[Union[dict, XYTokenizerConvolutionConfig]] = None,
        vocos_config: Optional[Union[dict, XYTokenizerVocosConfig]] = None,
        feature_extractor_config: Optional[Union[dict, XYTokenizerFeatureExtractorConfig]] = None,
        **kwargs,
    ):
        # Backward compatibility: handle old params pattern
        if "params" in kwargs:
            logger.warning(
                "The 'params' configuration pattern is deprecated and will be removed in "
                "Transformers v5.0. Please use explicit sub_configs instead. "
                "Your configuration will be automatically migrated."
            )
            params = kwargs.pop("params")
            self._migrate_from_params(params)
        else:
            # Initialize each sub-config
            self.semantic_encoder_config = self._init_subconfig(semantic_encoder_config, XYTokenizerEncoderConfig)
            self.acoustic_encoder_config = self._init_subconfig(acoustic_encoder_config, XYTokenizerEncoderConfig)
            self.semantic_encoder_adapter_config = self._init_subconfig(
                semantic_encoder_adapter_config, XYTokenizerTransformerConfig
            )
            self.pre_rvq_adapter_config = self._init_subconfig(pre_rvq_adapter_config, XYTokenizerTransformerConfig)
            self.post_rvq_adapter_config = self._init_subconfig(post_rvq_adapter_config, XYTokenizerTransformerConfig)
            self.acoustic_decoder_config = self._init_subconfig(acoustic_decoder_config, XYTokenizerDecoderConfig)
            self.quantizer_config = self._init_subconfig(quantizer_config, XYTokenizerResidualVQConfig)
            self.convolution_config = self._init_subconfig(convolution_config, XYTokenizerConvolutionConfig)
            self.vocos_config = self._init_subconfig(vocos_config, XYTokenizerVocosConfig)
            self.feature_extractor_config = self._init_subconfig(
                feature_extractor_config, XYTokenizerFeatureExtractorConfig
            )

        # Backward-compatible alias handling
        if "input_sample_rate" in kwargs and input_sampling_rate == 16000:
            input_sampling_rate = kwargs.pop("input_sample_rate")
        if "output_sample_rate" in kwargs and sampling_rate == 16000:
            sampling_rate = kwargs.pop("output_sample_rate")

        # Set top-level attributes
        self.input_sampling_rate = input_sampling_rate
        self.sampling_rate = sampling_rate
        self.encoder_downsample_rate = encoder_downsample_rate
        self.decoder_upsample_rate = decoder_upsample_rate
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Keep deprecated names for backward compatibility
        self.input_sample_rate = input_sampling_rate
        self.output_sample_rate = sampling_rate

        super().__init__(**kwargs)

    def _init_subconfig(self, config: Optional[Union[dict, PretrainedConfig]], config_class: type) -> PretrainedConfig:
        """
        Initialize a sub-config from None, dict, or existing config instance.

        Args:
            config: The configuration to initialize (None, dict, or config instance)
            config_class: The config class to instantiate if needed

        Returns:
            An instance of config_class
        """
        if config is None:
            return config_class()
        elif isinstance(config, dict):
            return config_class(**config)
        else:
            return config

    def _migrate_from_params(self, params: dict) -> None:
        """
        Migrate old params dictionary to new sub_configs structure.

        This ensures models saved with old config format can be loaded.

        Args:
            params: Dictionary containing old-style nested parameter dictionaries
        """
        # Migrate semantic encoder
        if "semantic_encoder_kwargs" in params:
            self.semantic_encoder_config = XYTokenizerEncoderConfig(**params["semantic_encoder_kwargs"])
        else:
            self.semantic_encoder_config = XYTokenizerEncoderConfig()

        # Migrate acoustic encoder
        if "acoustic_encoder_kwargs" in params:
            self.acoustic_encoder_config = XYTokenizerEncoderConfig(**params["acoustic_encoder_kwargs"])
        else:
            self.acoustic_encoder_config = XYTokenizerEncoderConfig()

        # Migrate semantic encoder adapter
        if "semantic_encoder_adapter_kwargs" in params:
            self.semantic_encoder_adapter_config = XYTokenizerTransformerConfig(
                **params["semantic_encoder_adapter_kwargs"]
            )
        else:
            self.semantic_encoder_adapter_config = XYTokenizerTransformerConfig()

        # Migrate pre-RVQ adapter
        if "pre_rvq_adapter_kwargs" in params:
            self.pre_rvq_adapter_config = XYTokenizerTransformerConfig(**params["pre_rvq_adapter_kwargs"])
        else:
            self.pre_rvq_adapter_config = XYTokenizerTransformerConfig()

        # Migrate post-RVQ adapter
        if "post_rvq_adapter_kwargs" in params:
            self.post_rvq_adapter_config = XYTokenizerTransformerConfig(**params["post_rvq_adapter_kwargs"])
        else:
            self.post_rvq_adapter_config = XYTokenizerTransformerConfig()

        # Migrate acoustic decoder
        if "acoustic_decoder_kwargs" in params:
            self.acoustic_decoder_config = XYTokenizerDecoderConfig(**params["acoustic_decoder_kwargs"])
        else:
            self.acoustic_decoder_config = XYTokenizerDecoderConfig()

        # Migrate quantizer (special handling for VectorQuantizerConfig)
        if "quantizer_kwargs" in params:
            q_kwargs = params["quantizer_kwargs"].copy()
            # VectorQuantizerConfig parameters are now flattened into XYTokenizerResidualVQConfig
            # Extract vq_config if it exists as a dict and merge its parameters
            if "vq_config" in q_kwargs:
                vq_config = q_kwargs.pop("vq_config")
                if isinstance(vq_config, dict):
                    # Merge VQ config parameters into quantizer kwargs
                    q_kwargs.update(vq_config)
            self.quantizer_config = XYTokenizerResidualVQConfig(**q_kwargs)
        else:
            self.quantizer_config = XYTokenizerResidualVQConfig()

        # Migrate convolution components (downsample + upsample)
        conv_kwargs = {}
        if "downsample_kwargs" in params:
            conv_kwargs["downsample_avg_pooler"] = params["downsample_kwargs"].get("avg_pooler", 4)
            conv_kwargs["d_model"] = params["downsample_kwargs"].get("d_model", 1280)
        if "upsample_kwargs" in params:
            conv_kwargs["upsample_stride"] = params["upsample_kwargs"].get("stride", 4)
            if "d_model" not in conv_kwargs:
                conv_kwargs["d_model"] = params["upsample_kwargs"].get("d_model", 1280)
        self.convolution_config = (
            XYTokenizerConvolutionConfig(**conv_kwargs) if conv_kwargs else XYTokenizerConvolutionConfig()
        )

        # Migrate Vocos
        if "vocos_kwargs" in params:
            self.vocos_config = XYTokenizerVocosConfig(**params["vocos_kwargs"])
        else:
            self.vocos_config = XYTokenizerVocosConfig()

        # Migrate feature extractor
        if "feature_extractor_kwargs" in params:
            self.feature_extractor_config = XYTokenizerFeatureExtractorConfig(**params["feature_extractor_kwargs"])
        else:
            self.feature_extractor_config = XYTokenizerFeatureExtractorConfig()


__all__ = [
    "XYTokenizerConfig",
    "XYTokenizerEncoderConfig",
    "XYTokenizerDecoderConfig",
    "XYTokenizerTransformerConfig",
    "XYTokenizerResidualVQConfig",
    "XYTokenizerConvolutionConfig",
    "XYTokenizerVocosConfig",
    "XYTokenizerFeatureExtractorConfig",
]
