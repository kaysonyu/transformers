# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Convert XY-Tokenizer checkpoints from MOSS-TTSD to Transformers format."""

import argparse
import os

import torch
import yaml

from transformers import logging as transformers_logging
from transformers.models.xy_tokenizer.configuration_xy_tokenizer import XYTokenizerConfig
from transformers.models.xy_tokenizer.feature_extraction_xy_tokenizer import XYTokenizerFeatureExtractor
from transformers.models.xy_tokenizer.modeling_xy_tokenizer import XYTokenizer


transformers_logging.set_verbosity_info()
logger = transformers_logging.get_logger("transformers.models.xy_tokenizer")


def load_config_from_yaml(yaml_path: str) -> XYTokenizerConfig:
    """
    Load configuration from YAML file and convert to XYTokenizerConfig.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        XYTokenizerConfig instance.
    """
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    params = yaml_config.get("generator_params", {})

    # Extract top-level parameters
    config = XYTokenizerConfig(
        input_sampling_rate=params.get("input_sample_rate", 16000),
        sampling_rate=params.get("output_sample_rate", 32000),
        encoder_downsample_rate=params.get("encoder_downsample_rate", 1280),
        decoder_upsample_rate=params.get("decoder_upsample_rate", 2560),
        # Sub-configurations
        semantic_encoder_config=params.get("semantic_encoder_kwargs"),
        acoustic_encoder_config=params.get("acoustic_encoder_kwargs"),
        semantic_encoder_adapter_config=params.get("semantic_encoder_adapter_kwargs"),
        pre_rvq_adapter_config=params.get("pre_rvq_adapter_kwargs"),
        post_rvq_adapter_config=params.get("post_rvq_adapter_kwargs"),
        acoustic_decoder_config=params.get("acoustic_decoder_kwargs"),
        quantizer_config=params.get("quantizer_kwargs"),
        downsample_config=params.get("downsample_kwargs", {}),
        upsample_config=params.get("upsample_kwargs", {}),
        vocos_config=params.get("vocos_kwargs"),
        feature_extractor_config=params.get("feature_extractor_kwargs"),
    )

    return config


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    """
    Recursively set weight in the HF model.

    Args:
        hf_pointer: Pointer to the model module.
        key: Key path within the module.
        value: Weight value to set.
        full_name: Full key name for logging.
        weight_type: Type of weight (weight, bias, weight_g, weight_v, etc.).

    Note:
        Handles both old and new PyTorch weight_norm APIs.
        Old API (PyTorch < 1.12): weight_g, weight_v
        New API (PyTorch >= 1.12): parametrizations.weight.original0, original1
        Defaults to old API for compatibility with existing checkpoints.
    """
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # Handle weight_norm parameters (old API: weight_g/weight_v, new API: parametrizations.weight.original0/1)
    if weight_type == "weight_g":
        if hasattr(hf_pointer, "weight_g"):
            # Old API: nn.utils.weight_norm
            hf_shape = hf_pointer.weight_g.shape
        elif hasattr(hf_pointer, "parametrizations"):
            # New API: nn.utils.parametrize
            hf_shape = hf_pointer.parametrizations.weight.original0.shape
        else:
            raise AttributeError(
                f"{full_name} has neither weight_g (old API) nor parametrizations (new API)"
            )
    elif weight_type == "weight_v":
        if hasattr(hf_pointer, "weight_v"):
            # Old API: nn.utils.weight_norm
            hf_shape = hf_pointer.weight_v.shape
        elif hasattr(hf_pointer, "parametrizations"):
            # New API: nn.utils.parametrize
            hf_shape = hf_pointer.parametrizations.weight.original1.shape
        else:
            raise AttributeError(
                f"{full_name} has neither weight_v (old API) nor parametrizations (new API)"
            )
    elif weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + ('.' + weight_type if weight_type else '')} is {hf_shape}, "
            f"but should be {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        if hasattr(hf_pointer, "weight_g"):
            # Old API: nn.utils.weight_norm
            hf_pointer.weight_g.data = value
        elif hasattr(hf_pointer, "parametrizations"):
            # New API: nn.utils.parametrize
            hf_pointer.parametrizations.weight.original0.data = value
        else:
            raise AttributeError(
                f"{full_name} has neither weight_g (old API) nor parametrizations (new API)"
            )
    elif weight_type == "weight_v":
        if hasattr(hf_pointer, "weight_v"):
            # Old API: nn.utils.weight_norm
            hf_pointer.weight_v.data = value
        elif hasattr(hf_pointer, "parametrizations"):
            # New API: nn.utils.parametrize
            hf_pointer.parametrizations.weight.original1.data = value
        else:
            raise AttributeError(
                f"{full_name} has neither weight_v (old API) nor parametrizations (new API)"
            )
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + ('.' + weight_type if weight_type else '')} was initialized from {full_name}.")


@torch.no_grad()
def recursively_load_weights(orig_dict, hf_model):
    """
    Recursively load weights from original checkpoint to HF model.

    Args:
        orig_dict: Original checkpoint state dict.
        hf_model: HF model to load weights into.

    Returns:
        List of unused weight names.
    """
    unused_weights = []
    used_keys = set()

    for name, value in orig_dict.items():
        is_used = False

        # Handle weight norm parameters
        if "weight_g" in name:
            weight_type = "weight_g"
        elif "weight_v" in name:
            weight_type = "weight_v"
        elif "bias" in name:
            weight_type = "bias"
        elif "weight" in name:
            weight_type = "weight"
        else:
            # For buffers like inited, cluster_size, embed_avg, etc.
            weight_type = None

        # Map key paths from original to HF format
        mapped_name = name
        # Handle MLP layer naming: layers.N.fc1/fc2 -> layers.N.mlp.fc1/fc2
        if ".layers." in name and (".fc1." in name or ".fc2." in name or name.endswith(".fc1") or name.endswith(".fc2")):
            mapped_name = mapped_name.replace(".fc1.", ".mlp.fc1.")
            mapped_name = mapped_name.replace(".fc2.", ".mlp.fc2.")
            # Handle the case when it ends with fc1/fc2
            if mapped_name.endswith(".fc1"):
                mapped_name = mapped_name[:-4] + ".mlp.fc1"
            elif mapped_name.endswith(".fc2"):
                mapped_name = mapped_name[:-4] + ".mlp.fc2"

        # Try mapping with key transformation
        try:
            if weight_type:
                # For parameters with weight types
                key_without_type = mapped_name.replace(f".{weight_type}", "")
                set_recursively(hf_model, key_without_type, value, name, weight_type)
            else:
                # For buffers and other weights
                set_recursively(hf_model, mapped_name, value, name, weight_type)
            is_used = True
            used_keys.add(name)
        except (AttributeError, ValueError):
            # If mapping fails, try original key as fallback
            try:
                if weight_type:
                    key_without_type = name.replace(f".{weight_type}", "")
                    set_recursively(hf_model, key_without_type, value, name, weight_type)
                else:
                    set_recursively(hf_model, name, value, name, weight_type)
                is_used = True
                used_keys.add(name)
            except (AttributeError, ValueError) as e:
                logger.debug(f"Could not map {name}: {e}")
                pass

        if not is_used:
            unused_weights.append(name)

    # Separate and report unused weights
    quantizer_proj_unused = [w for w in unused_weights if 'quantizer' in w and 'proj' in w]
    other_unused = [w for w in unused_weights if not ('quantizer' in w and 'proj' in w)]

    if quantizer_proj_unused:
        logger.warning(f"Unused quantizer proj weights ({len(quantizer_proj_unused)}): {quantizer_proj_unused}")
    if other_unused:
        logger.warning(f"Other unused weights ({len(other_unused)}): {other_unused[:10]}")  # Show first 10
    logger.info(f"Successfully loaded {len(used_keys)} / {len(orig_dict)} weights")

    return unused_weights


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path: str,
    yaml_config_path: str,
    pytorch_dump_folder_path: str,
    repo_id: str | None,
    push_to_hub: bool = False,
):
    """
    Convert XY-Tokenizer checkpoint from MOSS-TTSD to Transformers format.

    Args:
        checkpoint_path: Path to the original checkpoint (.ckpt file).
        yaml_config_path: Path to the YAML configuration file.
        pytorch_dump_folder_path: Path to save the converted model.
        repo_id: Optional HuggingFace Hub repo ID for uploading.
        push_to_hub: Whether to push the model to the Hub.
    """
    # 1. Load configuration from YAML
    logger.info(f"Loading configuration from {yaml_config_path}")
    config = load_config_from_yaml(yaml_config_path)

    # 2. Load original checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract generator state dict if present
    if "generator" in checkpoint:
        logger.info("Extracting 'generator' state dict from checkpoint")
        original_state_dict = checkpoint["generator"]
    else:
        original_state_dict = checkpoint

    logger.info(f"Loaded checkpoint with {len(original_state_dict)} parameters")

    # 3. Create model
    logger.info("Creating XYTokenizer model")
    model = XYTokenizer(config)

    # 4. Apply weight norm, convert weights, then remove weight norm
    # This follows the pattern from xcodec2 where weight norm is applied during conversion
    # to match the checkpoint format, then removed for efficient inference
    logger.info("Applying weight norm to model for conversion")
    model.apply_weight_norm()

    # 5. Create feature extractor
    logger.info("Creating feature extractor")
    feature_extractor = XYTokenizerFeatureExtractor(
        feature_size=config.feature_extractor_config.feature_size,
        sampling_rate=config.feature_extractor_config.sampling_rate,
        chunk_length_s=config.feature_extractor_config.chunk_length,
        hop_length=config.feature_extractor_config.hop_length,
    )

    # 6. Transfer weights
    logger.info("Transferring weights to HF model")
    unused_weights = recursively_load_weights(original_state_dict, model)

    # 7. Remove weight norm after conversion for efficient inference
    logger.info("Removing weight norm from model after conversion")
    model.remove_weight_norm()

    # 8. Save model and feature extractor
    logger.info(f"Saving model to {pytorch_dump_folder_path}")
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    model.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    logger.info("Successfully saved model and feature extractor")

    # 8. Optionally push to hub
    if push_to_hub and repo_id:
        logger.info(f"Pushing to hub: {repo_id}")
        model.push_to_hub(repo_id)
        feature_extractor.push_to_hub(repo_id)
        logger.info("Successfully pushed to hub")

    return model, feature_extractor, unused_weights


def main():
    parser = argparse.ArgumentParser(
        description="Convert XY-Tokenizer checkpoints from MOSS-TTSD to Transformers format."
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to the original checkpoint (.ckpt file).",
    )
    parser.add_argument(
        "--yaml_config_path",
        required=True,
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=str,
        help="Path to save the converted model.",
    )
    parser.add_argument(
        "--repo_id",
        default=None,
        type=str,
        help="Optional HuggingFace Hub repo ID for uploading (e.g., 'username/xy-tokenizer-32k').",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the HuggingFace Hub.",
    )

    args = parser.parse_args()

    # Convert checkpoint
    model, feature_extractor, unused_weights = convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        yaml_config_path=args.yaml_config_path,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
    )

    logger.info("Conversion completed!")


if __name__ == "__main__":
    main()
