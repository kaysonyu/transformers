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
"""Convert XY Tokenizer checkpoints from original format to HuggingFace format."""

import argparse
import json

import torch

from transformers import XYTokenizerConfig, XYTokenizerFeatureExtractor, XYTokenizerModel, logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def apply_weight_norm(model):
    """
    Apply weight normalization to the model's quantizer projection layers.

    This function is called before loading weights from the original checkpoint,
    which have weight norm applied (weight_g, weight_v parameters).
    """
    weight_norm = torch.nn.utils.weight_norm
    if hasattr(torch.nn.utils.parametrizations, "weight_norm"):
        weight_norm = torch.nn.utils.parametrizations.weight_norm

    # Apply to ResidualVQ input_proj and output_proj
    if hasattr(model.quantizer, "input_proj") and not isinstance(model.quantizer.input_proj, torch.nn.Identity):
        weight_norm(model.quantizer.input_proj, name="weight")

    if hasattr(model.quantizer, "output_proj") and not isinstance(model.quantizer.output_proj, torch.nn.Identity):
        weight_norm(model.quantizer.output_proj, name="weight")

    logger.info("Weight normalization applied to quantizer projection layers")


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path,
    config_path,
    pytorch_dump_folder_path,
    repo_id=None,
):
    """
    Convert XY Tokenizer checkpoint from original format to HuggingFace format.

    Args:
        checkpoint_path: Path to original checkpoint file (.ckpt or .pth)
        config_path: Path to original config file (JSON)
        pytorch_dump_folder_path: Path to save the converted model
        repo_id: Optional HuggingFace repo ID for uploading
    """
    # 1. Load original config
    logger.info(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        original_config = json.load(f)

    # 2. Convert config to HF format
    # Extract params and pass to XYTokenizerConfig
    # The config class will handle migration via _migrate_from_params
    params = original_config.get("params", {})
    config_dict = {
        "input_sampling_rate": original_config.get(
            "input_sample_rate", original_config.get("input_sampling_rate", 16000)
        ),
        "sampling_rate": original_config.get("output_sample_rate", original_config.get("sampling_rate", 16000)),
        "encoder_downsample_rate": original_config.get("encoder_downsample_rate", 1280),
        "decoder_upsample_rate": original_config.get("decoder_upsample_rate", 1920),
    }
    if params:
        config_dict["params"] = params

    logger.info("Creating XYTokenizerConfig...")
    config = XYTokenizerConfig(**config_dict)

    # 3. Initialize HF model
    logger.info("Initializing XYTokenizer model...")
    model = XYTokenizerModel(config)

    # 4. Load original checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle both direct checkpoint and nested state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        original_state_dict = checkpoint["state_dict"]
    else:
        original_state_dict = checkpoint

    # 5. Convert checkpoint keys from original to HF format
    logger.info("Converting checkpoint keys...")
    converted_state_dict = {}
    for key, value in original_state_dict.items():
        new_key = key
        # Add xy_tokenizer prefix if not present
        if not key.startswith("xy_tokenizer."):
            new_key = f"xy_tokenizer.{key}"
        converted_state_dict[new_key] = value

    # 6. Apply weight norm before loading (CRITICAL)
    # Original checkpoint has weight norm (weight_g, weight_v keys)
    apply_weight_norm(model)

    # 7. Load weights with strict checking
    logger.info("Loading weights...")
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=True)

    # Log any issues (should not happen with strict=True)
    if missing_keys:
        raise ValueError(f"Missing keys in converted checkpoint: {missing_keys}")
    if unexpected_keys:
        raise ValueError(f"Unexpected keys in converted checkpoint: {unexpected_keys}")

    # 8. Remove weight norm for inference efficiency
    logger.info("Removing weight normalization...")
    model.remove_weight_norm()

    # 9. Save model and config
    logger.info(f"Saving model to {pytorch_dump_folder_path}...")
    model.save_pretrained(pytorch_dump_folder_path)
    config.save_pretrained(pytorch_dump_folder_path)

    # 10. Create and save feature extractor
    logger.info("Creating feature extractor...")
    feature_extractor = XYTokenizerFeatureExtractor(
        sampling_rate=config.input_sampling_rate,
        feature_size=config.feature_extractor_config.feature_size,
        hop_length=config.feature_extractor_config.hop_length,
        chunk_length=config.feature_extractor_config.chunk_length,
        n_fft=config.feature_extractor_config.n_fft,
        padding_side=config.feature_extractor_config.padding_side,
    )
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    logger.info(f"Model successfully converted and saved to {pytorch_dump_folder_path}")

    # 11. Optional: Push to hub
    if repo_id:
        logger.info(f"Pushing to hub: {repo_id}...")
        feature_extractor.push_to_hub(repo_id)
        config.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
        logger.info(f"Model successfully pushed to {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert XY Tokenizer checkpoints from original format to HuggingFace Transformers format"
    )

    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to original checkpoint file (.ckpt or .pth)",
    )
    parser.add_argument(
        "--config_path",
        required=True,
        type=str,
        help="Path to original config file (JSON)",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        type=str,
        help="Optional HuggingFace repo ID (e.g., 'username/xy-tokenizer-model') to upload the converted model.",
    )

    args = parser.parse_args()

    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        repo_id=args.push_to_hub,
    )
