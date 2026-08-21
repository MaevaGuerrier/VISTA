#!/usr/bin/env python3
"""
Convert VISTA model to ONNX format.

Usage:
    python convert_vista_to_onnx.py --config config/vista.yaml --checkpoint_path checkpoints/model/latest.pth --output model.onnx
"""

import argparse
import os
import re

import torch
import yaml
from torch import nn
from vint_train.models.vint.vint_dino import ViNTWithDINOTokens


def resolve_env_vars(obj):
    """Resolve environment variables in config."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = resolve_env_vars(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = resolve_env_vars(item)
    elif isinstance(obj, str):
        pattern = re.compile(r"\$\{(\w+)}")

        def replace(match):
            env_var = match.group(1)
            return os.getenv(env_var, match.group(0))

        return pattern.sub(replace, obj)
    return obj


def load_config(config_path: str) -> dict:
    """Load and resolve config from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return resolve_env_vars(config)


def average_weights(state_dicts):
    """Average weights from multiple state dicts."""
    avg_state_dict = {}
    num_models = len(state_dicts)

    assert all([state_dicts[0].keys() == sd.keys() for sd in state_dicts]), (
        "All state dicts must have the same keys"
    )

    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / num_models

    return avg_state_dict


# TODO use the utils
def load_vint_dino_model(
    config: dict, checkpoint_path: str | list, device: torch.device
) -> nn.Module:
    """Load ViNT-DINO model from config and checkpoint."""
    # Create model
    model = ViNTWithDINOTokens(
        image_size=config["image_size"],
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        encoding_size=config["obs_encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        output_layers=config["output_layers"],
        positional_encoding_type=config.get("positional_encoding_type", "peg"),
        separate_tokens_and_heads=config.get("separate_tokens_and_heads", False),
        take_action_history=config.get("take_action_history", False),
        action_enc_layers=config.get("action_enc_layers", [256]),
    )

    # Load checkpoint
    if isinstance(checkpoint_path, str):
        checkpoint_path = [checkpoint_path]
    checkpoints = []
    for path in checkpoint_path:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Extract model state dict from checkpoint
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                loaded_model = checkpoint["model"]
                # Handle DataParallel wrapper
                if hasattr(loaded_model, "module"):
                    state_dict = loaded_model.module.state_dict()
                elif hasattr(loaded_model, "state_dict"):
                    state_dict = loaded_model.state_dict()
                else:
                    state_dict = loaded_model
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        checkpoints.append(state_dict)

    if len(checkpoints) > 1:
        print(f"Averaging weights from {len(checkpoints)} checkpoints")
        state_dict = average_weights(checkpoints)
    else:
        state_dict = checkpoints[0]

    # Clone everything just in case
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].clone()

    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def convert_to_onnx(
    model: nn.Module,
    config: dict,
    output_path: str,
    device: torch.device,
    opset_version: int = 14,
) -> None:
    """Convert model to ONNX format."""
    context_size = config["context_size"]
    image_size = config["image_size"]  # [width, height]
    batch_size = 2

    # Create dummy inputs
    # obs_img: (B, 3*(context_size+1), H, W) - stacked context + current obs
    # First 3 channels are current observation, then context_size previous obs
    obs_img = torch.randn(
        batch_size,
        3 * (context_size + 1),
        image_size[1],  # height
        image_size[0],  # width
        device=device,
    )

    # goal_img: (B, 3, H, W)
    goal_img = torch.randn(
        batch_size,
        3,
        image_size[1],  # height
        image_size[0],  # width
        device=device,
    )

    # Export to ONNX
    torch.onnx.export(
        model,
        (obs_img, goal_img),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["obs_img", "goal_img"],
        output_names=["distance", "action"],
        dynamic_axes={
            "obs_img": {0: "batch_size", 2: "height", 3: "width"},
            "goal_img": {0: "batch_size", 2: "height", 3: "width"},
            "distance": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"Model exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ViNT-DINO model to ONNX format"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file (e.g., config/vint_dino.yaml)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the directory containing model checkpoints",
    )
    parser.add_argument(
        "--weights_to_average",
        type=int,
        default=1,
        help="Number of latest checkpoints to average (default: 1, i.e., no averaging)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vista.onnx",
        help="Output ONNX file path (default: vint_dino_metric.onnx)",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for loading model (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Load model
    # print(f"Loading {args.weights_to_average} checkpoint(s) from: {args.checkpoint_dir}")
    # checkpoint_paths = [
    #     os.path.join(args.checkpoint_dir, f"{config['epochs'] - i - 1}.pth") for i in range(args.weights_to_average)
    # ]

    checkpoint_paths = [os.path.join(args.checkpoint_dir, "vista.pth")]
    model = load_vint_dino_model(config, checkpoint_paths, device)
    print("Model loaded successfully")

    # Convert to ONNX
    print("Converting model to ONNX format...")
    convert_to_onnx(model, config, args.output, device, args.opset_version)
    print("Conversion complete!")

    # Print model info
    print("\nModel Information:")
    print(
        f"  - Input: obs_img (B, {3 * (config['context_size'] + 1)}, {config['image_size'][1]}, {config['image_size'][0]})"
    )
    print(
        f"           goal_img (B, 3, {config['image_size'][1]}, {config['image_size'][0]})"
    )
    print("  - Output: distance (B, 1)")
    print(
        f"            action (B, {config['len_traj_pred']}, {3 if config['learn_angle'] else 2})"
    )
    print(f"  - Learn angle: {config['learn_angle']}")


if __name__ == "__main__":
    main()
