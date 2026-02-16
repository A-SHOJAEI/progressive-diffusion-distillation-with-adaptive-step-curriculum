"""Configuration utilities for loading and managing experiment settings."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save.
        output_path: Path where to save the configuration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to {output_path}")


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for training.

    Args:
        device: Requested device ('cuda', 'cpu', or None for auto-detect).

    Returns:
        PyTorch device object.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_device = torch.device(device)
    logger.info(f"Using device: {torch_device}")

    if torch_device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return torch_device


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")
