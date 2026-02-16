"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def image_size():
    """Default image size for tests."""
    return 64


@pytest.fixture
def latent_size():
    """Default latent size for tests."""
    return 8


@pytest.fixture
def dummy_batch(batch_size, image_size, device):
    """Create a dummy batch of data."""
    return {
        "pixel_values": torch.randn(batch_size, 3, image_size, image_size, device=device),
        "input_ids": torch.randint(0, 1000, (batch_size, 77), device=device),
        "attention_mask": torch.ones(batch_size, 77, device=device),
    }


@pytest.fixture
def dummy_latents(batch_size, latent_size, device):
    """Create dummy latent tensors."""
    return torch.randn(batch_size, 3, latent_size, latent_size, device=device)  # 3 channels for RGB


@pytest.fixture
def dummy_timesteps(batch_size, device):
    """Create dummy timesteps."""
    return torch.randint(0, 1000, (batch_size,), device=device)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "seed": 42,
        "data": {
            "dataset": "conceptual_captions",
            "image_size": 64,
            "batch_size": 2,
            "num_workers": 0,
            "max_train_samples": 10,
            "max_val_samples": 10,
        },
        "model": {
            "teacher": {
                "text_encoder": "openai/clip-vit-base-patch32",
                "num_inference_steps": 10,
                "scheduler_type": "ddim",
            },
            "student": {
                "text_encoder": "openai/clip-vit-base-patch32",
                "num_inference_steps": 4,
                "use_lora": False,
                "lora_rank": 4,
            },
        },
        "training": {
            "num_epochs": 2,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "mixed_precision": False,
            "scheduler_type": "cosine",
            "early_stopping_patience": 5,
            "log_interval": 10,
            "eval_interval": 20,
            "checkpoint_dir": "test_checkpoints",
        },
        "distillation": {
            "output_weight": 1.0,
            "feature_weight": 0.5,
            "kl_weight": 0.1,
            "progressive_schedule": "linear",
        },
        "curriculum": {
            "enabled": True,
            "num_regions": 3,
            "update_frequency": 10,
            "warmup_steps": 5,
        },
    }
