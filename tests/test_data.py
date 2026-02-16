"""Tests for data loading and preprocessing."""

import pytest
import torch
from PIL import Image

from progressive_diffusion_distillation_with_adaptive_step_curriculum.data.loader import (
    ConceptualCaptionsDataset,
    get_dataloader,
)
from progressive_diffusion_distillation_with_adaptive_step_curriculum.data.preprocessing import (
    preprocess_image,
    preprocess_text,
    ImageAugmentation,
)
from transformers import CLIPTokenizer


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create test image
        image = Image.new("RGB", (256, 256), color=(128, 128, 128))

        # Preprocess
        processed = preprocess_image(image, size=64, center_crop=True, normalize=True)

        # Check output
        assert processed.shape == (3, 64, 64)
        assert processed.dtype == torch.float32
        assert -1 <= processed.min() <= processed.max() <= 1

    def test_preprocess_image_grayscale(self):
        """Test preprocessing grayscale image."""
        # Create grayscale image
        image = Image.new("L", (256, 256), color=128)

        # Preprocess (should convert to RGB)
        processed = preprocess_image(image, size=64)

        assert processed.shape == (3, 64, 64)

    def test_preprocess_text(self):
        """Test text preprocessing."""
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        text = "a beautiful sunset over the ocean"

        input_ids, attention_mask = preprocess_text(text, tokenizer)

        # Check output
        assert input_ids.shape == (77,)
        assert attention_mask.shape == (77,)
        assert input_ids.dtype == torch.long
        assert attention_mask.dtype == torch.long

    def test_preprocess_text_empty(self):
        """Test preprocessing empty text."""
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        input_ids, attention_mask = preprocess_text("", tokenizer)

        assert input_ids.shape == (77,)
        assert attention_mask.shape == (77,)

    def test_image_augmentation(self):
        """Test image augmentation pipeline."""
        augmentation = ImageAugmentation(size=64, horizontal_flip_prob=0.5, color_jitter=True)

        image = Image.new("RGB", (256, 256), color=(128, 128, 128))

        augmented = augmentation(image)

        assert augmented.shape == (3, 64, 64)
        assert augmented.dtype == torch.float32


class TestDataset:
    """Tests for dataset class."""

    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = ConceptualCaptionsDataset(
            split="train",
            image_size=64,
            max_samples=10,
            augment=False,
        )

        assert len(dataset) > 0

    def test_dataset_getitem(self):
        """Test getting item from dataset."""
        dataset = ConceptualCaptionsDataset(
            split="train",
            image_size=64,
            max_samples=10,
            augment=False,
        )

        sample = dataset[0]

        assert "pixel_values" in sample
        assert "input_ids" in sample
        assert "attention_mask" in sample

        assert sample["pixel_values"].shape == (3, 64, 64)
        assert sample["input_ids"].shape == (77,)
        assert sample["attention_mask"].shape == (77,)

    def test_dataset_length(self):
        """Test dataset length."""
        max_samples = 5

        dataset = ConceptualCaptionsDataset(
            split="train",
            image_size=64,
            max_samples=max_samples,
            augment=False,
        )

        assert len(dataset) == max_samples


class TestDataLoader:
    """Tests for dataloader."""

    def test_dataloader_creation(self):
        """Test dataloader creation."""
        dataloader = get_dataloader(
            split="train",
            batch_size=2,
            image_size=64,
            max_samples=10,
            num_workers=0,
            shuffle=True,
        )

        assert len(dataloader) > 0

    def test_dataloader_iteration(self):
        """Test iterating through dataloader."""
        dataloader = get_dataloader(
            split="train",
            batch_size=2,
            image_size=64,
            max_samples=10,
            num_workers=0,
            shuffle=False,
        )

        batch = next(iter(dataloader))

        assert "pixel_values" in batch
        assert "input_ids" in batch
        assert "attention_mask" in batch

        assert batch["pixel_values"].shape[0] == 2
        assert batch["pixel_values"].shape[1:] == (3, 64, 64)

    def test_dataloader_batch_size(self):
        """Test dataloader batch size."""
        batch_size = 4

        dataloader = get_dataloader(
            split="train",
            batch_size=batch_size,
            image_size=64,
            max_samples=10,
            num_workers=0,
        )

        batch = next(iter(dataloader))

        assert batch["pixel_values"].shape[0] == batch_size
