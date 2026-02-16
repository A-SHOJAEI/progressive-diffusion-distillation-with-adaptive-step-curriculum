"""Tests for training loop."""

import pytest
import torch
from pathlib import Path

from progressive_diffusion_distillation_with_adaptive_step_curriculum.training.trainer import (
    ProgressiveDistillationTrainer,
)
from progressive_diffusion_distillation_with_adaptive_step_curriculum.models.model import (
    StudentDiffusionModel,
    TeacherDiffusionModel,
)
from progressive_diffusion_distillation_with_adaptive_step_curriculum.data.loader import get_dataloader


class TestTrainer:
    """Tests for training loop."""

    @pytest.fixture
    def student_model(self):
        """Create student model for testing."""
        return StudentDiffusionModel(num_inference_steps=4, use_lora=False)

    @pytest.fixture
    def teacher_model(self):
        """Create teacher model for testing."""
        return TeacherDiffusionModel(num_inference_steps=10)

    @pytest.fixture
    def train_dataloader(self):
        """Create training dataloader."""
        return get_dataloader(
            split="train",
            batch_size=2,
            image_size=64,
            max_samples=10,
            num_workers=0,
            shuffle=True,
        )

    @pytest.fixture
    def val_dataloader(self):
        """Create validation dataloader."""
        return get_dataloader(
            split="train",
            batch_size=2,
            image_size=64,
            max_samples=10,
            num_workers=0,
            shuffle=False,
        )

    def test_trainer_creation(
        self,
        student_model,
        teacher_model,
        train_dataloader,
        val_dataloader,
        device,
        tmp_path,
    ):
        """Test trainer creation."""
        trainer = ProgressiveDistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=2,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device=device,
            use_adaptive_curriculum=True,
        )

        assert trainer is not None
        assert trainer.num_epochs == 2

    def test_train_step(
        self,
        student_model,
        teacher_model,
        train_dataloader,
        device,
        tmp_path,
    ):
        """Test single training step."""
        trainer = ProgressiveDistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataloader=train_dataloader,
            num_epochs=1,
            gradient_accumulation_steps=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device=device,
            use_adaptive_curriculum=False,
        )

        batch = next(iter(train_dataloader))
        loss_dict = trainer.train_step(batch)

        assert "total_loss" in loss_dict
        assert loss_dict["total_loss"] >= 0

    def test_validate(
        self,
        student_model,
        teacher_model,
        train_dataloader,
        val_dataloader,
        device,
        tmp_path,
    ):
        """Test validation."""
        trainer = ProgressiveDistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device=device,
        )

        val_loss = trainer.validate()

        assert isinstance(val_loss, float)
        assert val_loss >= 0

    def test_checkpoint_save_load(
        self,
        student_model,
        teacher_model,
        train_dataloader,
        device,
        tmp_path,
    ):
        """Test checkpoint saving and loading."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        trainer = ProgressiveDistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataloader=train_dataloader,
            num_epochs=1,
            checkpoint_dir=str(checkpoint_dir),
            device=device,
        )

        # Save checkpoint
        trainer.global_step = 100
        trainer.save_checkpoint(name="test")

        checkpoint_path = checkpoint_dir / "test.pt"
        assert checkpoint_path.exists()

        # Load checkpoint
        new_trainer = ProgressiveDistillationTrainer(
            student_model=StudentDiffusionModel(use_lora=False),
            teacher_model=teacher_model,
            train_dataloader=train_dataloader,
            num_epochs=1,
            checkpoint_dir=str(checkpoint_dir),
            device=device,
        )

        new_trainer.load_checkpoint(str(checkpoint_path))

        assert new_trainer.global_step == 100

    def test_adaptive_curriculum_enabled(
        self,
        student_model,
        teacher_model,
        train_dataloader,
        device,
        tmp_path,
    ):
        """Test trainer with adaptive curriculum enabled."""
        trainer = ProgressiveDistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataloader=train_dataloader,
            num_epochs=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device=device,
            use_adaptive_curriculum=True,
            curriculum_config={
                "num_regions": 3,
                "update_frequency": 10,
                "warmup_steps": 5,
            },
        )

        assert trainer.use_adaptive_curriculum
        assert trainer.curriculum is not None

    def test_adaptive_curriculum_disabled(
        self,
        student_model,
        teacher_model,
        train_dataloader,
        device,
        tmp_path,
    ):
        """Test trainer with adaptive curriculum disabled."""
        trainer = ProgressiveDistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataloader=train_dataloader,
            num_epochs=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device=device,
            use_adaptive_curriculum=False,
        )

        assert not trainer.use_adaptive_curriculum
        assert trainer.curriculum is None
