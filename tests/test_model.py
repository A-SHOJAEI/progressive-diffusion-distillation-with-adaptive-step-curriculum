"""Tests for model implementations."""

import pytest
import torch

from progressive_diffusion_distillation_with_adaptive_step_curriculum.models.model import (
    StudentDiffusionModel,
    TeacherDiffusionModel,
)
from progressive_diffusion_distillation_with_adaptive_step_curriculum.models.components import (
    AdaptiveCurriculumScheduler,
    ProgressiveDistillationLoss,
    TimestepDivergenceTracker,
)


class TestStudentModel:
    """Tests for student model."""

    def test_model_creation(self):
        """Test model creation."""
        model = StudentDiffusionModel(
            num_inference_steps=4,
            use_lora=False,
        )

        assert model is not None
        assert model.num_inference_steps == 4

    def test_model_forward(self, dummy_latents, dummy_timesteps, device):
        """Test model forward pass."""
        model = StudentDiffusionModel(use_lora=False)
        model.to(device)
        model.eval()

        batch_size = dummy_latents.shape[0]
        encoder_hidden_states = torch.randn(batch_size, 77, 768, device=device)

        with torch.no_grad():
            output, features = model(
                dummy_latents,
                dummy_timesteps,
                encoder_hidden_states,
                return_features=False,
            )

        assert output.shape == dummy_latents.shape

    def test_text_encoding(self, device):
        """Test text encoding."""
        model = StudentDiffusionModel(use_lora=False)
        model.to(device)

        input_ids = torch.randint(0, 1000, (2, 77), device=device)
        attention_mask = torch.ones(2, 77, device=device)

        embeddings = model.encode_text(input_ids, attention_mask)

        assert embeddings.shape == (2, 77, 768)

    def test_parameter_count(self):
        """Test parameter counting."""
        model = StudentDiffusionModel(use_lora=False)

        total_params = model.count_parameters(trainable_only=False)
        trainable_params = model.count_parameters(trainable_only=True)

        assert total_params > 0
        assert trainable_params > 0


class TestTeacherModel:
    """Tests for teacher model."""

    def test_model_creation(self):
        """Test model creation."""
        model = TeacherDiffusionModel(num_inference_steps=50)

        assert model is not None
        assert model.num_inference_steps == 50

    def test_model_forward(self, dummy_latents, dummy_timesteps, device):
        """Test model forward pass."""
        model = TeacherDiffusionModel()
        model.to(device)

        batch_size = dummy_latents.shape[0]
        encoder_hidden_states = torch.randn(batch_size, 77, 768, device=device)

        with torch.no_grad():
            output, features = model(
                dummy_latents,
                dummy_timesteps,
                encoder_hidden_states,
                return_features=False,
            )

        assert output.shape == dummy_latents.shape

    def test_model_frozen(self):
        """Test that teacher model is frozen."""
        model = TeacherDiffusionModel()

        # Check all parameters are frozen
        for param in model.parameters():
            assert not param.requires_grad


class TestTimestepDivergenceTracker:
    """Tests for timestep divergence tracker."""

    def test_tracker_creation(self):
        """Test tracker creation."""
        tracker = TimestepDivergenceTracker(num_timesteps=1000, num_regions=3)

        assert tracker.num_regions == 3
        assert len(tracker.region_divergence) == 3

    def test_get_region(self):
        """Test region assignment."""
        tracker = TimestepDivergenceTracker(num_timesteps=1000, num_regions=3)

        # Test boundary cases
        assert tracker.get_region(0) == 0
        assert tracker.get_region(333) == 1
        assert tracker.get_region(666) == 2
        assert tracker.get_region(999) == 2

    def test_update(self, device):
        """Test updating divergence statistics."""
        tracker = TimestepDivergenceTracker(num_timesteps=1000, num_regions=3)

        timesteps = torch.tensor([100, 500, 900], device=device)
        divergences = torch.tensor([0.5, 0.8, 0.3], device=device)

        tracker.update(timesteps, divergences)

        # Check that statistics were updated
        assert tracker.region_counts[0] > 0
        assert tracker.region_counts[1] > 0
        assert tracker.region_counts[2] > 0

    def test_sampling_weights(self, device):
        """Test sampling weight computation."""
        tracker = TimestepDivergenceTracker(num_timesteps=1000, num_regions=3)

        # Update with varying divergences
        for _ in range(10):
            timesteps = torch.tensor([100, 500, 900], device=device)
            divergences = torch.tensor([0.2, 1.0, 0.3], device=device)
            tracker.update(timesteps, divergences)

        weights = tracker.get_sampling_weights()

        # Weights should sum to 1
        assert abs(weights.sum() - 1.0) < 1e-6

        # Region 1 (mid) should have highest weight
        assert weights[1] > weights[0]
        assert weights[1] > weights[2]


class TestAdaptiveCurriculumScheduler:
    """Tests for adaptive curriculum scheduler."""

    def test_scheduler_creation(self):
        """Test scheduler creation."""
        scheduler = AdaptiveCurriculumScheduler(
            num_timesteps=1000,
            num_regions=3,
            warmup_steps=100,
        )

        assert scheduler.num_timesteps == 1000
        assert scheduler.warmup_steps == 100

    def test_sample_timesteps_warmup(self, device):
        """Test timestep sampling during warmup."""
        scheduler = AdaptiveCurriculumScheduler(warmup_steps=100)

        timesteps = scheduler.sample_timesteps(batch_size=4, device=device)

        assert timesteps.shape == (4,)
        assert timesteps.min() >= 0
        assert timesteps.max() < 1000

    def test_sample_timesteps_curriculum(self, device):
        """Test timestep sampling with curriculum."""
        scheduler = AdaptiveCurriculumScheduler(warmup_steps=0)

        # Force past warmup
        scheduler.step_count = 1000

        timesteps = scheduler.sample_timesteps(batch_size=4, device=device)

        assert timesteps.shape == (4,)

    def test_update_curriculum(self, device):
        """Test curriculum update."""
        scheduler = AdaptiveCurriculumScheduler(update_frequency=10)

        for step in range(20):
            timesteps = torch.randint(0, 1000, (4,), device=device)
            divergences = torch.rand(4, device=device)

            scheduler.update_curriculum(timesteps, divergences)

        # Check statistics
        stats = scheduler.get_statistics()
        assert "curriculum_step" in stats


class TestProgressiveDistillationLoss:
    """Tests for progressive distillation loss."""

    def test_loss_creation(self):
        """Test loss creation."""
        loss_fn = ProgressiveDistillationLoss(
            output_weight=1.0,
            feature_weight=0.5,
        )

        assert loss_fn.output_weight == 1.0
        assert loss_fn.feature_weight == 0.5

    def test_loss_forward(self, dummy_latents, device):
        """Test loss forward pass."""
        loss_fn = ProgressiveDistillationLoss()
        loss_fn.to(device)

        student_output = dummy_latents
        teacher_output = dummy_latents + torch.randn_like(dummy_latents) * 0.1

        total_loss, loss_dict = loss_fn(
            student_output,
            teacher_output,
            progress=0.5,
        )

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0
        assert "total_loss" in loss_dict
        assert "output_loss" in loss_dict

    def test_progressive_weighting(self, dummy_latents, device):
        """Test progressive weight schedule."""
        loss_fn = ProgressiveDistillationLoss(progressive_schedule="linear")
        loss_fn.to(device)

        # Early training
        _, loss_dict_early = loss_fn(
            dummy_latents,
            dummy_latents.clone(),
            progress=0.0,
        )

        # Late training
        _, loss_dict_late = loss_fn(
            dummy_latents,
            dummy_latents.clone(),
            progress=1.0,
        )

        # Weights should change
        assert loss_dict_early["output_weight_mult"] != loss_dict_late["output_weight_mult"]

    def test_compute_divergence(self, dummy_latents, device):
        """Test divergence computation."""
        loss_fn = ProgressiveDistillationLoss()
        loss_fn.to(device)

        student_output = dummy_latents
        teacher_output = dummy_latents + torch.randn_like(dummy_latents) * 0.1

        divergence = loss_fn.compute_divergence(student_output, teacher_output)

        assert divergence.shape == (dummy_latents.shape[0],)
        assert (divergence >= 0).all()
