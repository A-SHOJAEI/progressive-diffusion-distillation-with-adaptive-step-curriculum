"""Training loop for progressive diffusion distillation with adaptive curriculum."""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.components import AdaptiveCurriculumScheduler, ProgressiveDistillationLoss
from ..models.model import StudentDiffusionModel, TeacherDiffusionModel

logger = logging.getLogger(__name__)


class ProgressiveDistillationTrainer:
    """Trainer for progressive diffusion distillation with adaptive curriculum."""

    def __init__(
        self,
        student_model: StudentDiffusionModel,
        teacher_model: TeacherDiffusionModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.01,
        num_epochs: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
        scheduler_type: str = "cosine",
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        eval_interval: int = 1000,
        device: Optional[torch.device] = None,
        use_adaptive_curriculum: bool = True,
        curriculum_config: Optional[Dict] = None,
        distillation_config: Optional[Dict] = None,
    ):
        """Initialize progressive distillation trainer.

        Args:
            student_model: Student model to train.
            teacher_model: Teacher model (frozen).
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader (optional).
            learning_rate: Learning rate.
            weight_decay: Weight decay for AdamW.
            num_epochs: Number of training epochs.
            gradient_accumulation_steps: Number of steps to accumulate gradients.
            max_grad_norm: Maximum gradient norm for clipping.
            mixed_precision: Whether to use mixed precision training.
            scheduler_type: LR scheduler type ('cosine' or 'plateau').
            early_stopping_patience: Patience for early stopping.
            checkpoint_dir: Directory to save checkpoints.
            log_interval: Logging interval (in steps).
            eval_interval: Evaluation interval (in steps).
            device: Training device.
            use_adaptive_curriculum: Whether to use adaptive curriculum.
            curriculum_config: Configuration for adaptive curriculum.
            distillation_config: Configuration for distillation loss.
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student_model.to(self.device)
        self.teacher_model.to(self.device)

        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        # Optimizer
        self.optimizer = AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=learning_rate * 0.01,
            )
            self.scheduler_type = "step"
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True,
            )
            self.scheduler_type = "plateau"

        # Mixed precision
        self.scaler = GradScaler('cuda') if mixed_precision else None

        # Distillation loss
        if distillation_config is None:
            distillation_config = {}

        self.distillation_loss = ProgressiveDistillationLoss(
            output_weight=distillation_config.get("output_weight", 1.0),
            feature_weight=distillation_config.get("feature_weight", 0.5),
            kl_weight=distillation_config.get("kl_weight", 0.1),
            progressive_schedule=distillation_config.get("progressive_schedule", "linear"),
        ).to(self.device)

        # Adaptive curriculum
        self.use_adaptive_curriculum = use_adaptive_curriculum
        if use_adaptive_curriculum:
            if curriculum_config is None:
                curriculum_config = {}

            self.curriculum = AdaptiveCurriculumScheduler(
                num_timesteps=1000,
                num_regions=curriculum_config.get("num_regions", 3),
                update_frequency=curriculum_config.get("update_frequency", 500),
                warmup_steps=curriculum_config.get("warmup_steps", 1000),
            )
        else:
            self.curriculum = None

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Student parameters: {student_model.count_parameters(trainable_only=True):,}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Batch of data.

        Returns:
            Dictionary of loss values.
        """
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        batch_size = pixel_values.shape[0]

        # Encode text
        encoder_hidden_states = self.student_model.encode_text(input_ids, attention_mask)

        # Sample timesteps
        if self.use_adaptive_curriculum:
            timesteps = self.curriculum.sample_timesteps(batch_size, self.device)
        else:
            timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)

        # Add noise to clean images (simulated latents)
        # In production, this would use a VAE encoder
        latents = F.interpolate(pixel_values, size=(64, 64), mode="bilinear")

        # Sample noise
        noise = torch.randn_like(latents)

        # Add noise according to timesteps
        noisy_latents = self._add_noise(latents, noise, timesteps)

        # Forward pass with mixed precision
        with autocast('cuda', enabled=self.mixed_precision):
            # Student prediction
            student_pred, student_features = self.student_model(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                return_features=True,
            )

            # Teacher prediction
            with torch.no_grad():
                teacher_pred, teacher_features = self.teacher_model(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    return_features=True,
                )

            # Compute training progress
            progress = self.global_step / (len(self.train_dataloader) * self.num_epochs / self.gradient_accumulation_steps)
            progress = min(1.0, progress)

            # Compute distillation loss
            loss, loss_dict = self.distillation_loss(
                student_pred,
                teacher_pred,
                student_features,
                teacher_features,
                progress=progress,
            )

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update adaptive curriculum
        if self.use_adaptive_curriculum:
            with torch.no_grad():
                divergences = self.distillation_loss.compute_divergence(
                    student_pred,
                    teacher_pred,
                )
                self.curriculum.update_curriculum(timesteps, divergences)

        return loss_dict

    def _add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to latents according to diffusion schedule.

        Args:
            latents: Clean latents.
            noise: Noise to add.
            timesteps: Timestep for each sample.

        Returns:
            Noisy latents.
        """
        # Use student's scheduler
        # Move timesteps to CPU for indexing, then move result back to device
        sqrt_alpha_prod = self.student_model.scheduler.alphas_cumprod[timesteps.cpu()] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.student_model.scheduler.alphas_cumprod[timesteps.cpu()]) ** 0.5

        # Move back to the device of the latents
        sqrt_alpha_prod = sqrt_alpha_prod.to(latents.device).flatten()
        while len(sqrt_alpha_prod.shape) < len(latents.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(latents.device).flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(latents.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
        return noisy_latents

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.student_model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            loss_dict = self.train_step(batch)

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.max_grad_norm,
                )

                # Optimizer step
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update learning rate
                if self.scheduler_type == "step":
                    self.scheduler.step()

                self.global_step += 1

                # Logging
                if self.global_step % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    log_str = f"Step {self.global_step} | Loss: {loss_dict['total_loss']:.4f} | LR: {lr:.2e}"

                    if self.use_adaptive_curriculum:
                        curriculum_stats = self.curriculum.get_statistics()
                        log_str += f" | Curriculum: {curriculum_stats.get('curriculum_step', 0)}"

                    logger.info(log_str)

                    # Try to log to MLflow
                    try:
                        import mlflow
                        mlflow.log_metrics({
                            "train/loss": loss_dict["total_loss"],
                            "train/output_loss": loss_dict["output_loss"],
                            "train/feature_loss": loss_dict["feature_loss"],
                            "train/learning_rate": lr,
                        }, step=self.global_step)

                        if self.use_adaptive_curriculum:
                            mlflow.log_metrics(curriculum_stats, step=self.global_step)
                    except Exception:
                        pass

                # Validation
                if self.val_dataloader is not None and self.global_step % self.eval_interval == 0:
                    val_loss = self.validate()
                    logger.info(f"Validation loss: {val_loss:.4f}")

                    try:
                        import mlflow
                        mlflow.log_metric("val/loss", val_loss, step=self.global_step)
                    except Exception:
                        pass

                    # Early stopping check
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint(is_best=True)
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        return epoch_loss / max(num_batches, 1)

                    # Update scheduler on plateau
                    if self.scheduler_type == "plateau":
                        self.scheduler.step(val_loss)

            epoch_loss += loss_dict["total_loss"]
            num_batches += 1

            progress_bar.set_postfix({"loss": loss_dict["total_loss"]})

        return epoch_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation.

        Returns:
            Average validation loss.
        """
        self.student_model.eval()
        val_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            batch_size = pixel_values.shape[0]

            # Encode text
            encoder_hidden_states = self.student_model.encode_text(input_ids, attention_mask)

            # Random timesteps for validation
            timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)

            # Prepare latents
            latents = F.interpolate(pixel_values, size=(64, 64), mode="bilinear")
            noise = torch.randn_like(latents)
            noisy_latents = self._add_noise(latents, noise, timesteps)

            # Forward pass
            student_pred, _ = self.student_model(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            )

            teacher_pred, _ = self.teacher_model(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            )

            # Compute loss
            loss, _ = self.distillation_loss(student_pred, teacher_pred, progress=0.5)

            val_loss += loss.item()
            num_batches += 1

        self.student_model.train()
        return val_loss / max(num_batches, 1)

    def train(self) -> None:
        """Full training loop."""
        logger.info("Starting training...")
        start_time = time.time()

        try:
            for epoch in range(self.num_epochs):
                self.epoch = epoch
                epoch_loss = self.train_epoch()

                logger.info(f"Epoch {epoch} completed | Average loss: {epoch_loss:.4f}")

                self.training_history["train_loss"].append(epoch_loss)
                self.training_history["learning_rate"].append(
                    self.optimizer.param_groups[0]["lr"]
                )

                # Save regular checkpoint
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(is_best=False)

                # Early stopping check
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 3600:.2f} hours")

        # Save final checkpoint
        self.save_checkpoint(is_best=False, name="final")

    def save_checkpoint(self, is_best: bool = False, name: Optional[str] = None) -> None:
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far.
            name: Optional name for checkpoint.
        """
        if name is None:
            name = f"checkpoint_step_{self.global_step}"

        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.student_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint.get("training_history", self.training_history)

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
