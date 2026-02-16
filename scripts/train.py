#!/usr/bin/env python
"""Training script for progressive diffusion distillation."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from progressive_diffusion_distillation_with_adaptive_step_curriculum.data.loader import get_dataloader
from progressive_diffusion_distillation_with_adaptive_step_curriculum.models.model import (
    StudentDiffusionModel,
    TeacherDiffusionModel,
)
from progressive_diffusion_distillation_with_adaptive_step_curriculum.training.trainer import (
    ProgressiveDistillationTrainer,
)
from progressive_diffusion_distillation_with_adaptive_step_curriculum.utils.config import (
    load_config,
    save_config,
    get_device,
    set_seed,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train progressive diffusion distillation model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="progressive_distillation",
        help="MLflow experiment name",
    )

    return parser.parse_args()


def setup_mlflow(experiment_name: str, config: dict) -> None:
    """Setup MLflow tracking.

    Args:
        experiment_name: Name of the experiment.
        config: Configuration dictionary.
    """
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

        # Log parameters
        mlflow.log_params({
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["data"]["batch_size"],
            "num_epochs": config["training"]["num_epochs"],
            "use_adaptive_curriculum": config["curriculum"]["enabled"],
            "student_inference_steps": config["model"]["student"]["num_inference_steps"],
            "teacher_inference_steps": config["model"]["teacher"]["num_inference_steps"],
        })

        logger.info(f"MLflow tracking initialized: experiment={experiment_name}")
    except Exception as e:
        logger.warning(f"MLflow not available or failed to initialize: {e}")


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    set_seed(config["seed"])

    # Get device
    device = get_device(args.device)

    # Setup MLflow
    setup_mlflow(args.experiment_name, config)

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, results_dir / "config.yaml")

    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_dataloader = get_dataloader(
            split=config["data"]["split_train"],
            batch_size=config["data"]["batch_size"],
            image_size=config["data"]["image_size"],
            max_samples=config["data"]["max_train_samples"],
            num_workers=config["data"]["num_workers"],
            shuffle=True,
            cache_dir=config["data"].get("cache_dir"),
        )

        val_dataloader = get_dataloader(
            split=config["data"]["split_val"],
            batch_size=config["data"]["batch_size"],
            image_size=config["data"]["image_size"],
            max_samples=config["data"]["max_val_samples"],
            num_workers=config["data"]["num_workers"],
            shuffle=False,
            cache_dir=config["data"].get("cache_dir"),
        )

        logger.info(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

        # Create models
        logger.info("Creating teacher model...")
        teacher_model = TeacherDiffusionModel(
            text_encoder_name=config["model"]["teacher"]["text_encoder"],
            num_inference_steps=config["model"]["teacher"]["num_inference_steps"],
            scheduler_type=config["model"]["teacher"]["scheduler_type"],
        )

        logger.info("Creating student model...")
        student_model = StudentDiffusionModel(
            text_encoder_name=config["model"]["student"]["text_encoder"],
            num_inference_steps=config["model"]["student"]["num_inference_steps"],
            use_lora=config["model"]["student"]["use_lora"],
            lora_rank=config["model"]["student"]["lora_rank"],
        )

        logger.info(
            f"Teacher parameters: {teacher_model.count_parameters():,}, "
            f"Student parameters: {student_model.count_parameters():,}"
        )

        # Create trainer
        logger.info("Creating trainer...")
        trainer = ProgressiveDistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            num_epochs=config["training"]["num_epochs"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            max_grad_norm=config["training"]["max_grad_norm"],
            mixed_precision=config["training"]["mixed_precision"],
            scheduler_type=config["training"]["scheduler_type"],
            early_stopping_patience=config["training"]["early_stopping_patience"],
            checkpoint_dir=config["training"]["checkpoint_dir"],
            log_interval=config["training"]["log_interval"],
            eval_interval=config["training"]["eval_interval"],
            device=device,
            use_adaptive_curriculum=config["curriculum"]["enabled"],
            curriculum_config=config["curriculum"],
            distillation_config=config["distillation"],
        )

        # Resume from checkpoint if specified
        if args.resume is not None:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Train model
        logger.info("Starting training...")
        trainer.train()

        logger.info("Training completed successfully!")

        # Save final metrics
        final_metrics = {
            "best_val_loss": trainer.best_val_loss,
            "final_epoch": trainer.epoch,
            "total_steps": trainer.global_step,
        }

        import json
        with open(results_dir / "training_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

        logger.info(f"Final validation loss: {trainer.best_val_loss:.4f}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Close MLflow run
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
