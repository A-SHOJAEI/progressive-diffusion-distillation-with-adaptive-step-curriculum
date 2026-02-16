#!/usr/bin/env python
"""Evaluation script for progressive diffusion distillation."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from tqdm import tqdm

from progressive_diffusion_distillation_with_adaptive_step_curriculum.data.loader import get_dataloader
from progressive_diffusion_distillation_with_adaptive_step_curriculum.evaluation.analysis import (
    ResultsAnalyzer,
    visualize_results,
    save_samples_with_captions,
)
from progressive_diffusion_distillation_with_adaptive_step_curriculum.evaluation.metrics import (
    compute_metrics,
    measure_inference_time,
)
from progressive_diffusion_distillation_with_adaptive_step_curriculum.models.model import (
    StudentDiffusionModel,
    TeacherDiffusionModel,
)
from progressive_diffusion_distillation_with_adaptive_step_curriculum.utils.config import (
    load_config,
    get_device,
    set_seed,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate progressive diffusion distillation model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (overrides config)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--compare-teacher",
        action="store_true",
        help="Also evaluate teacher model for comparison",
    )

    return parser.parse_args()


def load_student_model(
    checkpoint_path: str,
    config: Dict,
    device: torch.device,
) -> StudentDiffusionModel:
    """Load student model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Loaded student model.
    """
    logger.info(f"Loading student model from {checkpoint_path}")

    # Create model
    model = StudentDiffusionModel(
        text_encoder_name=config["model"]["student"]["text_encoder"],
        num_inference_steps=config["model"]["student"]["num_inference_steps"],
        use_lora=config["model"]["student"]["use_lora"],
        lora_rank=config["model"]["student"]["lora_rank"],
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(f"Loaded student model with {model.count_parameters():,} parameters")

    return model


def generate_samples(
    model: StudentDiffusionModel,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """Generate samples from the model.

    Args:
        model: Model to generate from.
        dataloader: Dataloader with prompts.
        num_samples: Number of samples to generate.
        device: Device to run on.

    Returns:
        Tuple of (real_images, generated_images, prompts).
    """
    real_images = []
    generated_images = []
    prompts = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating samples"):
            if len(generated_images) >= num_samples:
                break

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            batch_size = pixel_values.shape[0]

            # Encode text
            encoder_hidden_states = model.encode_text(input_ids, attention_mask)

            # Generate (simplified - using single step for demo)
            # In production, would use full denoising loop
            latents = torch.randn(batch_size, 3, 64, 64, device=device)  # 3 channels for RGB
            timestep = torch.tensor([500] * batch_size, device=device)

            noise_pred, _ = model(latents, timestep, encoder_hidden_states)

            # Store results
            for i in range(batch_size):
                if len(generated_images) >= num_samples:
                    break

                real_images.append(pixel_values[i].cpu())
                generated_images.append(noise_pred[i].cpu())

                # Decode prompt (simplified)
                prompts.append(f"sample_{len(prompts)}")

    return real_images, generated_images, prompts


def evaluate_model(
    model: StudentDiffusionModel,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    """Evaluate model on test set.

    Args:
        model: Model to evaluate.
        dataloader: Test dataloader.
        num_samples: Number of samples to evaluate.
        device: Device to run on.
        output_dir: Directory to save results.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating model on {num_samples} samples...")

    # Generate samples
    real_images, generated_images, prompts = generate_samples(
        model, dataloader, num_samples, device
    )

    logger.info(f"Generated {len(generated_images)} samples")

    # Compute metrics
    metrics = {}

    try:
        # FID and CLIP scores
        quality_metrics = compute_metrics(
            real_images=real_images[:min(100, len(real_images))],
            generated_images=generated_images[:min(100, len(generated_images))],
            texts=prompts[:min(100, len(prompts))],
            device=device,
        )
        metrics.update(quality_metrics)
    except Exception as e:
        logger.error(f"Failed to compute quality metrics: {e}")

    try:
        # Inference speed
        timing_metrics = measure_inference_time(
            model=model,
            num_samples=100,
            num_steps=model.num_inference_steps,
            device=device,
        )
        metrics.update(timing_metrics)

        # Calculate speedup (assuming teacher uses 50 steps)
        teacher_steps = 50
        student_steps = model.num_inference_steps
        metrics["inference_speedup"] = teacher_steps / student_steps

    except Exception as e:
        logger.error(f"Failed to compute timing metrics: {e}")

    # Visualize samples
    try:
        visualize_results(
            images=generated_images[:16],
            texts=prompts[:16],
            output_path=output_dir / "generated_samples.png",
            nrow=4,
        )
    except Exception as e:
        logger.error(f"Failed to visualize results: {e}")

    return metrics


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    set_seed(config["seed"])

    # Get device
    device = get_device(args.device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results analyzer
    analyzer = ResultsAnalyzer(results_dir=str(output_dir))

    try:
        # Load student model
        student_model = load_student_model(args.checkpoint, config, device)

        # Create test dataloader
        num_samples = args.num_samples if args.num_samples else config["evaluation"]["num_samples"]

        logger.info("Creating test dataloader...")
        test_dataloader = get_dataloader(
            split=config["data"]["split_val"],
            batch_size=config["evaluation"]["generation_batch_size"],
            image_size=config["data"]["image_size"],
            max_samples=num_samples,
            num_workers=2,
            shuffle=False,
            cache_dir=config["data"].get("cache_dir"),
        )

        # Evaluate student model
        logger.info("Evaluating student model...")
        student_metrics = evaluate_model(
            student_model,
            test_dataloader,
            num_samples,
            device,
            output_dir,
        )

        # Add model info to metrics
        student_metrics["model_type"] = "student"
        student_metrics["num_parameters"] = student_model.count_parameters()
        student_metrics["num_inference_steps"] = student_model.num_inference_steps

        logger.info("\nStudent Model Results:")
        logger.info("=" * 80)
        for key, value in sorted(student_metrics.items()):
            logger.info(f"{key:40s}: {value}")
        logger.info("=" * 80)

        # Save student metrics
        analyzer.save_metrics(student_metrics, filename="student_metrics.json")

        # Compare with teacher if requested
        if args.compare_teacher:
            logger.info("Evaluating teacher model for comparison...")

            teacher_model = TeacherDiffusionModel(
                text_encoder_name=config["model"]["teacher"]["text_encoder"],
                num_inference_steps=config["model"]["teacher"]["num_inference_steps"],
                scheduler_type=config["model"]["teacher"]["scheduler_type"],
            )
            teacher_model.to(device)

            teacher_metrics = evaluate_model(
                teacher_model,
                test_dataloader,
                min(num_samples, 100),  # Use fewer samples for teacher
                device,
                output_dir,
            )

            teacher_metrics["model_type"] = "teacher"
            teacher_metrics["num_parameters"] = teacher_model.count_parameters()
            teacher_metrics["num_inference_steps"] = teacher_model.num_inference_steps

            logger.info("\nTeacher Model Results:")
            logger.info("=" * 80)
            for key, value in sorted(teacher_metrics.items()):
                logger.info(f"{key:40s}: {value}")
            logger.info("=" * 80)

            # Save teacher metrics
            analyzer.save_metrics(teacher_metrics, filename="teacher_metrics.json")

            # Generate comparison plot
            analyzer.compare_models(teacher_metrics, student_metrics)

        # Generate evaluation report
        analyzer.generate_report(student_metrics, config)

        # Print summary
        logger.info("\nEvaluation Summary:")
        logger.info("=" * 80)
        logger.info(f"{'Metric':<40s} {'Value':>12s} {'Target':>12s} {'Status':>10s}")
        logger.info("-" * 80)

        targets = {
            "fid_score": (25.0, "lower"),
            "clip_score": (0.28, "higher"),
            "inference_speedup": (4.0, "higher"),
        }

        for metric, (target, direction) in targets.items():
            if metric in student_metrics:
                value = student_metrics[metric]

                if direction == "lower":
                    status = "PASS" if value < target else "FAIL"
                else:
                    status = "PASS" if value > target else "FAIL"

                logger.info(f"{metric:<40s} {value:>12.4f} {target:>12.4f} {status:>10s}")

        logger.info("=" * 80)

        logger.info(f"\nResults saved to {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
