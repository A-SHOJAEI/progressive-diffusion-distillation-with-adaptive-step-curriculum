"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzer for training and evaluation results."""

    def __init__(self, results_dir: str = "results"):
        """Initialize results analyzer.

        Args:
            results_dir: Directory containing results.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized results analyzer: {self.results_dir}")

    def save_metrics(
        self,
        metrics: Dict[str, float],
        filename: str = "metrics.json",
    ) -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics.
            filename: Output filename.
        """
        output_path = self.results_dir / filename

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {output_path}")

    def load_metrics(self, filename: str = "metrics.json") -> Dict[str, float]:
        """Load metrics from JSON file.

        Args:
            filename: Input filename.

        Returns:
            Dictionary of metrics.
        """
        input_path = self.results_dir / filename

        if not input_path.exists():
            logger.warning(f"Metrics file not found: {input_path}")
            return {}

        with open(input_path, 'r') as f:
            metrics = json.load(f)

        return metrics

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        learning_rates: Optional[List[float]] = None,
        filename: str = "training_curves.png",
    ) -> None:
        """Plot training curves.

        Args:
            train_losses: List of training losses.
            val_losses: Optional list of validation losses.
            learning_rates: Optional list of learning rates.
            filename: Output filename.
        """
        num_plots = 1
        if val_losses is not None:
            num_plots += 1
        if learning_rates is not None:
            num_plots += 1

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))

        if num_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Training loss
        axes[plot_idx].plot(train_losses, label="Train Loss", linewidth=2)
        if val_losses is not None:
            axes[plot_idx].plot(val_losses, label="Val Loss", linewidth=2)
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("Loss")
        axes[plot_idx].set_title("Training and Validation Loss")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Learning rate
        if learning_rates is not None and plot_idx < len(axes):
            axes[plot_idx].plot(learning_rates, linewidth=2, color="orange")
            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel("Learning Rate")
            axes[plot_idx].set_title("Learning Rate Schedule")
            axes[plot_idx].set_yscale("log")
            axes[plot_idx].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved training curves to {output_path}")

    def compare_models(
        self,
        baseline_metrics: Dict[str, float],
        model_metrics: Dict[str, float],
        filename: str = "model_comparison.png",
    ) -> None:
        """Compare baseline and model metrics.

        Args:
            baseline_metrics: Baseline model metrics.
            model_metrics: Current model metrics.
            filename: Output filename.
        """
        # Find common metrics
        common_keys = set(baseline_metrics.keys()) & set(model_metrics.keys())

        if not common_keys:
            logger.warning("No common metrics to compare")
            return

        metric_names = sorted(common_keys)
        baseline_values = [baseline_metrics[k] for k in metric_names]
        model_values = [model_metrics[k] for k in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(x - width/2, baseline_values, width, label="Baseline", alpha=0.8)
        ax.bar(x + width/2, model_values, width, label="Current Model", alpha=0.8)

        ax.set_xlabel("Metrics")
        ax.set_ylabel("Value")
        ax.set_title("Model Comparison: Baseline vs Current")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved model comparison to {output_path}")

    def analyze_curriculum(
        self,
        curriculum_stats: Dict[str, List[float]],
        filename: str = "curriculum_analysis.png",
    ) -> None:
        """Analyze adaptive curriculum statistics.

        Args:
            curriculum_stats: Dictionary with curriculum statistics over time.
            filename: Output filename.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot divergence per region
        num_regions = sum(1 for k in curriculum_stats.keys() if k.startswith("divergence_region_"))

        for i in range(num_regions):
            key = f"divergence_region_{i}"
            if key in curriculum_stats:
                axes[0].plot(curriculum_stats[key], label=f"Region {i}", linewidth=2)

        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Divergence")
        axes[0].set_title("Student-Teacher Divergence by Region")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot sampling weights per region
        for i in range(num_regions):
            key = f"sampling_weight_region_{i}"
            if key in curriculum_stats:
                axes[1].plot(curriculum_stats[key], label=f"Region {i}", linewidth=2)

        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Sampling Weight")
        axes[1].set_title("Adaptive Curriculum Weights by Region")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved curriculum analysis to {output_path}")

    def generate_report(
        self,
        metrics: Dict[str, float],
        config: Dict,
        filename: str = "evaluation_report.txt",
    ) -> None:
        """Generate text report of evaluation results.

        Args:
            metrics: Evaluation metrics.
            config: Configuration used.
            filename: Output filename.
        """
        output_path = self.results_dir / filename

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PROGRESSIVE DIFFUSION DISTILLATION - EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            for key, value in sorted(config.items()):
                f.write(f"{key:30s}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("EVALUATION METRICS\n")
            f.write("=" * 80 + "\n\n")

            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    f.write(f"{key:40s}: {value:>12.4f}\n")
                else:
                    f.write(f"{key:40s}: {value:>12}\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Saved evaluation report to {output_path}")


def visualize_results(
    images: List[torch.Tensor],
    texts: Optional[List[str]] = None,
    output_path: str = "results/generated_samples.png",
    nrow: int = 4,
) -> None:
    """Visualize generated images in a grid.

    Args:
        images: List of image tensors (C, H, W).
        texts: Optional list of text prompts.
        output_path: Path to save visualization.
        nrow: Number of images per row.
    """
    from torchvision.utils import make_grid

    # Normalize images to [0, 1]
    normalized_images = []
    for img in images:
        if img.min() < 0:
            img = (img + 1) / 2
        img = img.clamp(0, 1)
        normalized_images.append(img)

    # Create grid
    grid = make_grid(normalized_images, nrow=nrow, padding=2, normalize=False)

    # Convert to PIL and save
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    grid_pil = Image.fromarray(grid_np)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_pil.save(output_path)

    logger.info(f"Saved visualization to {output_path}")


def save_samples_with_captions(
    images: List[torch.Tensor],
    texts: List[str],
    output_dir: str = "results/samples",
) -> None:
    """Save individual images with their captions.

    Args:
        images: List of image tensors.
        texts: List of text prompts.
        output_dir: Directory to save samples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img, text) in enumerate(zip(images, texts)):
        # Normalize image
        if img.min() < 0:
            img = (img + 1) / 2
        img = img.clamp(0, 1)

        # Convert to PIL
        img_pil = Image.fromarray(
            (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )

        # Save image
        img_path = output_dir / f"sample_{idx:04d}.png"
        img_pil.save(img_path)

        # Save caption
        caption_path = output_dir / f"sample_{idx:04d}.txt"
        with open(caption_path, 'w') as f:
            f.write(text)

    logger.info(f"Saved {len(images)} samples to {output_dir}")
