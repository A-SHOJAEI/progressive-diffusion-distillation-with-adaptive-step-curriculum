#!/usr/bin/env python
"""Quick test to verify basic functionality without full training."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing imports...")

# Test basic imports
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

# Test project imports
try:
    from progressive_diffusion_distillation_with_adaptive_step_curriculum.utils.config import load_config
    print("✓ Config loading")
except ImportError as e:
    print(f"✗ Config loading failed: {e}")
    sys.exit(1)

try:
    from progressive_diffusion_distillation_with_adaptive_step_curriculum.models.components import (
        AdaptiveCurriculumScheduler,
        TimestepDivergenceTracker,
        ProgressiveDistillationLoss,
    )
    print("✓ Custom components")
except ImportError as e:
    print(f"✗ Custom components import failed: {e}")
    sys.exit(1)

try:
    from progressive_diffusion_distillation_with_adaptive_step_curriculum.data.loader import get_dataloader
    print("✓ Data loader")
except ImportError as e:
    print(f"✗ Data loader import failed: {e}")
    sys.exit(1)

# Test configuration loading
try:
    config = load_config("configs/default.yaml")
    print("✓ Configuration loaded")
    print(f"  - Curriculum enabled: {config['curriculum']['enabled']}")
    print(f"  - Student steps: {config['model']['student']['num_inference_steps']}")
except Exception as e:
    print(f"✗ Configuration loading failed: {e}")
    sys.exit(1)

# Test custom components initialization
try:
    device = torch.device("cpu")

    # Test divergence tracker
    tracker = TimestepDivergenceTracker(num_timesteps=1000, num_regions=3)
    timesteps = torch.tensor([100, 500, 900])
    divergences = torch.tensor([0.5, 0.8, 0.3])
    tracker.update(timesteps, divergences)
    weights = tracker.get_sampling_weights()
    print(f"✓ TimestepDivergenceTracker (weights sum: {weights.sum():.3f})")

    # Test curriculum scheduler
    scheduler = AdaptiveCurriculumScheduler(num_timesteps=1000, num_regions=3)
    timesteps = scheduler.sample_timesteps(batch_size=4, device=device)
    print(f"✓ AdaptiveCurriculumScheduler (sampled {len(timesteps)} timesteps)")

    # Test loss function
    loss_fn = ProgressiveDistillationLoss(output_weight=1.0, feature_weight=0.5)
    student_out = torch.randn(2, 3, 8, 8)  # 3 channels for RGB
    teacher_out = torch.randn(2, 3, 8, 8)  # 3 channels for RGB
    total_loss, loss_dict = loss_fn(student_out, teacher_out, progress=0.5)
    print(f"✓ ProgressiveDistillationLoss (loss: {total_loss.item():.4f})")

except Exception as e:
    print(f"✗ Component initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test data loader (synthetic data)
try:
    dataloader = get_dataloader(
        split="train",
        batch_size=2,
        image_size=64,
        max_samples=4,
        num_workers=0,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    print(f"✓ DataLoader (batch size: {batch['pixel_values'].shape[0]})")
except Exception as e:
    print(f"✗ DataLoader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All basic tests passed!")
print("\nNOTE: This does not test full training. To verify training:")
print("  python scripts/train.py --config configs/default.yaml")
