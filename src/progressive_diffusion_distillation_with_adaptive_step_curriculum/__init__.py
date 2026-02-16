"""Progressive Diffusion Distillation with Adaptive Step Curriculum.

This package implements a novel approach to distilling large text-to-image diffusion models
into faster few-step generators using progressive knowledge distillation combined with an
adaptive curriculum that dynamically adjusts timestep sampling based on student model
convergence rate per region of the denoising trajectory.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from .models.model import StudentDiffusionModel, TeacherDiffusionModel
from .models.components import AdaptiveCurriculumScheduler, ProgressiveDistillationLoss
from .training.trainer import ProgressiveDistillationTrainer

__all__ = [
    "StudentDiffusionModel",
    "TeacherDiffusionModel",
    "AdaptiveCurriculumScheduler",
    "ProgressiveDistillationLoss",
    "ProgressiveDistillationTrainer",
]
