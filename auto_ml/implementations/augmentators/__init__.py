"""Data augmentation implementations."""

# Identity augmentator (baseline)
# Composite augmentations (combine multiple augmentations)
from auto_ml.implementations.augmentators.composite import (
    MultiplyDatasetAugmentator,
    OneOfAugmentator,
    RandomApplyAugmentator,
    RandomChoiceAugmentator,
    SequentialAugmentator,
)

# Geometric augmentations (affect both image and mask)
from auto_ml.implementations.augmentators.geometric import (
    HorizontalFlipAugmentator,
    RandomCropAugmentator,
    RotationAugmentator,
    ScaleAugmentator,
    TranslationAugmentator,
    VerticalFlipAugmentator,
)
from auto_ml.implementations.augmentators.identity import IdentityAugmentator

# Photometric augmentations (affect only image)
from auto_ml.implementations.augmentators.photometric import (
    BrightnessAugmentator,
    ContrastAugmentator,
    GammaAugmentator,
    GaussianBlurAugmentator,
    GaussianNoiseAugmentator,
)

# SEM-specific augmentations
from auto_ml.implementations.augmentators.sem_specific import (
    AdaptiveHistogramEqualizationAugmentator,
    ChargingArtifactAugmentator,
    ElasticDeformationAugmentator,
    ScanLineNoiseAugmentator,
)

__all__ = [
    # Identity
    "IdentityAugmentator",
    # Geometric
    "RotationAugmentator",
    "HorizontalFlipAugmentator",
    "VerticalFlipAugmentator",
    "ScaleAugmentator",
    "TranslationAugmentator",
    "RandomCropAugmentator",
    # Photometric
    "BrightnessAugmentator",
    "ContrastAugmentator",
    "GaussianNoiseAugmentator",
    "GaussianBlurAugmentator",
    "GammaAugmentator",
    # Composite
    "SequentialAugmentator",
    "RandomChoiceAugmentator",
    "RandomApplyAugmentator",
    "OneOfAugmentator",
    "MultiplyDatasetAugmentator",
    # SEM-specific
    "ElasticDeformationAugmentator",
    "AdaptiveHistogramEqualizationAugmentator",
    "ChargingArtifactAugmentator",
    "ScanLineNoiseAugmentator",
]
