"""Define the basic Image entity used across the project."""

from typing import TypeAlias

import numpy as np

Image: TypeAlias = np.ndarray
"""
A type alias for images.

Images are represented as numpy arrays. The shape of the array will depend
on the image type (e.g., grayscale, RGB, etc.).
- (H, W) for grayscale images.
- (H, W, C) for multi-channel images like RGB, where C is the number of channels.
"""
