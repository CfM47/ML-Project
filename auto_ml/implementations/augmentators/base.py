"""Base classes and utilities for data augmentation."""

from typing import Tuple

import numpy as np
from scipy import ndimage

from auto_ml.interfaces import ImageArray, MaskArray


def apply_affine_to_image(
    image: ImageArray,
    matrix: np.ndarray,
    order: int = 1,
    fill_value: float = 0.0,
) -> ImageArray:
    """
    Apply affine transformation to an image.

    Args:
        image: Input image array (H, W) or (H, W, C).
        matrix: 2x3 or 3x3 affine transformation matrix.
        order: Interpolation order (0=nearest, 1=bilinear, 3=cubic).
        fill_value: Value to use for pixels outside boundaries.

    Returns:
        Transformed image with same shape as input.

    """
    # Handle both 2D and 3D images
    is_grayscale = image.ndim == 2

    if is_grayscale:
        result = ndimage.affine_transform(
            image,
            matrix[:2, :2],
            offset=matrix[:2, 2] if matrix.shape[0] == 3 else np.zeros(2),
            order=order,
            mode="constant",
            cval=fill_value,
        )
    else:
        # Apply to each channel separately
        channels = []
        for c in range(image.shape[2]):
            channel = ndimage.affine_transform(
                image[:, :, c],
                matrix[:2, :2],
                offset=matrix[:2, 2] if matrix.shape[0] == 3 else np.zeros(2),
                order=order,
                mode="constant",
                cval=fill_value,
            )
            channels.append(channel)
        result = np.stack(channels, axis=2)

    return result.astype(image.dtype)  # type: ignore[no-any-return]


def apply_affine_to_mask(
    mask: MaskArray,
    matrix: np.ndarray,
    fill_value: int = 0,
) -> MaskArray:
    """
    Apply affine transformation to a mask using nearest neighbor interpolation.

    Args:
        mask: Input mask array (H, W).
        matrix: 2x3 or 3x3 affine transformation matrix.
        fill_value: Value to use for pixels outside boundaries.

    Returns:
        Transformed mask with same shape as input.

    """
    result = ndimage.affine_transform(
        mask,
        matrix[:2, :2],
        offset=matrix[:2, 2] if matrix.shape[0] == 3 else np.zeros(2),
        order=0,  # Always use nearest neighbor for masks
        mode="constant",
        cval=fill_value,
    )

    return result.astype(mask.dtype)  # type: ignore[no-any-return]


def rotation_matrix(angle_degrees: float, center: Tuple[float, float]) -> np.ndarray:
    """
    Create a 3x3 rotation matrix.

    Args:
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise).
        center: Center of rotation (x, y).

    Returns:
        3x3 affine transformation matrix.

    """
    angle_rad = np.deg2rad(angle_degrees)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    cx, cy = center

    # Translation to origin -> Rotation -> Translation back
    matrix = np.array(
        [
            [cos_theta, -sin_theta, cx - cx * cos_theta + cy * sin_theta],
            [sin_theta, cos_theta, cy - cx * sin_theta - cy * cos_theta],
            [0, 0, 1],
        ],
    )

    return matrix


def scale_matrix(
    scale_x: float,
    scale_y: float,
    center: Tuple[float, float],
) -> np.ndarray:
    """
    Create a 3x3 scale matrix.

    Args:
        scale_x: Scale factor along x-axis.
        scale_y: Scale factor along y-axis.
        center: Center of scaling (x, y).

    Returns:
        3x3 affine transformation matrix.

    """
    cx, cy = center

    matrix = np.array(
        [
            [scale_x, 0, cx * (1 - scale_x)],
            [0, scale_y, cy * (1 - scale_y)],
            [0, 0, 1],
        ],
    )

    return matrix


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    """
    Create a 3x3 translation matrix.

    Args:
        tx: Translation along x-axis (positive = right).
        ty: Translation along y-axis (positive = down).

    Returns:
        3x3 affine transformation matrix.

    """
    return np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ],
    )
