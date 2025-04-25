"""
transformation.py

Transformation utilities for 3D-to-2D geometric modeling of the acetabular cup.
Includes axis-aligned rotation matrices, rotation of coordinates about an axis,
and perspective projection for synthetic landmark simulation.

These functions are used in the optimization process to simulate the projection
of 3D anatomical structures onto a 2D imaging plane.

Author: Yehyun Suh  
Date: 2025-04-27
"""

import torch


def rotation_matrices(radian_angle, matrix_type):
    """
    Generate a 3x3 rotation matrix around the specified axis.

    Args:
        radian_angle (Tensor): Rotation angle in radians.
        matrix_type (str): One of {'x-axis', 'y-axis', 'z-axis'}.

    Returns:
        Tensor: A [3, 3] rotation matrix.
    """
    cos = torch.cos(radian_angle).to(dtype=torch.float64)
    sin = torch.sin(radian_angle).to(dtype=torch.float64)

    if matrix_type == 'x-axis':
        return torch.stack([
            torch.stack([torch.ones_like(cos), torch.zeros_like(cos), torch.zeros_like(cos)]),
            torch.stack([torch.zeros_like(cos), cos, -sin]),
            torch.stack([torch.zeros_like(cos), sin, cos])
        ])
    elif matrix_type == 'y-axis':
        return torch.stack([
            torch.stack([cos, torch.zeros_like(cos), sin]),
            torch.stack([torch.zeros_like(cos), torch.ones_like(cos), torch.zeros_like(cos)]),
            torch.stack([-sin, torch.zeros_like(cos), cos])
        ])
    elif matrix_type == 'z-axis':
        return torch.stack([
            torch.stack([cos, -sin, torch.zeros_like(cos)]),
            torch.stack([sin, cos, torch.zeros_like(cos)]),
            torch.stack([torch.zeros_like(cos), torch.zeros_like(cos), torch.ones_like(cos)])
        ])
    else:
        raise ValueError(f"Invalid matrix_type '{matrix_type}'. Choose from 'x-axis', 'y-axis', or 'z-axis'.")


def rotate_coordinates(coordinate, translation, rotation_matrix):
    """
    Rotate 3D coordinates around a given axis with respect to a translation origin.

    Args:
        coordinate (Tensor): [N, 3] input 3D coordinates.
        translation (Tensor): [3] translation vector applied before and after rotation.
        rotation_matrix (Tensor): [3, 3] rotation matrix.

    Returns:
        Tensor: [N, 3] rotated coordinates.
    """
    coordinate_origin = coordinate - translation
    coordinate_origin_rotated = torch.matmul(rotation_matrix, coordinate_origin.t()).t()
    coordinate_rotated = coordinate_origin_rotated + translation

    return coordinate_rotated


def project_coordinates(ratio, coordinate):
    """
    Apply perspective projection from 3D to 2D using camera model.

    Args:
        ratio (Tensor): [N] projection ratio (e.g., H / (H - z)).
        coordinate (Tensor): [N, 3] input 3D coordinates.

    Returns:
        Tensor: [N, 3] projected 2D coordinates with z = 0.
    """
    x_proj = coordinate[:, 0] * ratio
    y_proj = coordinate[:, 1] * ratio
    z_proj = torch.zeros_like(x_proj)

    coordinate_projected = torch.stack([x_proj, y_proj, z_proj], dim=1)

    return coordinate_projected
