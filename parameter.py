"""
parameter.py

Generates synthetic 3D coordinates for circular structures representing the acetabular cup.
This includes:
- A dense set of points defining the full circle contour (used for ellipse fitting).
- A sparse set of landmark points (used for projection and optimization).

These geometric primitives are defined based on the radius of the implant and the object-to-detector
distance and are aligned on the x-z plane, assuming the implant lies flat at y = 0.

Author: Yehyun Suh  
Date: 2025-04-27
"""

import torch


def parameter_circle(r, h):
    """
    Generate a dense set of points forming a circle representing the implant cross-section.

    Args:
        r (Tensor): Radius of the acetabular cup.
        h (Tensor): Object-to-detector distance.

    Returns:
        Tensor: Circle points of shape [1000, 3].
    """
    theta_circle = torch.linspace(0, 2 * torch.pi, 1000)
    x_circle = r * torch.cos(theta_circle)
    y_circle = torch.zeros(1000, dtype=torch.float64)
    z_circle = r * torch.sin(theta_circle) + h

    C = torch.stack([x_circle, y_circle, z_circle], dim=1)

    return C


def parameter_landmarks(args, r, h):
    """
    Generate synthetic sparse landmarks sampled along the implant circle.

    Args:
        args (Namespace): Parsed command-line arguments.
        r (Tensor): Radius of the acetabular cup.
        h (Tensor): Object-to-detector distance.

    Returns:
        Tensor: Landmark points of shape [n_landmarks, 3].
    """
    theta_object = torch.linspace(0, 2 * torch.pi, args.n_landmarks)
    x_object = r * torch.cos(theta_object)
    y_object = torch.zeros(args.n_landmarks, dtype=torch.float64)
    z_object = r * torch.sin(theta_object) + h

    S = torch.stack([x_object, y_object, z_object], dim=1)

    return S