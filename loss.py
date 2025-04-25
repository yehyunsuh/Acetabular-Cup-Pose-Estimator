"""
loss.py

Custom loss functions for ellipse parameter regression tasks.
Includes:
- CustomMSELoss: Angle-aware mean squared error with 180-degree periodicity handling.
- CustomHausdorffLoss: Hausdorff distance between predicted and ground truth ellipses.

Author: Yehyun Suh  
Date: 2025-04-27
"""

import torch
from scipy.spatial.distance import directed_hausdorff


class CustomMSELoss(torch.nn.Module):
    """
    Custom MSE loss for ellipse parameters.
    Handles 180-degree periodicity for the final angle component.
    """

    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, prediction, ground_truth):
        """
        Compute angle-aware MSE between predicted and ground truth ellipse parameters.

        Args:
            prediction (Tensor): Predicted parameters [x, y, a, b, angle] (shape: [5])
            ground_truth (Tensor): Ground truth parameters [x, y, a, b, angle] (shape: [5])

        Returns:
            Tensor: Scalar MSE loss.
        """
        diff = prediction[:4] - ground_truth[:4]

        angle_diff = torch.abs(prediction[4] - ground_truth[4])
        angle_diff = torch.min(angle_diff, 180 - angle_diff)

        combined_diff = torch.cat((diff, angle_diff.unsqueeze(0)), dim=0)
        loss = torch.mean(combined_diff ** 2)

        return loss


def generate_ellipse_points(x, y, semi_major, semi_minor, rotation, num_points=100):
    """
    Generate sampled (x, y) coordinates along an ellipse.

    Args:
        x (float): Center x-coordinate.
        y (float): Center y-coordinate.
        semi_major (float): Semi-major axis length.
        semi_minor (float): Semi-minor axis length.
        rotation (float): Rotation angle in radians.
        num_points (int): Number of points to generate.

    Returns:
        Tensor: Rotated and translated points along the ellipse (shape: [num_points, 2]).
    """
    theta = torch.linspace(0, 2 * torch.pi, num_points)
    ellipse_x = semi_major * torch.cos(theta).to(dtype=torch.float64)
    ellipse_y = semi_minor * torch.sin(theta).to(dtype=torch.float64)

    cos_r = torch.cos(rotation)
    sin_r = torch.sin(rotation)

    rotation_matrix = torch.tensor([[cos_r, -sin_r], [sin_r, cos_r]])
    ellipse_points = torch.stack((ellipse_x, ellipse_y))

    rotated_points = torch.matmul(rotation_matrix, ellipse_points)
    rotated_points[0] += x
    rotated_points[1] += y

    return rotated_points.T


class CustomHausdorffLoss(torch.nn.Module):
    """
    Custom Hausdorff distance loss for ellipses.
    Uses non-differentiable scipy directed_hausdorff.
    """

    def __init__(self):
        super(CustomHausdorffLoss, self).__init__()

    def forward(self, ellipse1, ellipse2):
        """
        Compute Hausdorff distance between two ellipses.

        Args:
            ellipse1 (Tensor): [x, y, a, b, theta] of predicted ellipse
            ellipse2 (Tensor): [x, y, a, b, theta] of ground truth ellipse

        Returns:
            Tensor: Scalar Hausdorff distance (non-differentiable)
        """
        x1, y1, a1, b1, theta1 = ellipse1
        x2, y2, a2, b2, theta2 = ellipse2

        points1 = generate_ellipse_points(x1, y1, a1, b1, theta1)
        points2 = generate_ellipse_points(x2, y2, a2, b2, theta2)

        np1 = points1.detach().cpu().numpy()
        np2 = points2.detach().cpu().numpy()

        d1 = directed_hausdorff(np1, np2)[0]
        d2 = directed_hausdorff(np2, np1)[0]

        hausdorff = max(d1, d2)

        return torch.tensor(hausdorff, dtype=torch.float32, requires_grad=True)
