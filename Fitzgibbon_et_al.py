"""
Fitzgibbon_et_al.py

Differentiable implementation of Fitzgibbon's direct least-squares ellipse fitting algorithm
for 2D landmark projections.

Reference:
    Fitzgibbon, A., Pilu, M., & Fisher, R. B. (1999).
    Direct least square fitting of ellipses.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 21(5), 476–480.

Author: Yehyun Suh
Date: 2025-04-27
"""

import torch


def coords_euclidean_to_ep(X):
    """
    Map 2D coordinates to the extended parameter (EP) space for conic fitting.

    Args:
        X (Tensor): Input coordinates of shape [N, 2].

    Returns:
        Tensor: Transformed coordinates of shape [N, 6].
    """
    x = X[:, 0]
    y = X[:, 1]

    A = x**2
    B = x * y
    C = y**2
    D = x
    E = y
    F = torch.ones_like(x)

    return torch.stack([A, B, C, D, E, F], dim=-1)  # shape [N, 6]


def coords_to_scatter_mat(X):
    """
    Compute the scatter matrix S = X^T X for least-squares ellipse fitting.

    Args:
        X (Tensor): Input coordinates of shape [N, 2].

    Returns:
        Tensor: Scatter matrix of shape [6, 6].
    """
    X_ep = coords_euclidean_to_ep(X)
    return X_ep.T @ X_ep


def fitzgibbon_ellipse(M):
    """
    Solves the generalized eigenvalue problem to find ellipse coefficients.

    This function computes the eigenvector corresponding to the smallest eigenvalue
    of the inverse of the scatter matrix, which gives the conic parameters.

    Args:
        scatter_matrix (Tensor): The 6x6 scatter matrix (XᵀX), torch.float64.

    Returns:
        Tensor: Row vector of ellipse coefficients with shape [1, 6].
    """
    # Compute inverse of M using SVD
    U_M, S_M, V_M = torch.svd(M)
    M_inv = U_M @ torch.diag(1.0 / S_M) @ V_M.T

    # Solve S⁻¹ x = λx (eigenvalue problem)
    U, _, _ = torch.svd(M_inv)

    # First eigenvector corresponds to the minimum eigenvalue direction
    conic_params = U[:, 0].unsqueeze(0)  # shape [1, 6]

    return conic_params


def params_ep_to_ab(P):
    """
    Convert general conic parameters into ellipse parameters (center, axes, angle).

    Args:
        P (Tensor): Fitted conic parameters of shape [1, 6].

    Returns:
        Tensor: Ellipse parameters [x, y, major_axis, minor_axis, angle (degrees)].
    """
    A, B, C, D, E, F = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4], P[:, 5]
    B_half = B / 2

    denominator = 2 * (B_half**2 - A * C)
    x = (C * D - B_half * E) / denominator  # x-coordinate of center
    y = (A * E - B_half * D) / denominator  # y-coordinate of center

    mu = 1.0 / (A * x**2 + 2 * B_half * x * y + C * y**2 - F)

    m11 = mu * A
    m12 = mu * B_half
    m22 = mu * C

    eig_sum = m11 + m22
    eig_diff = m11 - m22
    eig_cross = 2 * m12

    lambda1 = 0.5 * (eig_sum + torch.sqrt(eig_diff**2 + eig_cross**2))
    lambda2 = 0.5 * (eig_sum - torch.sqrt(eig_diff**2 + eig_cross**2))

    a = 1.0 / torch.sqrt(lambda1)  # semi-major axis
    b = 1.0 / torch.sqrt(lambda2)  # semi-minor axis

    if a < b:
        a, b = b, a

    alpha = 0.5 * torch.atan2(-2 * B_half, C - A)  # orientation of ellipse
    alpha_deg = torch.rad2deg(alpha)
    alpha_deg = (alpha_deg + 180) % 180

    return torch.stack([x, y, a, b, alpha_deg], dim=-1)


def fitzgibbon_et_al(projected_landmarks):
    """
    Perform direct least-squares ellipse fitting for a given set of 2D landmarks.

    Args:
        landmark_proj (Tensor): Landmark coordinates of shape [N, 3].

    Returns:
        Tensor: Fitted ellipse parameters [center_x, center_y, major_axis, minor_axis, angle_deg].
    """
    projected_landmarks_2D = projected_landmarks[:, :2]
    mean_2D = projected_landmarks_2D.mean(dim=0)
    centered_2D = projected_landmarks_2D - mean_2D

    M = coords_to_scatter_mat(centered_2D)
    conic_params = fitzgibbon_ellipse(M)
    ellipse_params = params_ep_to_ab(conic_params)

    center_x, center_y = ellipse_params[0, 0] + mean_2D[0], ellipse_params[0, 1] + mean_2D[1]
    major_axis, minor_axis, angle_deg = ellipse_params[0, 2], ellipse_params[0, 3], ellipse_params[0, 4]

    return torch.stack([
        center_x.detach().cpu(),
        center_y.detach().cpu(),
        major_axis,
        minor_axis,
        angle_deg
    ])
