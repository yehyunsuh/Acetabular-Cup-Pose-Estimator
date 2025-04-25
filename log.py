"""
log.py

Logging utilities for optimization-based ellipse fitting,
including formatted console outputs of variables and losses.

Author: Yehyun Suh  
Date: 2025-04-27
"""

import torch


def print_process(args, epoch, loss, vars, E, E_hat, hd_diff, patience):
    """
    Print the training process, including loss, variables, gradients,
    ellipse parameters, and Hausdorff distance.

    Args:
        args (Namespace): Configuration arguments.
        epoch (int): Current epoch number.
        loss (Tensor): Current loss value.
        vars (list[Tensor]): Optimization variables (e.g., angles, translations).
        E (Tensor): Ground truth ellipse parameters [center_x, center_y, major, minor, angle].
        E_hat (Tensor): Predicted ellipse parameters.
        hd_diff (float): Current Hausdorff distance between ellipses.
        patience (int): Current patience counter for early stopping.
    """
    print(
        f"Epoch {epoch}: "
        f"Loss = {loss.item():.6f}, "
        f"Vars = {', '.join(f'{v.item():.2f}' for v in vars)} | "
        f"Patience: {patience}"
    )

    print('\t', end='')
    print(
        "Gradients: " +
        ", ".join(f"{v.grad.item():.2f}" for v in vars)
    )

    print('\t', end='')
    print(
        "E     : " +
        ", ".join(f"{e.item():.2f}" for e in E)
    )

    print('\t', end='')
    print(
        "E_hat : " +
        ", ".join(f"{eh.item():.2f}" for eh in E_hat)
    )

    print('\t', end='')
    print(f"Hausdorff distance: {hd_diff:.4f}")
