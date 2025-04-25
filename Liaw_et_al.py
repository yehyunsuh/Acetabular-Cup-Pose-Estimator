"""
Liaw_et_al.py

Implementation of the Liaw et al. method to estimate acetabular cup anteversion
from 2D ellipse parameters fitted to plain radiographs.

Reference:
    Park, Y. S., Shin, W. C., Lee, S. M., Kwak, S. H., Bae, J. Y., & Suh, K. T. (2018). 
    The best method for evaluating anteversion of the acetabular component after total hip arthroplasty on plain radiographs. 
    Journal of orthopaedic surgery and research, 13, 1-8.

    Liaw, C. K., Hou, S. M., Yang, R. S., Wu, T. Y., & Fuh, C. S. (2006).  
    A new tool for measuring cup orientation in total hip arthroplasties from plain radiographs.  
    Clinical Orthopaedics and Related Research, 451, 134–139.

Author: Yehyun Suh  
Date: 2025-04-27
"""


import torch


def calculate_vertices_and_covertices(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    major_axis: torch.Tensor,
    minor_axis: torch.Tensor,
    angle_deg: torch.Tensor
):
    """
    Calculate the vertices and co-vertices of an ellipse given its parameters.

    Args:
        center_x (Tensor): X-coordinate of the ellipse center.
        center_y (Tensor): Y-coordinate of the ellipse center.
        major_axis (Tensor): Semi-major axis length.
        minor_axis (Tensor): Semi-minor axis length.
        angle_deg (Tensor): Rotation angle of the ellipse in degrees.

    Returns:
        tuple:
            - vertex1 (Tensor): Endpoint of the major axis (positive direction).
            - vertex2 (Tensor): Endpoint of the major axis (negative direction).
            - co_vertex1 (Tensor): Endpoint of the minor axis (positive direction).
            - co_vertex2 (Tensor): Endpoint of the minor axis (negative direction).
    """
    angle_rad = torch.deg2rad(angle_deg)

    # Major axis vertices
    vertex1 = torch.stack([
        center_x + major_axis * torch.cos(angle_rad),
        center_y + major_axis * torch.sin(angle_rad)
    ])

    vertex2 = torch.stack([
        center_x - major_axis * torch.cos(angle_rad),
        center_y - major_axis * torch.sin(angle_rad)
    ])

    # Minor axis co-vertices (90 degrees offset)
    co_vertex1 = torch.stack([
        center_x + minor_axis * torch.cos(angle_rad + torch.pi / 2),
        center_y + minor_axis * torch.sin(angle_rad + torch.pi / 2)
    ])

    co_vertex2 = torch.stack([
        center_x - minor_axis * torch.cos(angle_rad + torch.pi / 2),
        center_y - minor_axis * torch.sin(angle_rad + torch.pi / 2)
    ])

    return vertex1, vertex2, co_vertex1, co_vertex2


def liaw_et_al(ellipse: torch.Tensor) -> torch.Tensor:
    """
    Estimate acetabular cup anteversion from an ellipse fitted to a radiograph.

    Args:
        ellipse (Tensor): Ellipse parameters [center_x, center_y, major, minor, angle_deg].

    Returns:
        Tensor: Anteversion angle in degrees.
    """
    center_x, center_y, major_axis, minor_axis, angle_deg = ellipse

    vertex1, vertex2, _, co_vertex2 = calculate_vertices_and_covertices(
        center_x, center_y, major_axis, minor_axis, angle_deg
    )

    # Vectors along major and minor axes
    u = vertex2 - vertex1
    v = vertex2 - co_vertex2

    # Calculate the angle between the vectors
    cos_beta = torch.dot(u, v) / (torch.norm(u) * torch.norm(v))
    beta = torch.acos(cos_beta)

    # Anteversion is derived from β
    anteversion = torch.rad2deg(torch.asin(torch.tan(beta)))

    return anteversion
