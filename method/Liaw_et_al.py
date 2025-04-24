"""
Park, Y. S., Shin, W. C., Lee, S. M., Kwak, S. H., Bae, J. Y., & Suh, K. T. (2018). 
The best method for evaluating anteversion of the acetabular component after total hip arthroplasty on plain radiographs. 
Journal of orthopaedic surgery and research, 13, 1-8.

Liaw, C. K., Hou, S. M., Yang, R. S., Wu, T. Y., & Fuh, C. S. (2006). 
A new tool for measuring cup orientation in total hip arthroplasties from plain radiographs. 
Clinical Orthopaedics and Related Research (1976-2007), 451, 134-139.
"""

import torch


def calculate_vertices_covertices(center_x, center_y, minor_axis, major_axis, angle):
    # Convert the angle to radians
    angle_rad = torch.deg2rad(angle)
    
    # Calculate half lengths of the axes
    a = major_axis
    b = minor_axis
    
    # Vertices (endpoints of the major axis)
    vertex1 = torch.tensor([center_x + a * torch.cos(angle_rad), center_y + a * torch.sin(angle_rad)])
    vertex2 = torch.tensor([center_x - a * torch.cos(angle_rad), center_y - a * torch.sin(angle_rad)])
    
    # Co-vertices (endpoints of the minor axis)
    co_vertex1 = torch.tensor([center_x + b * torch.cos(angle_rad + torch.pi / 2), center_y + b * torch.sin(angle_rad + torch.pi / 2)])
    co_vertex2 = torch.tensor([center_x - b * torch.cos(angle_rad + torch.pi / 2), center_y - b * torch.sin(angle_rad + torch.pi / 2)])
    
    return vertex1, vertex2, co_vertex1, co_vertex2


def liaw_et_al(ellipse):
    center_x, center_y, major, minor, angle = ellipse
    vertex1, vertex2, _, co_vertex2 = calculate_vertices_covertices(center_x, center_y, minor, major, angle)

    # Create 2 vectors to calculate the angle between the vectors
    vector_u = vertex2 - vertex1
    vector_v = vertex2 - co_vertex2

    # Calculate the angle between the vectors
    beta = torch.acos(torch.dot(vector_u, vector_v) / (torch.norm(vector_u) * torch.norm(vector_v)))

    # Calculate the anteversion angle
    anteversion = torch.rad2deg(torch.asin(torch.tan(beta)))

    return anteversion