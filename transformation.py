import torch


def rotation_matrices(radian_angle, matrix_type):
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
    

def rotate_coordinates(coordinate, translation, rotation_matrix):
    coordinate_origin = coordinate - translation
    coordinate_origin_rotated = torch.matmul(rotation_matrix, coordinate_origin.t()).t()
    coordinate_rotated = coordinate_origin_rotated + translation

    return coordinate_rotated


def project_coordinates(ratio, coordinate):
    x_proj = coordinate[:, 0] * ratio
    y_proj = coordinate[:, 1] * ratio
    z_proj = torch.zeros_like(x_proj)
    coordinate_projected = torch.stack([x_proj, y_proj, z_proj], dim=1)

    return coordinate_projected


def rotate_translate_project(args, S, C, theta, phi, k, l, H, h, E, vis=False):
    S_R, C_R, S_theta, C_theta = None, None, None, None
    
    # 2. Rotate the landmarks by theta on y-axis and phi on z-axis: S_R = R(S)
    rotation_matrix_x = rotation_matrices(torch.deg2rad(theta), 'x-axis')
    rotation_matrix_z = rotation_matrices(torch.deg2rad(phi), 'z-axis')
    rotation_matrix = torch.matmul(rotation_matrix_z, rotation_matrix_x)

    extra_plane_translation = torch.stack([torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64), h])
    S_R = rotate_coordinates(S, extra_plane_translation, rotation_matrix)
    C_R = rotate_coordinates(C, extra_plane_translation, rotation_matrix)

    # 3. Translate the landmarks by k on x-axis and l on y-axis: S_T = T(S_R)
    # Translation uses k, l directly in the computation graph
    in_plane_translation = torch.stack([k, l, torch.tensor(0.0, dtype=k.dtype)], dim=0)
    S_T = S_R + in_plane_translation  
    C_T = C_R + in_plane_translation

    # 4. Project the landmarks on to the image plane: S_p = P(S_T)
    ratio_S = H / (H - S_T[:, 2])
    S_P = project_coordinates(ratio_S, S_T)
    ratio_C = H / (H - C_T[:, 2])
    C_P = project_coordinates(ratio_C, C_T)

    return S_T, C_T, S_P, C_P