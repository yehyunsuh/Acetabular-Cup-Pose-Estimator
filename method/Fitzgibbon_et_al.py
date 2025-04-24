"""
Fitzgibbon, A., Pilu, M., & Fisher, R. B. (1999).
Direct least square fitting of ellipses. 
IEEE Transactions on pattern analysis and machine intelligence, 21(5), 476-480.
"""

import torch


def coords_euclidean_to_ep(X):
    new_coords = torch.stack([
        X[:,0]**2,
        X[:,0]*X[:,1],
        X[:,1]**2,
        X[:,0],
        X[:,1],
        torch.ones_like(X[:,0])
    ],axis=1)

    return new_coords


def coords_to_scatter_mat(X):
    X_ep = coords_euclidean_to_ep(X)
    X_ep_T = torch.transpose(X_ep,-2,-1)

    return torch.matmul(X_ep_T, X_ep)


def constraint_mats():
    constraint_mat_1 = torch.tensor([[1,0],[0,0]], dtype=torch.float64)
    constraint_mat_2 = torch.tensor([[0,0,2],[0,-1,0],[2,0,0]], dtype=torch.float64)
    C = torch.kron(constraint_mat_1,constraint_mat_2)

    return constraint_mat_1, constraint_mat_2, C


def fitzgibbon_ellipse(M_mat, C_mat):
    U_M, s_M, V_M = torch.svd(M_mat.to(torch.float64))
    M_inv = U_M @ torch.diag(1./s_M) @ V_M
    U, s, V = torch.svd(M_inv)
    a = U[:, 0:1]
    return torch.transpose(a, -2, -1)


def params_ep_to_ab(P):
    A,B,C,D,E,F = P[:,0],P[:,1],P[:,2],P[:,3],P[:,4],P[:,5]
    
    B_half = B/2.
    
    k1 = (C*D - B_half*E) / (2*(B_half*B_half - A*C))
    k2 = (A*E - B_half*D) / (2*(B_half*B_half - A*C))
    mu = 1./(A*k1*k1 + 2*B_half*k1*k2 + C*k2*k2 - F)
    
    m11 = mu*A
    m12 = mu*B_half
    m22 = mu*C
    
    lambda1 = (0.5)*(m11 + m22+torch.sqrt((m11-m22)**2 + 4*(m12**2)))
    lambda2 = (0.5)*(m11 + m22-torch.sqrt((m11-m22)**2 + 4*(m12**2)))

    a = 1./torch.sqrt(lambda1)
    b = 1./torch.sqrt(lambda2)
    if a < b:
        a,b = b,a
    
    theta = 0.5 * torch.atan2(-2*B_half,C-A)
    theta_deg = torch.rad2deg(theta)
    if theta_deg < 0:
        theta_deg += 180
    if theta_deg > 180:
        theta_deg -= 180
    
    return torch.stack([k1, k2, a, b, theta_deg], dim=-1)


def fitzgibbon_et_al(landmark_proj):
    landmark_proj_2D = landmark_proj[:, :2]
    landmark_proj_2D_avg = torch.mean(landmark_proj_2D, axis=0)
    landmark_proj_2D_origin = landmark_proj_2D - landmark_proj_2D_avg

    M_matrix = coords_to_scatter_mat(landmark_proj_2D_origin)
    _, _, C = constraint_mats()

    fitzgibbon_ellipse_equation = fitzgibbon_ellipse(M_matrix, C)
    fitzgibbon_ellipse_parameters = params_ep_to_ab(fitzgibbon_ellipse_equation)
    
    E_hat_x, E_hat_y, E_hat_major, E_hat_minor, E_hat_angle = fitzgibbon_ellipse_parameters[0]
    E_hat_x, E_hat_y = E_hat_x + landmark_proj_2D_avg[0], E_hat_y + landmark_proj_2D_avg[1] 

    return torch.stack([E_hat_x.detach().cpu(), E_hat_y.detach().cpu(), E_hat_major, E_hat_minor, E_hat_angle])