import torch


def parameter_carm(args):
    # Parameter setting - C-arm
    H = torch.tensor(args.sdd, dtype=torch.float64)
    h = torch.tensor(args.pdd, dtype=torch.float64)

    return H, h


def parameter_implant(args):
    # Parameter setting - implant rotation, translation, and radius
    theta = torch.tensor(args.theta, dtype=torch.float64)
    phi = torch.tensor(args.phi, dtype=torch.float64)
    k = torch.tensor(args.k, dtype=torch.float64)
    l = torch.tensor(args.l, dtype=torch.float64)
    r = torch.tensor(args.r, dtype=torch.float64)
    
    return theta, phi, k, l, r


def parameter_circle(r, h):
    theta_circle = torch.linspace(0, 2 * torch.pi, 1000)
    x_circle = r * torch.cos(theta_circle)
    y_circle = torch.zeros(1000)
    z_circle = r * torch.sin(theta_circle) + h

    C = torch.stack([x_circle, y_circle, z_circle], dim=1)  # C represents Circle

    return C


def parameter_landmarks(args, r, h):
    # Environment setting - create landmarks on the circle
    theta_object = torch.linspace(0, 2 * torch.pi, args.n_landmarks)
    x_object = r * torch.cos(theta_object)
    y_object = torch.zeros(args.n_landmarks)
    z_object = r * torch.sin(theta_object) + h
    
    S = torch.stack([x_object, y_object, z_object], dim=1)  # S represents Set of landmarks

    return S


def parameter(args):
    H, h = parameter_carm(args)
    theta, phi, k, l, r = parameter_implant(args)
    C = parameter_circle(r, h)
    S = parameter_landmarks(args, r, h)

    return H, h, theta, phi, k, l, r, C, S