import torch

from scipy.spatial.distance import directed_hausdorff


class CustomMSELoss(torch.nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, prediction, ground_truth):
        # Calculate the error between the first four elements
        four_E = prediction[:4] - ground_truth[:4]

        # Calculate the angle error and adjust for 180-degree periodicity
        angle_E = torch.abs(prediction[4] - ground_truth[4])
        angle_E = torch.min(angle_E, 180 - angle_E)

        # Create a tensor of 5 elements with the angle error as the last element
        diff_E = torch.cat((four_E, angle_E.unsqueeze(0)), dim=0)
        diff_SE = diff_E ** 2
        diff_MSE = torch.mean(diff_SE)

        return diff_MSE
    

def generate_ellipse_points(x, y, semi_major, semi_minor, rotation, num_points=100):
    theta = torch.linspace(0, 2 * torch.pi, num_points)  # Parameterize the ellipse
    ellipse_x = semi_major * torch.cos(theta).to(dtype=torch.float64)
    ellipse_y = semi_minor * torch.sin(theta).to(dtype=torch.float64)
    
    # Apply rotation matrix
    cos_r = torch.cos(rotation)
    sin_r = torch.sin(rotation)
    
    rotation_matrix = torch.tensor([[cos_r, -sin_r], [sin_r, cos_r]])
    ellipse_points = torch.stack((ellipse_x, ellipse_y))
    
    # Rotate and translate the ellipse
    rotated_points = torch.matmul(rotation_matrix, ellipse_points)
    rotated_points[0] += x
    rotated_points[1] += y
    
    return rotated_points.T


class CustomHausdorffLoss(torch.nn.Module):
    def __init__(self):
        super(CustomHausdorffLoss, self).__init__()

    def forward(self, ellipse1, ellipse2):
        # Unpack the ellipse parameters
        x1, y1, semi_major1, semi_minor1, rotation1 = ellipse1
        x2, y2, semi_major2, semi_minor2, rotation2 = ellipse2
        
        # Generate point sets for both ellipses
        points_ellipse1 = generate_ellipse_points(x1, y1, semi_major1, semi_minor1, rotation1)
        points_ellipse2 = generate_ellipse_points(x2, y2, semi_major2, semi_minor2, rotation2)
        
        # Convert the points to NumPy arrays for use with directed_hausdorff (since it's not differentiable)
        points_ellipse1_np = points_ellipse1.detach().cpu().numpy()
        points_ellipse2_np = points_ellipse2.detach().cpu().numpy()
        
        # Compute Hausdorff distance (both directions)
        forward_hausdorff = directed_hausdorff(points_ellipse1_np, points_ellipse2_np)[0]
        reverse_hausdorff = directed_hausdorff(points_ellipse2_np, points_ellipse1_np)[0]
        
        # Hausdorff distance is the maximum of the two directed distances
        hausdorff_distance = max(forward_hausdorff, reverse_hausdorff)
        
        # Return the Hausdorff distance as a tensor (for backpropagation compatibility)
        return torch.tensor(hausdorff_distance, requires_grad=True)
