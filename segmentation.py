"""
segmentation.py

Performs semantic segmentation of acetabular cup implants in fluoroscopic images
using a pretrained U-Net model. Includes inference and contour extraction for
shape fitting and downstream projection-based pose estimation.

The predicted segmentation mask is thresholded and upsampled to the original image
resolution. Extracted contours are centered and lifted to 3D coordinates.

Author: Yehyun Suh  
Date: 2025-04-27
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from segmentation_model import UNet
from visualization import visualize_segmentation_result


def test_model(args, model, device, data_loader, vis_dir, timestamp):
    """
    Perform inference on test images and extract 3D contour coordinates.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (nn.Module): Loaded segmentation model.
        device (str): Device to perform inference ('cuda' or 'cpu').
        data_loader (DataLoader): DataLoader providing input images.
        vis_dir (str): Directory to save visualization results.
        timestamp (str): Timestamp to distinguish different experiment runs.

    Returns:
        list: A list of tuples containing:
              (image_path, sdd, radius, original_height, original_width, contour_points_3D)
    """
    model.eval()
    row = []

    with torch.no_grad():
        for idx, (image, sdd, radius, orig_h, orig_w, image_path) in enumerate(tqdm(data_loader, desc="Validation")):
            image = image.to(device)
            pred = model(image)

            # Post-process prediction
            pred = torch.sigmoid(pred[0])
            scaled_pred = F.interpolate(
                pred.unsqueeze(0),
                size=(orig_h[0], orig_w[0]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            scaled_pred_np = scaled_pred.detach().cpu().numpy()
            scaled_pred_np = (scaled_pred_np > 0.5).astype(np.float32)
            scaled_pred_np = (scaled_pred_np * 255).clip(0, 255).astype(np.uint8)

            if scaled_pred_np.ndim == 3 and scaled_pred_np.shape[0] == 1:
                scaled_pred_np = scaled_pred_np.squeeze(0)

            # Extract contours
            _, binary = cv2.threshold(scaled_pred_np, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_reshaped = np.array(contours).reshape(-1, 2)

            # Center contours and lift to 3D
            contours_reshaped = contours_reshaped.astype(np.float64)
            contours_reshaped[:, 0] -= orig_h[0].item() / 2
            contours_reshaped[:, 1] -= orig_w[0].item() / 2
            contours_3d = np.concatenate([contours_reshaped, np.zeros((contours_reshaped.shape[0], 1))], axis=1)
            contours_3d = torch.tensor(contours_3d, dtype=torch.float64)

            row.append((image_path[0], sdd, radius, orig_h, orig_w, contours_3d))

            visualize_segmentation_result(vis_dir, timestamp, image_path[0], scaled_pred_np, contours)

    return row


def segment_ellipse(args, data_loader, vis_dir, timestamp):
    """
    Load the pretrained U-Net model and run segmentation on the provided data.

    Args:
        args (argparse.Namespace): Parsed command-line arguments including model config.
        data_loader (DataLoader): DataLoader for test images.
        vis_dir (str): Directory to save segmentation visualizations.
        timestamp (str): Timestamp for saving outputs distinctly.

    Returns:
        list: Output from `test_model` representing image-wise segmentation results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = UNet(device)

    # Load model weights
    weight_path = f"{args.model_dir}/{args.model_name}.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )
        print(f"✅ Model weights loaded from {weight_path}")
    else:
        raise FileNotFoundError(f"❌ Weight file not found: {weight_path}")

    model.to(device)
    print("✅ Model loaded successfully.")

    results = test_model(args, model, device, data_loader, vis_dir, timestamp)

    return results
