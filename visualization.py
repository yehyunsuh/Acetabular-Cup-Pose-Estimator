"""
visualization.py

Visualization utilities for evaluating segmentation and ellipse fitting results.
This module provides image overlay functions for:
- Segmentation prediction masks
- Manual landmark annotations
- Fitted ellipses on the original input images

Saved visual outputs are written to the `./tmp/` directory for inspection.

Author: Yehyun Suh  
Date: 2025-04-27
"""

import cv2
import numpy as np


def visualize_segmentation_result(vis_dir, timestamp, image_path, scaled_pred, contours):
    """
    Save visualizations of segmentation results, including:
    - Raw thresholded prediction
    - Upscaled segmentation output
    - Overlay of mask on original image
    - Contours drawn on original image

    Args:
        image_path (str): Path to the original image file.
        pred (Tensor): Predicted segmentation mask tensor [1, H, W].
        upscaled_pred_np (ndarray): Upscaled prediction in [H, W] format.
        contours (list): List of contours detected from the predicted mask.
    """ 
    # Overlay prediction on the original image
    org_img = cv2.imread(image_path)
    upsclaed_pred_np_bgr = cv2.cvtColor(scaled_pred, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(org_img, 0.7, upsclaed_pred_np_bgr, 0.3, 0)

    image_name = image_path.split('/')[-1].split('.')[0]
    cv2.imwrite(f'./{vis_dir}/{timestamp}/{image_name}_segmentation.png', overlay)

    # Draw contours on the image
    contour_img = org_img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), thickness=2)
    cv2.imwrite(f'./{vis_dir}/{timestamp}/{image_name}_contours.png', contour_img)


def visualize_manual_result(vis_dir, timestamp, image_path, image, landmarks):
    """
    Draw ground truth landmarks and fitted ellipse on an image.

    Args:
        image (ndarray): Input RGB image in numpy format.
        landmarks (ndarray): Array of (x, y) landmark coordinates.
        idx (int): Image index used in the filename.
    """
    for landmark in landmarks:
        cv2.circle(image, (int(landmark[0]), int(landmark[1])), 5, (0, 255, 0), -1)
    
    ellipse = cv2.fitEllipse(landmarks)
    cv2.ellipse(image, ellipse, (0, 255, 0), 2)

    image_name = image_path.split('/')[-1].split('.')[0]
    cv2.imwrite(f'./{vis_dir}/{timestamp}/{image_name}_landmarks_S_P.png', image)


def visualize_observed_ellipse_E(vis_dir, timestamp, image_path, E, orig_h, orig_w):
    """
    Draw the observed ellipse (E) estimated from manual or predicted landmarks.

    Args:
        image_path (str): Path to the image on which the ellipse is drawn.
        E (Tensor or ndarray): Ellipse parameters [center_x, center_y, major, minor, angle].
        orig_h (int): Original image height before padding/resizing.
        orig_w (int): Original image width before padding/resizing.
        idx (int): Image index used in the filename.
    """
    img = cv2.imread(image_path)
    center = (int(E[0] + orig_h / 2), int(E[1] + orig_w / 2))
    axes = (int(E[2]), int(E[3]))
    angle = float(E[4])
    cv2.ellipse(img, center, axes, angle + 90, 0, 360, (0, 255, 0), 2)

    image_name = image_path.split('/')[-1].split('.')[0]
    cv2.imwrite(f'./{vis_dir}/{timestamp}/{image_name}_observed_ellipse_E.png', img)


def visualize_optimized_ellipse_E_hat(vis_dir, timestamp, image_path, E_hat, orig_h, orig_w):
    """
    Draw the optimized ellipse (Ê) estimated through parameter optimization.

    Args:
        image_path (str): Path to the image on which the ellipse is drawn.
        E_hat (Tensor or ndarray): Optimized ellipse parameters [center_x, center_y, major, minor, angle].
        orig_h (int): Original image height before padding/resizing.
        orig_w (int): Original image width before padding/resizing.
        idx (int): Image index used in the filename.
    """
    img = cv2.imread(image_path)
    center = (int(E_hat[0] + orig_h / 2), int(E_hat[1] + orig_w / 2))
    axes = (int(E_hat[2]), int(E_hat[3]))
    angle = float(E_hat[4])
    cv2.ellipse(img, center, axes, angle + 90, 0, 360, (255, 0, 0), 2)

    image_name = image_path.split('/')[-1].split('.')[0]
    cv2.imwrite(f'./{vis_dir}/{timestamp}/{image_name}_optimized_ellipse_E_hat.png', img)