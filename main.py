"""
main.py

Optimization pipeline for estimating acetabular cup pose parameters from fluoroscopy images.
This script fits synthetic 3D landmarks onto observed 2D landmarks by minimizing the error between
observed and projected ellipses using a differentiable optimization approach.

If you find this code useful, please cite the following paper:
@misc{suh20252d3dregistrationacetabularhip,
      title={2D/3D Registration of Acetabular Hip Implants Under Perspective Projection and Fully Differentiable Ellipse Fitting}, 
      author={Yehyun Suh and J. Ryan Martin and Daniel Moyer},
      year={2025},
      eprint={2503.07763},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.07763}, 
}

Author: Yehyun Suh  
Date: 2025-04-27
"""

import os
import csv
import torch
import argparse
import numpy as np

from datetime import datetime

from data_loader import data_loader
from Fitzgibbon_et_al import fitzgibbon_et_al
from Liaw_et_al import liaw_et_al
from parameter import parameter_circle, parameter_landmarks
from loss import CustomMSELoss, CustomHausdorffLoss
from transformation import rotation_matrices, rotate_coordinates, project_coordinates
from log import print_process
from visualization import visualize_observed_ellipse_E, visualize_optimized_ellipse_E_hat


def main(args):
    """
    Main pipeline to estimate acetabular cup pose.

    Steps:
        1. Load landmarks or generate them from segmentation.
        2. Fit observed ellipse E from projected landmarks.
        3. Initialize synthetic 3D landmarks and pose parameters.
        4. Optimize pose parameters by minimizing ellipse fitting loss.
        5. Save optimized results and generate visualizations.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{args.vis_dir}/{timestamp}", exist_ok=True)

    # Result CSV setup
    result_csv_name = f"optimization_result_{timestamp}.csv"
    csv_file = os.path.join(args.result_dir, result_csv_name)
    write_header = not os.path.exists(csv_file)

    csv_columns = [
        'image_name', 'image_width', 'image_height', 'source_to_detection_distance', 'radius',
        'anteversion', 'inclination', 'translation_x', 'translation_y', 'object_to_detection_distance',
        'opt_anteversion', 'opt_inclination', 'opt_translation_x', 'opt_translation_y', 'opt_object_to_detection_distance',
        'ellipse_center_x', 'ellipse_center_y', 'ellipse_major', 'ellipse_minor', 'ellipse_angle',
        'ellipse_opt_center_x', 'ellipse_opt_center_y', 'ellipse_opt_major', 'ellipse_opt_minor', 'ellipse_opt_angle',
        'loss', 'hausdorff_distance', 'anteversion_liaw', 'inclination_parallel'
    ]

    # Load dataset
    dataset_loader = data_loader(args, timestamp)

    for idx, data in enumerate(dataset_loader):
        print('\n\n==========================')
        image, image_path, H, r, orig_h, orig_w, landmarks = data
        image = image[0]
        image_path = image_path[0]
        H = H[0]
        r = r[0]
        orig_h = orig_h[0].numpy().item()
        orig_w = orig_w[0].numpy().item()
        landmarks = landmarks[0]
        print(f"Processing image {idx + 1}/{len(dataset_loader)}: {image_path}")

        # Step 1: Observed ellipse E from 2D landmarks
        S_P = landmarks
        E = fitzgibbon_et_al(S_P)
        print(f"Observed Ellipse E: x={E[0]:.4f}, y={E[1]:.4f}, major={E[2]:.4f}, minor={E[3]:.4f}, angle={E[4]:.4f}")

        visualize_observed_ellipse_E(args.vis_dir, timestamp, image_path, E, orig_h, orig_w)

        # Step 2: Initial pose parameters
        beta = r / E[2]
        theta_hat_0, phi_hat_0 = 25, 40
        k_hat_0, l_hat_0, h_hat_0 = E[0] * beta, E[1] * beta, H * (1 - beta)

        if E[4] < 90:
            phi_hat_0 = -phi_hat_0

        # Generate synthetic 3D landmarks
        C = parameter_circle(r, h_hat_0)
        S = parameter_landmarks(args, r, h_hat_0)

        # Parameters to optimize
        vars = [
            torch.nn.Parameter(torch.tensor(theta_hat_0, dtype=torch.float64)),
            torch.nn.Parameter(torch.tensor(phi_hat_0, dtype=torch.float64)),
            torch.nn.Parameter(k_hat_0.clone().detach().requires_grad_(True)),
            torch.nn.Parameter(l_hat_0.clone().detach().requires_grad_(True)),
            torch.nn.Parameter(h_hat_0.clone().detach().requires_grad_(True)),
        ]

        loss_fn = CustomMSELoss()
        optimizer = torch.optim.SGD([
            {'params': vars[:2], 'lr': args.lr},
            {'params': vars[2:4], 'lr': args.lr * 0.01},
            {'params': vars[4:], 'lr': args.lr * 0.01}
        ], momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100)
        comparison_fn = CustomHausdorffLoss()
        best_loss, prev_loss = float('inf'), float('inf')
        patience = 0

        for epoch in range(args.num_epochs):
            # Step 3: Forward pass
            rotation_matrix_x = rotation_matrices(torch.deg2rad(vars[0]), 'x-axis')
            rotation_matrix_z = rotation_matrices(torch.deg2rad(vars[1]), 'z-axis')
            rotation_matrix = torch.matmul(rotation_matrix_z, rotation_matrix_x)

            extra_plane_translation = torch.stack([
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
                vars[4]
            ])

            S_R_hat = rotate_coordinates(S, extra_plane_translation, rotation_matrix)
            C_R_hat = rotate_coordinates(C, extra_plane_translation, rotation_matrix)

            in_plane_translation = torch.stack([
                vars[2], vars[3], torch.tensor(0.0, dtype=torch.float64)
            ])
            S_T_hat = S_R_hat + in_plane_translation
            C_T_hat = C_R_hat + in_plane_translation

            ratio_S = H / (H - S_T_hat[:, 2])
            S_P_hat = project_coordinates(ratio_S, S_T_hat)

            ratio_C = H / (H - C_T_hat[:, 2])
            C_P_hat = project_coordinates(ratio_C, C_T_hat)

            E_hat = fitzgibbon_et_al(S_P_hat)

            # Step 4: Compute loss and update
            loss = loss_fn(E_hat, E)
            hd_diff = comparison_fn(E_hat, E)
            loss.backward()

            if epoch == 0 or epoch % 100 == 0:
                print_process(args, epoch, loss, vars, E, E_hat, hd_diff, patience)

            if loss.item() < best_loss:
                if abs(best_loss - loss.item()) > 1e-6:
                    patience = 0
                best_loss = loss.item()
                best_hd_diff = hd_diff
                best_vars = [v.clone().detach() for v in vars]
                best_S_P_hat = S_P_hat.clone().detach()
                best_C_P_hat = C_P_hat.clone().detach()
                best_E_hat = E_hat.clone().detach()

            if prev_loss - loss.item() < 1e-3:
                patience += 1
                if patience > args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if loss.item() < 1e-3 or (epoch > 5000 and loss.item() < 0.5):
                print(f"Optimization converged at epoch {epoch}")
                break

            optimizer.step()

            vars[0].data = torch.clamp(vars[0].data, 0, 90)
            if E[4] >= 90:
                vars[1].data = torch.clamp(vars[1].data, 0, 90)
            else:
                vars[1].data = torch.clamp(vars[1].data, -90, 0)

            scheduler.step(loss)
            optimizer.zero_grad()
            prev_loss = loss.item()

        # Step 5: Post-processing
        theta_opt = best_vars[0].item()
        phi_opt = best_vars[1].item()
        k_opt = best_vars[2].item()
        l_opt = best_vars[3].item()
        h_opt = best_vars[4].item()

        visualize_optimized_ellipse_E_hat(args.vis_dir, timestamp, image_path, best_E_hat, orig_h, orig_w)

        liaw_et_al_anteversion = liaw_et_al(E)
        opt_ellipse_angle = best_E_hat[4].item()
        parallel_inclination = 90 - (opt_ellipse_angle if opt_ellipse_angle <= 90 else 180 - opt_ellipse_angle)

        result_row = {
            'image_name': os.path.basename(image_path),
            'image_width': orig_w,
            'image_height': orig_h,
            'source_to_detection_distance': H.item(),
            'radius': r.item(),
            'anteversion': np.NaN,
            'inclination': np.NaN,
            'translation_x': np.NaN,
            'translation_y': np.NaN,
            'object_to_detection_distance': np.NaN,
            'opt_anteversion': theta_opt,
            'opt_inclination': phi_opt,
            'opt_translation_x': k_opt,
            'opt_translation_y': l_opt,
            'opt_object_to_detection_distance': h_opt,
            'ellipse_center_x': E[0].item(),
            'ellipse_center_y': E[1].item(),
            'ellipse_major': E[2].item(),
            'ellipse_minor': E[3].item(),
            'ellipse_angle': E[4].item(),
            'ellipse_opt_center_x': best_E_hat[0].item(),
            'ellipse_opt_center_y': best_E_hat[1].item(),
            'ellipse_opt_major': best_E_hat[2].item(),
            'ellipse_opt_minor': best_E_hat[3].item(),
            'ellipse_opt_angle': best_E_hat[4].item(),
            'loss': best_loss,
            'hausdorff_distance': best_hd_diff.item(),
            'anteversion_liaw': liaw_et_al_anteversion.item(),
            'inclination_parallel': parallel_inclination,
        }

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(result_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acetabular Cup Pose Estimator")

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for reproducibility"
    )

    # Landmark option
    parser.add_argument(
        "--landmark_source", type=str, default="manual", required=True,
        help="Source of landmarks: 'manual' or 'auto'"
    )

    # Data
    parser.add_argument(
        "--image_dir", type=str, default="./data/images",
        help="Directory containing fluoroscopy images with acetabular cup"
    )
    parser.add_argument(
        "--label_dir", type=str, default="./data/labels",
        help="Directory containing labels for the images"
    )
    parser.add_argument(
        "--label_csv", type=str, default="label.csv",
        help="CSV file containing labels for the images"
    )

    # Segmentation Model
    parser.add_argument(
        "--model_dir", type=str, default="./data/model_weight",
        help="Directory to save the trained segmentation model"
    )
    parser.add_argument(
        "--model_name", type=str, default="segmentation",
        help="Name of the segmentation model to use"
    )

    # Synthetic Landmarks
    parser.add_argument(
        "--n_landmarks", type=int, default=100,
        help="Number of synthetic landmarks to generate"
    )

    # Optimization
    parser.add_argument(
        "--num_epochs", type=int, default=20000,
        help="Number of epochs for optimization"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate for optimization"
    )
    parser.add_argument(
        "--patience", type=int, default=1000,
        help="Patience for early stopping"
    )

    args = parser.parse_args()

    if args.landmark_source == 'auto':
        args.vis_dir = './visualization_auto'
        args.result_dir = './results_auto'
    elif args.landmark_source == 'manual':
        args.vis_dir = './visualization_manual'
        args.result_dir = './results_manual'
    else:
        raise ValueError("landmark_source must be either 'manual' or 'auto'")

    os.makedirs(args.result_dir, exist_ok=True)
    
    main(args) 