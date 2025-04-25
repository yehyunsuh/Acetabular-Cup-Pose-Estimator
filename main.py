import os
import cv2
import torch
import argparse

from data_loader import data_loader
from method.Fitzgibbon_et_al import fitzgibbon_et_al
from parameter import parameter_circle, parameter_landmarks
from loss import CustomMSELoss, CustomHausdorffLoss
from transformation import rotation_matrices, rotate_coordinates, project_coordinates


def main(args):
    # Step 1: Observe ellipse E using landmarks S_P
    dataset_loader = data_loader(args)

    for idx, data in enumerate(dataset_loader):
        image, image_path, H, r, orig_h, orig_w, landmarks = data
        image, image_path, H, r, orig_h, orig_w, landmarks = image[0], image_path[0], H[0], r[0], orig_h[0].numpy().item(), orig_w[0].numpy().item(), landmarks[0]

        # Step 1-1: Get 2D landmarks S_P from the image
        S_P = landmarks

        # Step 1-2: Estimate observed ellipse E using 2D landmarks S_P
        E = fitzgibbon_et_al(S_P)
        print(f'Observed Ellipse E: x: {E[0]:.4f}, y: {E[1]:.4f}, major: {E[2]:.4f}, minor: {E[3]:.4f}, angle: {E[4]:.4f}')

        # Draw ellipse on the image
        img = cv2.imread(image_path)
        center = (
            int(E[0]) + orig_h,
            int(E[1]) + orig_w
        )
        axes = (int(E[2]), int(E[3]))
        angle = float(E[4])  # Make sure it's a Python float, not numpy.float32 or tensor
        cv2.ellipse(img, center, axes, angle+90, 0, 360, (0, 255, 0), 2)
        cv2.imwrite(f'./tmp/ellipse_{idx}.png', img)

        # Step 2
        # Initial variable setting: theta_hat_0, phi_hat_0, k_hat_0, l_hat_0, h_hat_0
        beta = r / E[2]  # beta: ratio between the radius of the acetabular cup and the major axis of the ellipse
        theta_hat_0, phi_hat_0, k_hat_0, l_hat_0, h_hat_0 = 25, 40, E[0] * beta, E[1] * beta, H * (1 - beta)
        if E[4] < 90:
            phi_hat_0 = -phi_hat_0

        # Create synthetic 3D landmark S_hat
        C = parameter_circle(r, h_hat_0)
        S = parameter_landmarks(args, r, h_hat_0)

        # Define each variable as a separate torch.nn.Parameter
        theta_hat_0_ = torch.nn.Parameter(torch.tensor(theta_hat_0, dtype=torch.float64))
        phi_hat_0_ = torch.nn.Parameter(torch.tensor(phi_hat_0, dtype=torch.float64))
        k_hat_0_ = torch.nn.Parameter(k_hat_0.clone().detach().requires_grad_(True))
        l_hat_0_ = torch.nn.Parameter(l_hat_0.clone().detach().requires_grad_(True))
        h_hat_0_ = torch.nn.Parameter(h_hat_0.clone().detach().requires_grad_(True))
        vars = [theta_hat_0_, phi_hat_0_, k_hat_0_, l_hat_0_, h_hat_0_]

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
            # Step 2-1: Rotate S_hat to get S_R_hat
            rotation_matrix_x = rotation_matrices(torch.deg2rad(vars[0]), 'x-axis')
            rotation_matrix_z = rotation_matrices(torch.deg2rad(vars[1]), 'z-axis')
            rotation_matrix = torch.matmul(rotation_matrix_z, rotation_matrix_x)
            extra_plane_translation = torch.stack([torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64), vars[4]])
            S_R_hat = rotate_coordinates(S, extra_plane_translation, rotation_matrix)
            S_R_hat = rotate_coordinates(C, extra_plane_translation, rotation_matrix)

            # Step 2-2: Translate S_R_hat to get S_T_hat
            in_plane_translation = torch.stack([vars[2], vars[3], torch.tensor(0.0, dtype=vars[2].dtype)], dim=0)
            S_T_hat = S_R_hat + in_plane_translation  
            C_T_hat = S_R_hat + in_plane_translation

            # Step 2-3: Project S_T_hat to get S_P_hat
            ratio_S = H / (H - S_T_hat[:, 2])
            S_P_hat = project_coordinates(ratio_S, S_T_hat)
            ratio_C = H / (H - C_T_hat[:, 2])
            C_P_hat = project_coordinates(ratio_C, C_T_hat)

            # Step 2-4: Estimate nominal projected ellipse E_hat using 2D landmarks S_P_hat
            E_hat = fitzgibbon_et_al(S_P_hat)

            # Step 3: Calculate error between the parameters of observed ellipse E and the nominal projected ellipse E_hat
            loss = loss_fn(E_hat, E)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            optimizer.zero_grad()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Vars = {vars[0].item():.2f}, {vars[1].item():.2f}, {vars[2].item():.2f}, {vars[3].item():.2f}, {vars[4].item():.2f} | patience: {patience}")

        break


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
        "--num_epochs", type=int, default=1000,
        help="Number of epochs for optimization"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate for optimization"
    )

    args = parser.parse_args()

    os.makedirs('tmp', exist_ok=True)
    
    main(args) 