import os
import cv2
import argparse

from data_loader import data_loader
from method.Fitzgibbon_et_al import fitzgibbon_et_al


def main(args):
    # Step 1: Observe ellipse E using landmarks S_P
    dataset_loader = data_loader(args)

    for idx, data in enumerate(dataset_loader):
        image, image_path, sdd, radius, orig_h, orig_w, contours_reshaped = data

        # Step 1-1: Get 2D landmarks S_P from the image
        S_P = contours_reshaped[0]

        # Step 1-2: Estimate observed ellipse E using 2D landmarks S_P
        E = fitzgibbon_et_al(S_P)
        print(f'Observed Ellipse E: x: {E[0]:.4f}, y: {E[1]:.4f}, major: {E[2]:.4f}, minor: {E[3]:.4f}, angle: {E[4]:.4f}')

        # Draw ellipse on the image
        img = cv2.imread(image_path[0])
        center = (
            int(E[0]) + int((orig_h[0].numpy() / 2).item()),
            int(E[1]) + int((orig_w[0].numpy() / 2).item())
        )
        axes = (int(E[2]), int(E[3]))
        angle = float(E[4])  # Make sure it's a Python float, not numpy.float32 or tensor
        cv2.ellipse(img, center, axes, angle, 0, 360, (0, 255, 0), 2)
        cv2.imwrite(f'./tmp/ellipse_{idx}.png', img)

        # Step 2: Create synthetic 3D landmark S_hat

        # Step 2-1: Rotate S_hat to get S_hat_R

        # Step 2-2: Translate S_hat_R to get S_hat_T

        # Step 2-3: Project S_hat_T to get S_hat_P

        # Step 2-4: Estimate nominal projected ellipse E_hat using 2D landmarks S_hat_P

        # Step 3: Calculate error between the parameters of observed ellipse E and the nominal projected ellipse E_hat

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

    args = parser.parse_args()

    os.makedirs('tmp', exist_ok=True)
    
    main(args) 