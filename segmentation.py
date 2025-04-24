import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from segmentation_model import UNet
from visualization import visualize_segmentation_result


def test_model(args, model, device, data_loader):
    model.eval()
    row = []

    with torch.no_grad():
        for idx, (image, sdd, radius, orig_h, orig_w, image_path) in enumerate(tqdm(data_loader, desc="Validation")):
        # for idx, (image, sdd, radius, orig_h, orig_w, image_path) in enumerate(data_loader):
        #     print(f"Processing image {idx + 1}/{len(data_loader)}: {image_path[0]}")
            image = image.to(device)
            pred = model(image)

            pred = torch.sigmoid(pred[0])
            upscaled_pred = F.interpolate(pred.unsqueeze(0), size=(orig_h[0], orig_w[0]), mode='bilinear', align_corners=False).squeeze(0)
            upscaled_pred_np = upscaled_pred.detach().cpu().numpy()
            upscaled_pred_np = (upscaled_pred_np > 0.5).astype(np.float32) * 255
            if upscaled_pred_np.shape[0] == 1:
                upscaled_pred_np = upscaled_pred_np.squeeze(0)
            upscaled_pred_np = (upscaled_pred_np * 255).clip(0, 255).astype(np.uint8)

            # Extract contours from the upscaled prediction
            _, binary = cv2.threshold(upscaled_pred_np, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_reshaped = np.array(contours).reshape(-1, 2)

            # Translate the center points to center of the image
            contours_reshaped[:, 0] = contours_reshaped[:, 0] - orig_h[0].numpy()/2
            contours_reshaped[:, 1] = contours_reshaped[:, 1] - orig_w[0].numpy()/2

            # Add 0 to the z-axis to convert to 3D coordinates
            contours_reshaped = np.concatenate([contours_reshaped, np.zeros((contours_reshaped.shape[0], 1))], axis=1)
            contours_reshaped = torch.tensor(contours_reshaped, dtype=torch.float64)

            row.append((image_path[0], sdd, radius, orig_h, orig_w, contours_reshaped))

            if idx == 0:
                visualize_segmentation_result(image_path[0], pred, upscaled_pred_np, contours)

    return row


def segment_ellipse(args, data_loader):
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

    results = test_model(args, model, device, data_loader)

    return results