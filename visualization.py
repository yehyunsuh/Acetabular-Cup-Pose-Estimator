import cv2
import numpy as np


def visualize_segmentation_result(image_path, pred, upscaled_pred_np, contours):
    pred_np = pred.squeeze().cpu().numpy()
    pred_thresholded = (pred_np > 0.5).astype(np.float32) * 255
    cv2.imwrite('./tmp/segmentation_output.png', pred_thresholded)

    cv2.imwrite('./tmp/segmentation_output_upscaled.png', upscaled_pred_np)

    # Overlay the segmentation mask on the original image
    org_img = cv2.imread(image_path)
    upsclaed_pred_np_bgr = cv2.cvtColor(upscaled_pred_np, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(org_img, 0.7, upsclaed_pred_np_bgr, 0.3, 0)
    cv2.imwrite('./tmp/segmentation_overlay.png', overlay)

    # Draw contours on the original image (make a copy to keep original clean)
    contour_img = org_img.copy()

    # cv2.drawContours modifies the image in-place
    cv2.drawContours(
        image=contour_img,             # image to draw on
        contours=contours,            # list of contours
        contourIdx=-1,                # draw all contours
        color=(0, 255, 0),            # green color
        thickness=2                   # line thickness
    )

    # Save the image with contours
    cv2.imwrite("./tmp/segmentation_with_contours.png", contour_img)


def visualize_manual_result(image, landmarks, idx):
    # draw landmarks on the image
    for landmark in landmarks:
        cv2.circle(image, (int(landmark[0]), int(landmark[1])), 5, (0, 255, 0), -1)
    
    # using the landmarks, calculate the fit ellipse
    ellipse = cv2.fitEllipse(landmarks)
    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
    cv2.imwrite(f'./tmp/landmarks_{idx}.png', image)