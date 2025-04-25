"""
data_loader.py

Dataset and DataLoader utilities for reading hip implant images,
manually annotated landmarks, or automatically segmented ellipses.

This module supports:
- Loading manual landmark coordinates for each image
- Loading images for automatic segmentation
- Loading segmentation results for ellipse fitting

Author: Yehyun Suh
Date: 2025-04-27
"""

import os
import csv
import cv2
import torch
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from segmentation import segment_ellipse
from visualization import visualize_manual_result


class SegmentationDataset_w_Landmark(Dataset):
    """
    Dataset for images with manually annotated landmarks (coordinates).
    
    Each sample contains:
      - Original RGB image
      - Associated landmark coordinates (centered around image center)
      - Imaging parameters like source-to-detector distance (SDD) and implant radius.
    """

    def __init__(self, csv_path, image_dir):
        """
        Initialize dataset by parsing annotation CSV.

        Args:
            csv_path (str): Path to annotation CSV file.
            image_dir (str): Directory containing input images.
        """
        self.image_dir = image_dir
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                image_name = row[0]
                image_width = int(row[1])
                image_height = int(row[2])
                sdd = float(row[3])
                radius = float(row[4])
                n_landmarks = int(row[5])
                coords = list(map(int, row[6:]))
                landmarks = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                self.samples.append((image_name, image_width, image_height, sdd, radius, landmarks))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, _, _, sdd, radius, landmarks = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        landmarks = np.array(landmarks, dtype=np.float32)
        landmarks[:, 0] -= orig_w / 2
        landmarks[:, 1] -= orig_h / 2

        landmarks = np.concatenate([landmarks, np.zeros((landmarks.shape[0], 1))], axis=1)
        landmarks = torch.tensor(landmarks, dtype=torch.float64)

        return image, image_path, sdd, radius, orig_h, orig_w, landmarks


class SegmentationDataset_wo_Landmark(Dataset):
    """
    Dataset for images without annotated landmarks.

    Each sample contains:
      - Original RGB image
      - Imaging parameters (SDD and implant radius).
    
    Landmarks will be obtained later automatically through segmentation.
    """

    def __init__(self, csv_path, image_dir):
        """
        Initialize dataset using CSV with image names and metadata.

        Args:
            csv_path (str): Path to annotation CSV file.
            image_dir (str): Directory containing input images.
        """
        self.image_dir = image_dir
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                image_name = row[0]
                sdd = float(row[1])
                radius = float(row[2])
                self.samples.append((image_name, sdd, radius))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, sdd, radius = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        transform = A.Compose([
            A.PadIfNeeded(min_height=max(orig_h, orig_w), min_width=max(orig_h, orig_w), border_mode=cv2.BORDER_CONSTANT),
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        image = transform(image=image)['image']

        return image, sdd, radius, orig_h, orig_w, image_path


class SegmentationDataset_wo_Landmark_after_segmentation(Dataset):
    """
    Dataset for images after automatic segmentation-based landmark extraction.

    Each sample contains:
      - Original RGB image
      - Imaging parameters (SDD and implant radius)
      - Fitted ellipse parameters or segmented contour coordinates.
    """

    def __init__(self, image_dir, results):
        """
        Args:
            image_dir (str): Directory containing input images.
            results (list): Output list from segmentation model.
        """
        self.image_dir = image_dir
        self.samples = results

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, sdd, radius, orig_h, orig_w, contours_reshaped = self.samples[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, image_path, sdd[0], radius[0], orig_h, orig_w, contours_reshaped


def data_loader(args):
    """
    Construct and return a dataloader for either manual or automated landmark sources.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        DataLoader: Final PyTorch dataloader.
    """
    csv_path = os.path.join(args.label_dir, args.label_csv)

    if args.landmark_source == "manual":
        print("===== Using manual landmark source =====")
        dataset = SegmentationDataset_w_Landmark(
            csv_path=csv_path,
            image_dir=args.image_dir
        )

    elif args.landmark_source == "auto":
        print("===== Using auto landmark source =====")
        pre_seg_dataset = SegmentationDataset_wo_Landmark(
            csv_path=csv_path,
            image_dir=args.image_dir
        )
        pre_seg_loader = DataLoader(pre_seg_dataset, batch_size=1, shuffle=False)
        results = segment_ellipse(args, pre_seg_loader)

        dataset = SegmentationDataset_wo_Landmark_after_segmentation(
            image_dir=args.image_dir,
            results=results
        )

    else:
        raise ValueError("Invalid landmark_source. Choose from ['manual', 'auto'].")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Dataset size: {len(loader.dataset)}")

    return loader
