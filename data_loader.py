"""
data_loader.py

Dataset and DataLoader utilities for reading hip implant images,
manually annotated landmarks, or automatically segmented ellipses.

Supports:
- Manual landmark coordinate loading
- Automatic landmark extraction via segmentation
- Post-segmentation ellipse fitting for landmark estimation

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


class SegmentationDatasetWithLandmark(Dataset):
    """
    Dataset class for images with manually annotated landmarks.

    Each sample contains:
        - RGB image
        - Landmark coordinates (centered to image center)
        - Imaging parameters (SDD, implant radius)
    """

    def __init__(self, csv_path, image_dir, vis_dir, timestamp):
        """
        Initialize dataset by parsing the manual annotation CSV.

        Args:
            csv_path (str): Path to the landmark CSV file.
            image_dir (str): Directory containing input images.
            vis_dir (str): Directory to save landmark visualizations.
            timestamp (str): Timestamp string to organize outputs.
        """
        self.image_dir = image_dir
        self.vis_dir = vis_dir
        self.timestamp = timestamp
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
        visualize_manual_result(self.vis_dir, self.timestamp, image_path, image, landmarks)

        landmarks[:, 0] -= orig_w / 2
        landmarks[:, 1] -= orig_h / 2
        landmarks = np.concatenate([landmarks, np.zeros((landmarks.shape[0], 1))], axis=1)

        landmarks = torch.tensor(landmarks, dtype=torch.float64)

        return image, image_path, sdd, radius, orig_h, orig_w, landmarks


class SegmentationDatasetWithoutLandmark(Dataset):
    """
    Dataset class for images without manually annotated landmarks.

    Each sample contains:
        - RGB image
        - Imaging parameters (SDD, implant radius)
    
    Landmarks will be generated later through segmentation.
    """

    def __init__(self, csv_path, image_dir):
        """
        Initialize dataset using CSV with image metadata.

        Args:
            csv_path (str): Path to CSV file containing image names and metadata.
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


class SegmentationDatasetPostSegmentation(Dataset):
    """
    Dataset class for images after automatic landmark extraction (post segmentation).

    Each sample contains:
        - RGB image
        - Imaging parameters (SDD, implant radius)
        - Extracted ellipse parameters or segmented contours
    """

    def __init__(self, image_dir, results, vis_dir, timestamp):
        """
        Initialize dataset using segmentation results.

        Args:
            image_dir (str): Directory containing input images.
            results (list): List of segmentation model outputs.
            vis_dir (str): Directory for saving visualization results.
            timestamp (str): Timestamp string for organized saving.
        """
        self.image_dir = image_dir
        self.vis_dir = vis_dir
        self.timestamp = timestamp
        self.samples = results

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, sdd, radius, orig_h, orig_w, contours_reshaped = self.samples[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, image_path, sdd[0], radius[0], orig_h, orig_w, contours_reshaped


def data_loader(args, timestamp):
    """
    Constructs the PyTorch dataloader for training or evaluation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        timestamp (str): Timestamp used for organizing outputs.

    Returns:
        DataLoader: Configured PyTorch DataLoader instance.
    """
    csv_path = os.path.join(args.label_dir, args.label_csv)

    if args.landmark_source == "manual":
        print("===== Using manual landmark source =====")
        dataset = SegmentationDatasetWithLandmark(
            csv_path=csv_path,
            image_dir=args.image_dir,
            vis_dir=args.vis_dir,
            timestamp=timestamp
        )

    elif args.landmark_source == "auto":
        print("===== Using auto landmark source =====")
        pre_seg_dataset = SegmentationDatasetWithoutLandmark(
            csv_path=csv_path,
            image_dir=args.image_dir
        )
        pre_seg_loader = DataLoader(pre_seg_dataset, batch_size=1, shuffle=False)

        results = segment_ellipse(args, pre_seg_loader, args.vis_dir, timestamp)

        dataset = SegmentationDatasetPostSegmentation(
            image_dir=args.image_dir,
            results=results,
            vis_dir=args.vis_dir,
            timestamp=timestamp
        )

    else:
        raise ValueError("Invalid landmark_source. Choose from ['manual', 'auto'].")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Dataset size: {len(loader.dataset)}")

    return loader
