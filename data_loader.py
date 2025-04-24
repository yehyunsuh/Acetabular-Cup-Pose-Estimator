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
    Custom dataset for anatomical landmark segmentation.
    Each sample includes an RGB image and a multi-channel binary mask,
    where each channel corresponds to a dilated landmark point.
    """

    def __init__(self, csv_path, image_dir):
        """
        Initializes the dataset by parsing CSV annotations and storing image/landmark paths.

        Args:
            csv_path (str): Path to the annotation CSV file.
            image_dir (str): Directory containing input images.
        """
        self.image_dir = image_dir
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                image_name = row[0]
                image_width = int(row[1])
                image_height = int(row[2])
                sdd = float(row[3])  # source to detector distance
                radius = float(row[4])  # radius of the acetabular cup
                n_landmarks = int(row[5])
                coords = list(map(int, row[6:]))
                landmarks = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                self.samples.append((image_name, image_width, image_height, sdd, radius, landmarks))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, _, _, sdd, radius, landmarks = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load and convert image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        landmarks = np.array(landmarks, dtype=np.float32)

        # visualize_manual_result(image, landmarks, idx)

        # Translate the center points to center of the image
        landmarks[:, 0] = landmarks[:, 0] - orig_w / 2
        landmarks[:, 1] = landmarks[:, 1] - orig_h / 2
        
        # Add 0 to the z-axis to convert to 3D coordinates
        landmarks = np.concatenate([landmarks, np.zeros((landmarks.shape[0], 1))], axis=1)
        landmarks = torch.tensor(landmarks, dtype=torch.float64)

        return image, image_path, sdd, radius, orig_h, orig_w, landmarks
    

class SegmentationDataset_wo_Landmark(Dataset):
    """
    Custom dataset for anatomical landmark segmentation.
    Each sample includes an RGB image and a multi-channel binary mask,
    where each channel corresponds to a dilated landmark point.
    """

    def __init__(self, csv_path, image_dir):
        """
        Initializes the dataset by parsing CSV annotations and storing image/landmark paths.

        Args:
            csv_path (str): Path to the annotation CSV file.
            image_dir (str): Directory containing input images.
            n_landmarks (int): Number of landmarks per image.
            dilation_iters (int): Number of dilation iterations for landmark masks.
        """
        self.image_dir = image_dir
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                image_name = row[0]
                sdd = float(row[1])  # source to detector distance
                radius = float(row[2])  # radius of the acetabular cup
                self.samples.append((image_name, sdd, radius))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, sdd, radius = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load and convert image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        max_side = max(orig_h, orig_w)
        transform = A.Compose([
            A.PadIfNeeded(
                min_height=max_side, min_width=max_side,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        transformed = transform(image=image)
        image = transformed["image"]

        return image, sdd, radius, orig_h, orig_w, image_path
    

class SegmentationDataset_wo_Landmark_after_segmentation(Dataset):
    """
    Custom dataset for anatomical landmark segmentation.
    Each sample includes an RGB image and a multi-channel binary mask,
    where each channel corresponds to a dilated landmark point.
    """

    def __init__(self, csv_path, image_dir, results):
        """
        Initializes the dataset by parsing CSV annotations and storing image/landmark paths.

        Args:
            csv_path (str): Path to the annotation CSV file.
            image_dir (str): Directory containing input images.
            results (list): List of segmentation results.
        """
        self.image_dir = image_dir
        self.samples = results

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, sdd, radius, orig_h, orig_w, contours_reshaped = self.samples[idx]

        # Load and convert image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, image_path, sdd, radius, orig_h, orig_w, contours_reshaped


def data_loader(args):
    """
    Constructs and returns PyTorch dataloaders for training and validation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing all config.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Option 1: Labels from csv file
    if args.landmark_source == "manual":
        print("===== Using manual landmark source =====")
        dataset = SegmentationDataset_w_Landmark(
            csv_path=os.path.join(args.label_dir, args.label_csv),
            image_dir=args.image_dir,
        )
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Option 2: Labels from segmentation model
    elif args.landmark_source == "auto":
        print("===== Using auto landmark source =====")
        dataset = SegmentationDataset_wo_Landmark(
            csv_path=os.path.join(args.label_dir, args.label_csv),
            image_dir=args.image_dir,
        )
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Segment the images using the segmentation model
        results = segment_ellipse(args, data_loader)

        dataset = SegmentationDataset_wo_Landmark_after_segmentation(
            csv_path=os.path.join(args.label_dir, args.label_csv),
            image_dir=args.image_dir,
            results=results
        )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Dataset size: {len(data_loader.dataset)}")

    return data_loader