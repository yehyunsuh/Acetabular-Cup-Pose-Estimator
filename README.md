# Acetabular-Cup-Pose-Estimator
<img width="952" alt="Screenshot 2025-04-28 at 12 08 46 PM" src="https://github.com/user-attachments/assets/fb011ae2-0a28-4ebb-ade2-7fe4d26167a0" />

This project estimates 3D pose parameters of acetabular hip implants from a single 2D fluoroscopy image.

It supports both manual landmark annotations and automatic landmark extraction via segmentation, using a fully differentiable ellipse fitting and optimization pipeline.

- For **annotating** landmarks: [Anatomical-Landmark-Annotator](https://github.com/yehyunsuh/Anatomical-Landmark-Annotator)
---

## 📂 Directory Structure
```
# Acetabular-Cup-Pose-Estimator
│
├── data/                           # Data directory
│   ├── images/                     # Fluoroscopy input images (with acetabular cup implants)
│   ├── labels/                     # CSV files with or without landmark coordinates
│   └── model_weight/               # Pretrained segmentation model weights (optional to use)
│
├── results_auto/                   # Optimization results (landmarks extracted automatically by segmentation)
├── results_manual/                 # Optimization results (manually annotated landmarks)
├── visualization_auto/             # Visualizations based on automatically extracted landmarks
├── visualization_manual/           # Visualizations based on manually annotated landmarks
│
├── .gitignore                      # List of files and folders to ignore in Git version control
├── data_loader.py                  # Dataset and DataLoader for handling images and landmarks
├── Fitzgibbon_et_al.py             # Differentiable ellipse fitting method
├── Liaw_et_al.py                   # Baseline method for estimating anteversion from ellipse
├── log.py                          # Utilities for logging training/optimization progress
├── loss.py                         # Custom loss functions (e.g., ellipse fitting loss, Hausdorff loss)
├── main.py                         # Main optimization pipeline for 2D/3D registration of hip implants
├── parameter.py                    # Functions to generate synthetic 3D landmarks and circles
├── README.md                       # You're here!
├── requirements.txt                # Python package dependencies
├── segmentation_model.py           # U-Net segmentation model definition
├── segmentation.py                 # Inference code for performing segmentation and contour extraction
├── transformation.py               # Functions for rotation, projection, and coordinate transformations
└── visualization.py                # Visualization utilities for images, landmarks, segmentation masks, ellipses

```

---

## 🚀 Getting Started

### 1. Install Dependencies

We recommend using a virtual environment:

```bash
git clone https://github.com/yehyunsuh/Anatomical-Landmark-Detector-Testing.git
cd Anatomical-Landmark-Detector-Testing
conda create -n detector python=3.10 -y
conda activate detector
pip3 install -r requirements.txt
```

### 2. Prepare Your Data

Place your training images under:
```
data/images/
```

Place your annotation CSV file under:
```
data/labels/label.csv
```

Format of the CSV if you have manual landmarks:
```
image_name,image_width,image_height,source_to_detection_distance,radius,n_landmarks,landmark_1_x,landmark_1_y,...
image1.jpg,1098,1120,1040,22,3,123,145,...
image2.jpg,1400,1210,1040,26,3,108,132,...
```

Format of the CSV if you do not have manual landmarks:
```
image_name,source_to_detection_distance,radius
image1.jpg,1040,22
image2.jpg,1040,26
```

### 3. Run Pose Estimator
With manual annotation:
```bash
python main.py --landmark_source manual
```

Without manual annotation:
```bash
python main.py --landmark_source auto
```

You can also customize:
```bash
python main.py \
    --landmark_source manual \
    --image_dir ./data/images \
    --label_dir ./data/labels \
    --label_csv label.csv \
    --model_dir ./data/model_weight \
    --model_name segmentation \
    --n_landmarks 100 \
    --num_epochs 20000 \
    --lr 0.01 \
    --patience 1000
```

### 🧩 Argument Reference

| Argument                  | Description                                                    | Default                      |
|----------------------------|----------------------------------------------------------------|-------------------------------|
| `--landmark_source`        | Source of landmarks: `manual` or `auto` (REQUIRED)             | manual                       |
| `--image_dir`              | Path to the directory containing fluoroscopy images           | `./data/images`              |
| `--label_dir`              | Path to the directory containing annotation CSVs              | `./data/labels`              |
| `--label_csv`              | CSV filename containing image metadata and landmarks          | `label.csv`                  |
| `--model_dir`              | Path to the directory containing model weights (for auto mode) | `./data/model_weight`        |
| `--model_name`             | Name of the model weight file (without extension)             | `segmentation`               |
| `--n_landmarks`            | Number of synthetic landmarks generated (REQUIRED)            | 100                          |
| `--num_epochs`             | Number of optimization epochs                                 | 20000                        |
| `--lr`                     | Learning rate for optimization                                | 0.01                         |
| `--patience`               | Early stopping patience                                       | 1000                         |
| `--seed`                   | Random seed for reproducibility                               | 42                           |

## Citation
If you find this helpful, please cite this [paper](https://arxiv.org/abs/2503.07763):
```
@misc{suh20252d3dregistrationacetabularhip,
      title={2D/3D Registration of Acetabular Hip Implants Under Perspective Projection and Fully Differentiable Ellipse Fitting}, 
      author={Yehyun Suh and J. Ryan Martin and Daniel Moyer},
      year={2025},
      eprint={2503.07763},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.07763}, 
}
```
