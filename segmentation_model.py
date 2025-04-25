"""
model.py

Defines the U-Net model architecture for binary segmentation using the 
Segmentation Models PyTorch (SMP) library. The encoder is based on a 
pretrained ResNet-101 backbone. The final sigmoid activation is removed 
to allow compatibility with loss functions like BCEWithLogitsLoss.

Author: Yehyun Suh  
Date: 2025-04-27
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


def UNet(device):
    """
    Instantiate a U-Net model with a ResNet-101 encoder.

    Args:
        device (str): Device to move the model to ('cuda' or 'cpu').

    Returns:
        nn.Module: U-Net model with final activation removed.
    """
    print("---------- Loading Model ----------")

    model = smp.Unet(
        encoder_name='resnet101',
        encoder_weights='imagenet',
        classes=1,
        activation='sigmoid',  # Temporarily included, removed below
    )

    print("---------- Model Loaded ----------")

    # Remove the final sigmoid activation to use raw logits
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])

    return model.to(device)
