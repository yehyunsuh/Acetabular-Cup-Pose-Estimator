import torch.nn as nn
import segmentation_models_pytorch as smp


def UNet(device):
    print("---------- Loading Model ----------")

    model = smp.Unet(
        encoder_name='resnet101',
        encoder_weights='imagenet',
        classes=1,
        activation='sigmoid',  # This will be removed below
    )

    print("---------- Model Loaded ----------")

    # Remove the final sigmoid activation from the segmentation head
    # so loss functions like BCEWithLogitsLoss can be used instead
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])

    return model.to(device)