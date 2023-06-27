import torch
import torch.nn as nn

import model.UNet as UNet
import model.UNetWithGradCAM as UNetWithGradCAM
import xai.GradCAM as GradCAM

# Create an instance of the U-Net model
in_channels = 3
num_classes = 1
unet = UNet(in_channels, num_classes)

# Create an instance of the U-Net model with Grad-CAM
unet_with_gradcam = UNetWithGradCAM(unet)

# Input image, replace with your code
input_image = torch.randn(
    1, in_channels, 64, 64
)  # Example input image with size 64x64 and 3 channels

# Perform inference with the U-Net model and obtain the predicted output and intermediate feature maps
output, feature_maps = unet_with_gradcam(input_image)

# Run GradCam to calculate gradients
explainability_results = GradCAM(output, feature_maps)
