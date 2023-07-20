import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same"
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        self.down1 = ConvBlock(in_channels, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        # self.down4 = ConvBlock(256, 512)
        self.center = ConvBlock(256, 512)
        # self.up4 = ConvBlock(1536, 512)
        self.up3 = ConvBlock(768, 256)
        self.up2 = ConvBlock(384, 128)
        self.up1 = ConvBlock(192, 64)
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)
        self.feature_maps = []

    def forward(self, x):
        self.feature_maps = []  # Clear the list before each forward pass
        down1 = self.down1(x)
        self.feature_maps.append(down1)
        down2 = self.down2(nn.MaxPool2d(2)(down1))
        self.feature_maps.append(down2)
        down3 = self.down3(nn.MaxPool2d(2)(down2))
        self.feature_maps.append(down3)
        # down4 = self.down4(nn.MaxPool2d(2)(down3))
        center = self.center(nn.MaxPool2d(2)(down3))
        self.feature_maps.append(center)
        # up4 = self.up4(torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(center), down4), dim=1))
        up3 = self.up3(
            torch.cat(
                (nn.Upsample(scale_factor=2, mode="bilinear")(center), down3), dim=1
            )
        )
        self.feature_maps.append(up3)
        up2 = self.up2(
            torch.cat((nn.Upsample(scale_factor=2, mode="bilinear")(up3), down2), dim=1)
        )
        self.feature_maps.append(up2)
        up1 = self.up1(
            torch.cat((nn.Upsample(scale_factor=2, mode="bilinear")(up2), down1), dim=1)
        )
        self.feature_maps.append(up1)
        output = self.output(up1)
        self.feature_maps.append(output)
        return output


# Load pre-trained ResNet model for gradient computation
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet = nn.Sequential(
    *list(resnet.children())[:-1]
)  # Remove the last fully connected layer
resnet.eval()  # Set the model to evaluation mode


class UNetWithGradCAM(nn.Module):
    def __init__(self, unet):
        super(UNetWithGradCAM, self).__init__()
        self.unet = unet

    def forward(self, x):
        features = self.unet.feature_maps  # Intermediate feature maps from the U-Net
        output = self.unet(x)
        return output, features
