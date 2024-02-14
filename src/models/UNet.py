import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: str = "same",
    ):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depth: int = 4,
        conv_kernel_size: int = 1,
        channel_multiple: int = 64,
        pool_kernel_size: int = 2,
    ):
        super(UNet, self).__init__()

        assert (
            1 <= depth <= 4
        ), "UNet cannot must have depth of at least 1 and at most 4"

        self.kernel_size = pool_kernel_size
        self.down_samplers = []
        self.up_samplers = []

        self.down_samplers.append(ConvBlock(in_channels, channel_multiple))
        for _ in range(depth):
            self.down_samplers.append(ConvBlock(channel_multiple, channel_multiple * 2))
            channel_multiple *= 2

        self.center = ConvBlock(channel_multiple, channel_multiple * 2)
        channel_multiple *= 2

        for _ in range(depth):
            self.up_samplers.append(ConvBlock(channel_multiple, channel_multiple / 2))
            channel_multiple /= 2

        self.output = nn.Conv2d(
            channel_multiple, num_classes, kernel_size=conv_kernel_size
        )

    def forward(self, x):
        sample_maps = []

        for i, down_sampler in enumerate(self.down_samplers):
            if i > 0:
                x = nn.MaxPool2d(kernel_size=self.kernel_size)(x)
            x = down_sampler(x)
            sample_maps.append(x)

        x = self.center(nn.MaxPool2d(kernel_size=self.kernel_size)(x))
        sample_maps.append(x)

        for i, up_sampler in enumerate(self.up_samplers):
            up_sampled_img = nn.Upsample(
                scale_factor=self.kernel_size, mode="bilinear"
            )(x)
            down_sampled_img = sample_maps[i - 1]
            x = torch.cat(up_sampled_img, down_sampled_img, dim=1)
            x = up_sampler(x)

        output = self.output(x)
        return output


# # Load pre-trained ResNet model for gradient computation
# resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# resnet = nn.Sequential(
#     *list(resnet.children())[:-1]
# )  # Remove the last fully connected layer
# resnet.eval()  # Set the model to evaluation mode


# class UNetWithGradCAM(nn.Module):
#     def __init__(self, unet):
#         super(UNetWithGradCAM, self).__init__()
#         self.unet = unet

#     def forward(self, x):
#         features = self.unet.feature_maps  # Intermediate feature maps from the U-Net
#         output = self.unet(x)
#         return output, features
