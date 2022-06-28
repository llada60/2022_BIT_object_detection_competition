import torch.nn as nn
from resnet_block import ResNetLayer


class ResNet50(nn.Module):
    def __init__(self, config):
        super(ResNet50, self).__init__()
        width = config['width']
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=64
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.layer1 = ResNetLayer(
            in_channels=64,
            out_channels=256,
            blocks=3,
            stride=1,
            width=width
        )
        self.layer2 = ResNetLayer(
            in_channels=256,
            out_channels=512,
            blocks=4,
            stride=2,
            width=width
        )
        self.layer3 = ResNetLayer(
            in_channels=512,
            out_channels=1024,
            blocks=6,
            stride=2,
            width=width
        )
        self.layer4 = ResNetLayer(
            in_channels=1024,
            out_channels=2048,
            blocks=3,
            stride=2,
            width=width
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        return [layer1_out, layer2_out, layer3_out, layer4_out]
