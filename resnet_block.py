import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, downsample=False):
        super(BasicBlock, self).__init__()

        if downsample is True:
            out_channels = 2 * in_channels
        else:
            out_channels = in_channels

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=out_channels
        )
        if downsample is True:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(
                    num_features=out_channels
                )
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU()

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, width=1):
        super(Bottleneck, self).__init__()
        bottleneck_channels = width * out_channels // 4
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(
            num_features=bottleneck_channels
        )
        self.conv2 = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=bottleneck_channels
        )
        self.conv3 = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=out_channels
        )
        self.relu = nn.ReLU()
        if downsample is True:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(
                    num_features=out_channels
                )
            )
        else:
            self.downsample = None
        self.stride = stride

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetLayer(nn.Sequential):
    def __init__(self, in_channels, blocks, stride=None, out_channels=None, fisrt_layer_downsample=True, width=1):
        if out_channels is None:
            layers = [
                BasicBlock(
                    in_channels=in_channels,
                    downsample=fisrt_layer_downsample
                )
            ]
            for _ in range(1, blocks):
                layers.append(
                    BasicBlock(
                        in_channels=in_channels if not fisrt_layer_downsample else in_channels * 2
                    )
                )
        else:
            layers = [
                Bottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    downsample=True,
                    width=width
                )
            ]
            for _ in range(1, blocks):
                layers.append(
                    Bottleneck(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        width=width
                    )
                )
        super(ResNetLayer, self).__init__(*layers)
