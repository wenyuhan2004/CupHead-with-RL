"""Custom CNN feature extractors for Cuphead PPO."""

from typing import Sequence

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    """A lightweight residual block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.act(out)


class CupheadCNNExtractor(BaseFeaturesExtractor):
    """ResNet-style extractor for 16:9 Cuphead observations."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        channels: Sequence[int] = (32, 64, 96, 128, 192, 256),
    ):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]

        layers = [
            nn.Conv2d(in_channels, channels[0], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        ]
        curr = channels[0]
        for ch in channels[1:]:
            layers.append(ResidualBlock(curr, ch, stride=2))
            layers.append(ResidualBlock(ch, ch))
            curr = ch
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.backbone = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.proj = nn.Sequential(
            nn.Linear(curr, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.backbone(observations)
        x = self.flatten(x)
        return self.proj(x)


class ResidualBlock3D(nn.Module):
    """3D residual block for temporal-spatial modeling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, stride, stride),
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, stride, stride),
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.act(out)


class CupheadConv3DExtractor(BaseFeaturesExtractor):
    """Conv3D extractor treating stacked frames as temporal depth."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        channels: Sequence[int] = (16, 32, 64, 96, 128),
    ):
        super().__init__(observation_space, features_dim)
        stack, height, width = observation_space.shape

        layers = [
            nn.Conv3d(
                1,
                channels[0],
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
            ),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
        ]
        curr = channels[0]
        for ch in channels[1:]:
            layers.append(ResidualBlock3D(curr, ch, stride=2))
            layers.append(ResidualBlock3D(ch, ch))
            curr = ch
        layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.backbone = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.proj = nn.Sequential(
            nn.Linear(curr, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, stack, H, W)
        x = observations.unsqueeze(1)
        x = self.backbone(x)
        x = self.flatten(x)
        return self.proj(x)


class CupheadHKStyleExtractor(BaseFeaturesExtractor):
    """
    HK 风格：前端少量 3D 卷积提取时间特征，随后按帧求和再走 2D ResNet。
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        conv3d_channels: Sequence[int] = (32, 48, 64),
        res_channels: Sequence[int] = (64, 96, 128, 256),
    ):
        super().__init__(observation_space, features_dim)
        stack, height, width = observation_space.shape

        c3d_layers = []
        in_ch = 1
        for i, ch in enumerate(conv3d_channels):
            stride = (1, 2, 2) if i == 0 else (1, 1, 1)
            c3d_layers.extend(
                [
                    nn.Conv3d(
                        in_ch,
                        ch,
                        kernel_size=(2, 3, 3),
                        stride=stride,
                        padding=(0, 1, 1),
                    ),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = ch
        self.conv3d = nn.Sequential(*c3d_layers)

        res_layers = []
        cur = conv3d_channels[-1]
        for idx, ch in enumerate(res_channels):
            stride = 2 if idx > 0 else 1
            res_layers.append(ResidualBlock(cur, ch, stride=stride))
            res_layers.append(ResidualBlock(ch, ch))
            cur = ch
        res_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.resnet2d = nn.Sequential(*res_layers)
        self.flatten = nn.Flatten()
        self.proj = nn.Sequential(
            nn.Linear(cur, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, stack, H, W) 灰度
        x = observations.unsqueeze(1)  # -> (B,1,stack,H,W)
        x = self.conv3d(x)  # (B,C3D,T,H,W)
        x = torch.sum(x, dim=2)  # 简单时间汇聚 -> (B,C3D,H,W)
        x = self.resnet2d(x)
        x = self.flatten(x)
        return self.proj(x)
