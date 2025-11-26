import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """3D residual block，用于时空特征下采样。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(1, stride, stride),
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, stride, stride)),
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


class Conv3DDQN(nn.Module):
    """
    单头 Conv3D + ResBlock DQN（RGB 堆叠）。
    """

    def __init__(self, stack: int, n_actions: int, features_dim: int = 256, channels=(16, 32, 64, 96, 128)):
        super().__init__()
        layers = [
            nn.Conv3d(3, channels[0], kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
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
        self.q_head = nn.Linear(features_dim, n_actions)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, stack, 1, H, W)
        if x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        x = x.permute(0, 2, 1, 3, 4)  # (B,3,stack,H,W)
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.proj(x)
        return self.q_head(x)
