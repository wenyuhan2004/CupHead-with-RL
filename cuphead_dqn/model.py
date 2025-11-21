import torch
import torch.nn as nn


class ConvDQN(nn.Module):
    """简化版 DQN CNN 头，输入 shape: (B, C, H, W)."""

    def __init__(self, in_channels: int, n_actions: int, conv_input_shape=(4, 128, 72)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *conv_input_shape)
            feat_dim = self.features(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.head(feats)
