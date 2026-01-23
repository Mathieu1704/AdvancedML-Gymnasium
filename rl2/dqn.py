from __future__ import annotations

import torch
import torch.nn as nn


class CNNQNetwork(nn.Module):
    """
    DQN CNN (Atari-like).
    Entrée : (B, C, H, W) float dans [0,1]
    Sortie : (B, n_actions) Q-values
    """

    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        q = self.fc(z)
        return q


@torch.no_grad()
def soft_update_(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Polyak averaging: target <- (1-tau)*target + tau*online."""
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)
