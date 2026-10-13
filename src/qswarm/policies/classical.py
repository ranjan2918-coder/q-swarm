"""Classical neural network policies and parameter-matched baselines."""

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from qswarm.policies.features import compress_features_deterministic, extract_drone_features


class ClassicalMLPPolicy(nn.Module):
    """Standard compact MLP Deep Q-Network baseline."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.hidden_dims = hidden_dims
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for action space."""
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def select_action(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Select epsilon-greedy discrete action for a single drone observation."""
        if rng is not None and rng.random() < epsilon:
            return int(rng.integers(0, self.output_dim))

        self.eval()
        with torch.no_grad():
            feat = extract_drone_features(obs)
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            q_vals = self.forward(x)
            return int(torch.argmax(q_vals, dim=-1).item())

    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "type": "classical_mlp",
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ClassicalMLPPolicy":
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint.get("hidden_dims", [64, 32]),
            output_dim=checkpoint["output_dim"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model


class ClassicalMatchedPolicy(nn.Module):
    """Parameter-matched classical baseline intentionally constrained to ~25-45 parameters.

    Operates on the 4 compressed features:
    Linear(4, 5) -> 4 * 5 + 5 = 25 parameters.
    This provides a direct, honest parameter-for-parameter control for the 4-qubit Hybrid VQC.
    """

    def __init__(self, input_dim: int = 4, output_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def select_action(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        if rng is not None and rng.random() < epsilon:
            return int(rng.integers(0, self.output_dim))

        self.eval()
        with torch.no_grad():
            comp_feat = compress_features_deterministic(obs, target_dim=self.input_dim)
            x = torch.tensor(comp_feat, dtype=torch.float32).unsqueeze(0)
            q_vals = self.forward(x)
            return int(torch.argmax(q_vals, dim=-1).item())

    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "type": "classical_matched",
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ClassicalMatchedPolicy":
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        model = cls(input_dim=checkpoint["input_dim"], output_dim=checkpoint["output_dim"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model
