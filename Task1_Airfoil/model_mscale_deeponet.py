import torch
import torch.nn as nn


class PhiActivation(nn.Module):
    """B-spline of order 3, compact support on [0, 3] (Liu et al. 2020)."""
    def forward(self, x):
        return (torch.relu(x) ** 2
                - 3 * torch.relu(x - 1) ** 2
                + 3 * torch.relu(x - 2) ** 2
                - torch.relu(x - 3) ** 2)


class MscaleDeepONet(nn.Module):
    """DeepONet architecture aligned with BoltzmannDeepONet; trunk uses Phi activation.

    Args:
        branch_dim: Input dimension of the branch net (sensor values).
        trunk_dim:  Input dimension of the trunk net (coordinates).
        hidden_dim: Width of hidden layers.
        num_outputs: Output channels.
        depth: Number of hidden [Linear -> Act] blocks.
        activation: Branch activation ('GELU' or 'Tanh'); trunk always uses Phi.
    """
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, num_outputs=4,
                 depth=5, activation='GELU'):
        super().__init__()

        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        trunk_act = PhiActivation()

        # Branch Net: Input -> [Hidden -> Act] x depth -> Linear to (hidden_dim * num_outputs)
        branch_layers = [nn.Linear(branch_dim, hidden_dim), act]
        for _ in range(depth - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), act])
        branch_layers.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk Net: Input -> [Hidden -> Act] x depth -> Linear -> Act, same shape
        # as BoltzmannDeepONet trunk but with Phi (B-spline) activation throughout
        trunk_layers = [nn.Linear(trunk_dim, hidden_dim), trunk_act]
        for _ in range(depth - 1):
            trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), trunk_act])
        trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), trunk_act])
        self.trunk_net = nn.Sequential(*trunk_layers)

        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def forward(self, x_branch, x_trunk):
        """
        x_branch: [Batch, branch_dim]
        x_trunk:  [N_points, trunk_dim]
        """
        B_out = self.branch_net(x_branch)           # [B, hidden * num_outputs]
        T_out = self.trunk_net(x_trunk)             # [N_points, hidden]

        B_out = B_out.view(-1, self.num_outputs, self.hidden_dim)
        pred = torch.einsum("bkh, nh -> bkn", B_out, T_out)
        return pred.permute(0, 2, 1)                # [B, N, num_outputs]
