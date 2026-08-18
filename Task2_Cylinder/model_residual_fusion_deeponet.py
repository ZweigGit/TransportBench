import torch
import torch.nn as nn

class sin_act(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Residual_Fusion_DeepONet(nn.Module):
    def __init__(self, branch_dim=2, trunk_dim=2, hidden_dim=278, num_outputs=4, depth=5, activation='GELU'):
        """
        Fusion DeepONet with residual cross-modulation for Cylinder Flow Task.

        Identical to Fusion_DeepONet except the branch-to-trunk fusion gate:
            Fusion_DeepONet:          x_trunk = x_trunk * skip[i]          (raw product)
            Residual_Fusion_DeepONet: x_trunk = x_trunk * (1.0 + skip[i])  (residual gate)

        The identity term keeps the trunk signal/gradient path alive when the
        branch signal is weak, preventing branch collapse with the raw
        product. Same 1,010,034 params as Fusion_DeepONet.

        Args:
            branch_dim: 2 (Kn, Ma)
            trunk_dim: 2 (x, y)
            hidden_dim: 278 (Target ~1.0M Params)
            num_outputs: 4 (Physics quantities)
        """
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.act2 = sin_act()

        #Fusion-deeponet additional parameters
        self.L = depth + 1

        self.ab  = nn.Parameter(torch.full((self.L,),0.1))
        self.cb  = nn.Parameter(torch.full((self.L,),0.1))
        self.a1b = nn.Parameter(torch.zeros(self.L,))
        self.F1b = nn.Parameter(torch.full((self.L,),0.1))
        self.c1b = nn.Parameter(torch.zeros(self.L,))

        self.at  = nn.Parameter(torch.full((self.L,),0.1))
        self.ct  = nn.Parameter(torch.full((self.L,),0.1))
        self.a1t = nn.Parameter(torch.zeros(self.L,))
        self.F1t = nn.Parameter(torch.full((self.L,),0.1))
        self.c1t = nn.Parameter(torch.zeros(self.L,))

        # Branch Net
        self.branch_net = nn.ModuleList()
        self.branch_net.append(nn.Linear(branch_dim, hidden_dim))
        for _ in range(depth - 1):
            self.branch_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))

        # Trunk Net: [Linear1, Linear2, ..., Last Linear]
        self.trunk_net = nn.ModuleList()
        self.trunk_net.append(nn.Linear(trunk_dim, hidden_dim))
        for _ in range(depth - 1):
            self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))

        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def forward(self, x_branch, x_trunk):

        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """

        if x_trunk.dim() == 2:
            # [N, trunk_dim] -> [1, N, trunk_dim] -> [B, N, trunk_dim]
            x_trunk = x_trunk.unsqueeze(0).repeat(x_branch.shape[0], 1, 1)

        skip = []

        for i in range(self.L-1):

            x_branch = self.act(10*self.ab[i]*self.branch_net[i](x_branch)+self.cb[i])+\
            10*self.a1b[i]*self.act2(10*self.F1b[i]*self.branch_net[i](x_branch)+self.c1b[i])
            skip.append(x_branch)

        for i in range(1,self.L-1):
            skip[i] = skip[i-1]+skip[i]

        for i in range(self.L-1):

            x_trunk = self.act(10*self.at[i]*self.trunk_net[i](x_trunk)+self.ct[i])+\
            10*self.a1t[i]*self.act2(10*self.F1t[i]*self.trunk_net[i](x_trunk)+self.c1t[i])

            # Residual gate: identity path preserves trunk magnitude/gradient
            x_trunk = torch.einsum('bk,bik->bik', 1.0 + skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        # [Batch, num_outputs, hidden_dim]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        return prediction
