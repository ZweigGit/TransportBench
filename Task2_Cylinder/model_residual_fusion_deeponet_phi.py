import torch
import torch.nn as nn

class sin_act(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class PhiActivation(nn.Module):
    """B-spline of order 3, compact support on [0, 3] (Liu et al. 2020)."""
    def forward(self, x):
        return (torch.relu(x) ** 2
                - 3 * torch.relu(x - 1) ** 2
                + 3 * torch.relu(x - 2) ** 2
                - torch.relu(x - 3) ** 2)

class Residual_Fusion_DeepONet_Phi(nn.Module):
    def __init__(self, branch_dim=2, trunk_dim=2, hidden_dim=278, num_outputs=4, depth=5, activation='GELU'):
        """
        Residual Fusion DeepONet with Phi (B-spline) trunk activation.

        Identical to Residual_Fusion_DeepONet except the trunk:
            - hidden activation is Phi (B-spline, support [0, 3]) instead of
              GELU/Tanh, same as MscaleDeepONet
            - the sin modulation term is removed (no a1t/F1t/c1t params)
            - xavier gain=6 init on trunk Linears (Phi + default init kills
              all trunk gradients, same bug as MscaleDeepONet)

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
        self.trunk_act = PhiActivation()

        #Fusion-deeponet additional parameters
        self.L = depth + 1

        self.ab  = nn.Parameter(torch.full((self.L,),0.1))
        self.cb  = nn.Parameter(torch.full((self.L,),0.1))
        self.a1b = nn.Parameter(torch.zeros(self.L,))
        self.F1b = nn.Parameter(torch.full((self.L,),0.1))
        self.c1b = nn.Parameter(torch.zeros(self.L,))

        self.at = nn.Parameter(torch.full((self.L,),0.1))
        self.ct = nn.Parameter(torch.full((self.L,),0.1))

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

        # Phi is exactly 0 for x<=0, so default Linear init kills all trunk
        # gradients (same bug as MscaleDeepONet). Larger init keeps pre-acts
        # inside the support [0, 3] where Phi has unit-scale outputs.
        for m in self.trunk_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=6.0)
                fan_in = m.weight.shape[1]
                nn.init.uniform_(m.bias, -6.0 / fan_in ** 0.5, 6.0 / fan_in ** 0.5)

    def forward(self, x_branch, x_trunk):

        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """

        if x_trunk.dim() == 2:
            # Shared 2D trunk: [N, trunk_dim] -> [B, N, trunk_dim] (view, no copy)
            B = x_branch.shape[0]
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)

        skip = []

        for i in range(self.L-1):

            z_b = self.branch_net[i](x_branch)
            x_branch = self.act(10*self.ab[i]*z_b+self.cb[i])+\
            10*self.a1b[i]*self.act2(10*self.F1b[i]*z_b+self.c1b[i])
            skip.append(x_branch)

        for i in range(1,self.L-1):
            skip[i] = skip[i-1]+skip[i]

        for i in range(self.L-1):

            z_t = self.trunk_net[i](x_trunk)
            x_trunk = self.trunk_act(10*self.at[i]*z_t+self.ct[i])

            # Residual gate: identity path preserves trunk magnitude/gradient
            x_trunk = torch.einsum('bk,bik->bik', 1.0 + skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        # [Batch, num_outputs, hidden_dim]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        return prediction
