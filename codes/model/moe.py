from torch import nn
import torch
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Expert(nn.Module):
    """
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(self.net(x))


# -------------------------------------------------------
# MoE(text_cls, image_cls)
# -------------------------------------------------------
class MoE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.in_proj = nn.ModuleDict({
            't': nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(),
                nn.Linear(in_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),      # <--- LN1
            ),
            'i': nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(),
                nn.Linear(in_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),      # <--- LN2
            )
        })

        self.experts = nn.ModuleDict({
            't': Expert(hidden_dim),
            'i': Expert(hidden_dim)
        })

        self.router_mix = Mlp(hidden_dim, hidden_dim // 2, 2)

        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)          # <--- LN3
        )

    def kl_align(self, fusion, expert):
        return F.kl_div(
            F.log_softmax(fusion, dim=-1),
            F.softmax(expert, dim=-1),
            reduction='batchmean'
        )

    def forward(self, text_cls, image_cls):

        proj_t = self.in_proj['t'](text_cls)
        proj_i = self.in_proj['i'](image_cls)

        expert_t_on_t = self.experts['t'](proj_t)
        expert_i_on_t = self.experts['i'](proj_t)

        expert_t_on_i = self.experts['t'](proj_i)
        expert_i_on_i = self.experts['i'](proj_i)

        wt = F.softmax(self.router_mix(proj_t), dim=-1)
        wi = F.softmax(self.router_mix(proj_i), dim=-1)
        # wt = wt.view(-1, 2)
        # wi = wi.view(-1, 2)
        fused_t = wt[:, 0:1] * expert_t_on_t + wt[:, 1:2] * expert_i_on_t
        fused_i = wi[:, 0:1] * expert_t_on_i + wi[:, 1:2] * expert_i_on_i

        fusion_h = self.fusion(torch.cat([fused_t, fused_i], dim=-1))

        return fusion_h

