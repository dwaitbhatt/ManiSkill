import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules. Adds 1 extra dimension.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness='different', **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs)

    def __repr__(self):
        return 'Vectorized ' + self._repr


class SimNorm(nn.Module):
    """
    Simplicial normalization. Same shape, don't care batch.
    Adapted from https://arxiv.org/abs/2204.00616.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim
    
    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)
        
    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout. Don't care batch
    """

    def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))
    
    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias is not None}{repr_dropout}, "\
            f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, final_act=None, dropout=0.):
    """
    Basic building block borrowed from TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=final_act) if final_act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)

def disc_mlp(in_dim, mlp_dims, out_dim, final_act="identity", dropout=0.3):
    """
    MLP with LayerNorm, LeakyReLU, and dropout in hidden layers output.
    Use final activation "identity" for WGAN, and "sigmoid" for GAN.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]

    hidden_act = nn.LeakyReLU(0.2, inplace=True)
    mlp = nn.ModuleList()
    # Linear first layer
    mlp.append(nn.Linear(dims[0], dims[1]))
    mlp.append(hidden_act)
    # LayerNorm, LeakyReLU, dropout in hidden layers
    for i in range(1, len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout, act=hidden_act))
    # Sigmoid output
    mlp.append(nn.Linear(dims[-2], dims[-1]))
    if final_act == "sigmoid":
        mlp.append(nn.Sigmoid())
    elif final_act == "identity":
        pass
    else:
        raise ValueError(f"Invalid final_act: {final_act}")
    return nn.Sequential(*mlp)

def copy_partial_mlp_weights(source_mlp: nn.Sequential, target_mlp: nn.Sequential, start_idx: int, end_idx: int):
    """Copy the weights of a source mlp model to a target mlp model. Keep the same architecture, but copy weights 
    only for the layers between start_idx and end_idx (both inclusive)."""
    if end_idx < 0:
        end_idx += len(source_mlp)
    # Copy state dict for all layers between start_idx and end_idx (both inclusive)
    for i in range(start_idx, end_idx+1):
        target_mlp[i].load_state_dict(source_mlp[i].state_dict())

def freeze_mlp_layers(mlp: nn.Sequential, start_idx: int, end_idx: int):
    """Freeze the weights of mlp layers between start_idx and end_idx (both inclusive)."""
    if end_idx < 0:
        end_idx += len(mlp)
    for i in range(start_idx, end_idx+1):
        mlp[i].requires_grad_(False)

# Initialization
def weight_init(m):
    """Custom weight initialization"""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ParameterList):
        for i,p in enumerate(m):
            if p.dim() == 3: # Linear
                nn.init.trunc_normal_(p, std=0.02) # Weight
                nn.init.constant_(m[i+1], 0) # Bias


def zero_(params):
    """Initialize parameters to zero."""
    for p in params:
        p.data.fill_(0)

