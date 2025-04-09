import torch
import torch.nn.functional as F
from utils import Args

def soft_ce(pred, target, args: Args):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, args)
    return -(target * pred).sum(-1, keepdim=True)


@torch.jit.script
def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, args: Args):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if args.num_bins == 0:
        return x
    elif args.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), args.vmin, args.vmax).squeeze(1)
    bin_idx = torch.floor((x - args.vmin) / args.bin_size).long()
    bin_offset = ((x - args.vmin) / args.bin_size - bin_idx.float()).unsqueeze(-1).to(torch.float32)
    soft_two_hot = torch.zeros(x.size(0), args.num_bins, device=x.device)
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % args.num_bins, bin_offset)
    return soft_two_hot


DREG_BINS = None

def two_hot_inv(x, args: Args):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    global DREG_BINS
    if args.num_bins == 0:
        return x
    elif args.num_bins == 1:
        return symexp(x)
    if DREG_BINS is None:
        DREG_BINS = torch.linspace(args.vmin, args.vmax, args.num_bins, device=x.device)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
    return symexp(x)
