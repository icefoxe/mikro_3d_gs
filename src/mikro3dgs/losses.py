import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def l1_loss(pred, target, mask=None):
    if mask is None:
        return F.l1_loss(pred, target)
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() * 3 + 1e-8)


def mse_loss(pred, target, mask=None):
    if mask is None:
        return F.mse_loss(pred, target)
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() * 3 + 1e-8)


def ssim_loss(pred, target):
    pred = pred.permute(2, 0, 1).unsqueeze(0)
    target = target.permute(2, 0, 1).unsqueeze(0)
    return 1.0 - ssim(pred, target, data_range=1.0, size_average=True)


def opacity_regularization(opacities):
    return torch.mean(opacities)


def scale_regularization(scales):
    return torch.mean(scales ** 2)


def anisotropy_regularization(scales):
    s_min = scales.min(dim=1).values.clamp_min(1e-6)
    s_max = scales.max(dim=1).values
    return torch.mean(torch.relu((s_max / s_min) - 25.0))


def combined_loss(pred, target, mask=None, opacities=None, scales=None):
    l1 = l1_loss(pred, target, mask)
    ssim_val = ssim_loss(pred, target)
    loss = 0.8 * l1 + 0.2 * ssim_val

    if opacities is not None:
        loss = loss + 2e-4 * opacity_regularization(opacities)

    if scales is not None:
        loss = loss + 1e-5 * scale_regularization(scales)
        loss = loss + 1e-5 * anisotropy_regularization(scales)

    return loss
