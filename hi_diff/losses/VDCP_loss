import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from math import exp
_reduction_modes = ['none', 'mean', 'sum']


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_eachItem(img1, img2, window_size=11, window=None, size_average=True, useeachItem=True, val_range=None): #useeachitem is fenbie return lumin structure constrast
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    mu1_mu2 = mu1 * mu2

    
    # E(x^2)-E(x)^2  mu1_sq is E(x)
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd,
                         groups=channel) - mu1_sq
    sigma1_sq[sigma1_sq<0]=0
    sigma1 = torch.sqrt(sigma1_sq)
    
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd,
                         groups=channel) - mu2_sq
    
    sigma2_sq[sigma2_sq<0]=0
    sigma2 = torch.sqrt(sigma2_sq)
    
    sigma12 = F.conv2d(img1 * img2, window, padding=padd,
                       groups=channel) - mu1_mu2
    
    struclossCov = torch.exp((sigma12-sigma1*sigma2)/(sigma1_sq+sigma2_sq+0.0001))

    return 1 - struclossCov.nanmean()


@LOSS_REGISTRY.register()
class VDCP_loss(torch.nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(3, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range=None, full=False):
        assert isinstance(self.w, torch.Tensor)

        num_channels = 3
        if data_range is None:
            data_range = torch.ones_like(Y) #* Y.max()
            p = (self.win_size - 1)//2
            data_range = data_range[:, :, p:-p, p:-p]
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w, groups=num_channels)  # typing: ignore
        uy = F.conv2d(Y, self.w, groups=num_channels)  #
        uxx = F.conv2d(X * X, self.w, groups=num_channels)
        uyy = F.conv2d(Y * Y, self.w, groups=num_channels)
        uxy = F.conv2d(X * Y, self.w, groups=num_channels)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        
        vx_root = torch.nan_to_num(torch.sqrt(vx),nan=0.0)
        vy_root = torch.nan_to_num(torch.sqrt(vy),nan=0.0)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        ###########SSIM  v0
        '''
        D = B1 * B2
        S = (A1 * A2) / D
        S = S.mean()
        
        S = 1 - S
        '''
        ###########VDLP  v1
        
        S_ = (vxy - vx_root*vy_root) / (vx+vy+0.01)
        S_ = S_.mean()
        S_ = - S_
        
        prod = F.l1_loss(X, Y)

        return prod + 0.01 * S_
