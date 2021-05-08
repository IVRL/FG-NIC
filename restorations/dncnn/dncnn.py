from typing import Any, Callable, List, Optional, Union, Tuple, Iterable
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class DnCNN(nn.Module):
    """ DnCNN (20 conv layers)
    From:
        https://github.com/cszn/KAIR.
    Note:
        Batch normalization and residual learning are beneficial to Gaussian denoising (especially
            for a single noise level). The residual of a noisy image corrupted by additive white
            Gaussian noise (AWGN) follows a constant Gaussian distribution which stablizes batch
            normalization during training.
    Args:
        in_nc (int): channel number of input
        out_nc (int): channel number of output
        nc (int): channel number
        nb (int): total number of conv layers
        act_mode (str): batch norm + activation function; 'BR' means BN+ReLU.
    """
    def __init__(self, 
                 in_nc: int = 3, 
                 out_nc: int = 1, 
                 nc: int = 64, 
                 nb: int =17, 
                 act_mode: str ='BR'
                ) -> None:
        
        super(DnCNN, self).__init__()
        
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = conv(nc, out_nc, mode='C', bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x-n

def sequential(*args: Iterable[Union[nn.Sequential, nn.Module]]
              ) -> nn.Sequential:
    """ Advanced nn.Sequential.
    From:
        https://github.com/xinntao/BasicSR
        https://github.com/cszn/KAIR
    Args:
        *args (nn.Sequential or nn.Module): Modules or sequential containers.
    Returns:
        (nn.Sequential): Modules will be sequentially connected in the order they are passed in.
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels: int = 64, 
         out_channels: int = 64, 
         kernel_size: int = 3, 
         stride: int = 1, 
         padding: int = 1, 
         bias: bool = True, 
         mode: str = 'CBR', 
         negative_slope: float = 0.2
        ) -> nn.Sequential:
    """ Implement sequential CNN model from mode string.
    From:
        https://github.com/xinntao/BasicSR
        https://github.com/cszn/KAIR
    Returns:
        (nn.Sequential): Modules will be sequentially connected in the order of string in mode.
    """
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding, 
                               bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=kernel_size, 
                                        stride=stride, 
                                        padding=padding, 
                                        bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
            
    return sequential(*L)