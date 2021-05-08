#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : FG-NIC
# @Author       : Xiaoyu LIN
# @File         : restoration.py
# @Description  : This file contains funtions for restoration networks.

from torch.nn import Module
from torch import Tensor
import torch.nn as nn
import torch
import os
import math

from restorations.dncnn.dncnn import DnCNN
from restorations.memnet.memnet import MemNet
    
def initialize_restoration(name: str, 
                           dataset: str,
                           path: str,
                           use_pretrain: bool=True,
                          ) -> Module:
    """ Initialize or load restoration networks for RGB color images. The network will be initalzied to 
            kaiming method if the pretrained network is not loaded.
        Args:
            name (str): Restoration network name in ['DnCNN', 'MemNet', 'RIDNet'].
            path (int): The name of the folder keep the pretrained network. 
            use_pretrain (bool): If true, load pretrained network.
        Return:
            model (Module): Restoration network.
    """
    model = None
    
    # DnCNN
    if "dncnn" in name.lower():
        model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
    
    # MemNet
    elif "memnet" in name.lower():
        model = MemNet(in_channels=3, channels=20, num_memblock=6, num_resblock=4)
    
    else:
        raise ValueError("Invalid model name.")
    
    if use_pretrain:
        model.load_state_dict(torch.load(os.path.join(path, '-'.join([dataset, name.lower()]), 'model.pth')), strict=True)
    else:
        model.apply(weights_init_kaiming)

    return model


def weights_init_kaiming(m) -> None:
    """ Initalzied the restoration network with kaiming method.
    Notes:
        Copy from: https://github.com/IVRL/DEU.
    """
    classname = m.__class__.__name__
    if classname != 'BNReLUConv': # filtered for MemNet: BNReLUConv, ResidualBlock, MemoryBlock
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
            nn.init.constant_(m.bias.data, 0.0)

            
def getPSNR(inputs: Tensor, 
            origins: Tensor
           ) -> Tensor:
    """ Calculate average PSNR for a batch of data.
    Args:
        inputs (Tensor): Pytorh Tensor with shape (N, C, H, W).
        origins (Tensor): Pytorh Tensor with the shape as inputs. 
    Returns:
        (Tensor): average PSNR for a batch of images.
    """
    mse = torch.mean((inputs - origins + torch.finfo(torch.float32).eps) ** 2.0, (-2, -1)).mean(-1)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).mean()
