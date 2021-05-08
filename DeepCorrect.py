#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : FG-NIC
# @Author       : Xiaoyu LIN
# @File         : DeepCorrect.py
# @Description  : This file contains full model architecture for the DeepCorrect baseline method 
#                 base on the paper: https://ieeexplore.ieee.org/document/8746775.
#                 Note: to simplify implementation, we apply correct units to every filter in all convolutional layers.

from typing import Any, Callable, List, Optional, Union, Tuple, Iterable
import copy
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math

from util import set_parameter_requires_grad
from degradation import Normalize

class DeepCorrect(nn.Module):
    
    def __init__(self,
                 feature_extractor: Module,
                 classifier: Module,
                 MEAN: list=[0.485, 0.456, 0.406],
                 STD: list=[0.229, 0.224, 0.225]
                ) -> None:
        
        super(DeepCorrect, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.img_normal = Normalize(MEAN, STD)
    
        # spatial multiplication and spatial addition
        self.correction_units = nn.ModuleList([])
        for child in self.feature_extractor.children():
            if isinstance(child, nn.Sequential):
                for subchild in child.children():
                    if isinstance(subchild, nn.Conv2d):
                        self.correction_units.append(CorrectionUnit(in_channels=subchild.out_channels))
            else:
                if isinstance(child, nn.Conv2d):
                    self.correction_units.append(CorrectionUnit(in_channels=child.out_channels))
                                                      
    def forward(self, degraded):
        
        # normalize
        activations = self.img_normal(degraded)
        
        i = 0
        for child in self.feature_extractor.children():
            if isinstance(child, nn.Sequential):
                for subchild in child.children():
                    activations = subchild(activations)
                    if isinstance(subchild, nn.Conv2d):
                        activations = self.correction_units[i](activations) + activations
                        i += 1
            else:
                activations = child(activations)
                if isinstance(child, nn.Conv2d):
                    activations = self.correction_units[i](activations) + activations
                    i += 1
            
        feature = torch.flatten(activations, 1)

        return self.classifier(feature)
    
    
def CorrectionUnit(in_channels: int,
                   kernel_size: int=3, 
                   depth: int=4, 
                   stride: int=1) -> nn.Sequential:
    
    modules = []
    modules.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=stride))
    modules.append(nn.BatchNorm2d(num_features=in_channels))    
    modules.append(nn.ReLU())    
    
    for _ in range(depth-2):
        modules.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,  
                                 padding=kernel_size%2, padding_mode='reflect'))
        modules.append(nn.BatchNorm2d(num_features=in_channels))    
        modules.append(nn.ReLU())    
    
    modules.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=stride))
    
    return nn.Sequential(*modules)
        