#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : FG-NIC
# @Author       : Xiaoyu LIN
# @File         : ablation.py
# @Description  : This file contains simplied model architecture based on model.py for ablation studies.

import copy
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

from degradation import Normalize

class SpatialMultiplication(nn.Module):
    """ Original full model without spatial multiplication.
    """
    def __init__(self,
                 mode: str,
                 restoration: Module,
                 fidelity_input: str,
                 fidelity_output: str,
                 feature_extractor: Module,
                 feature_size: int,
                 classifier: Module,
                 downsample: str='bilinear',
                 fidelity=None,
                 num_channel=4,
                 increase=0.25,
                 MEAN: list=[0.485, 0.456, 0.406],
                 STD: list=[0.229, 0.224, 0.225]
                ) -> None:
        
        super(SpatialMultiplication, self).__init__()
        
        self.mode = mode.lower()
        self.downsample_mode = downsample
        
        # prepare restoration network, feature extractor, classifier
        self.restoration = restoration
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.img_normal = Normalize(MEAN, STD)
        self.feature_size = feature_size
        
        # prepare fidelity map estimator
        self.fidelity_output = fidelity_output
        if fidelity is not None:
            self.fidelity = fidelity
            self.fidelity_input = fidelity_input
        if 'cos' in fidelity_output.lower():
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            in_channels = 1
        else:
            in_channels = 3
            
        # basic trainable module
        basic_cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=num_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=num_channel, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect')
                                 )
        # spatial multiplication and spatial addition
        cnn_layers = nn.ModuleList([copy.deepcopy(basic_cnn)])
        for child in feature_extractor.children():
            if isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                cnn_layers.append(copy.deepcopy(basic_cnn))
        self.cnn_layers_bias = copy.deepcopy(cnn_layers)
        
        if 'cos' not in self.fidelity_output.lower():
            sigma = math.sqrt((0.1**2+0.2**2+0.3**2+0.4**2+0.5**2) / (6.0**2 * 2.0))
            if 'l1' in self.fidelity_output.lower():
                mean = sigma * math.sqrt(2.0 / math.pi)
                std = sigma * math.sqrt(1.0- 2.0 / math.pi)
            elif 'l2' in self.fidelity_output.lower():
                mean = sigma**2.0
                std = math.sqrt(2.0) * sigma**2.0
            self.fidelity_normal = Normalize([mean]*in_channels, [std]*in_channels)
        
        # channel multiplication
        self.mul_fc = nn.Sequential(nn.Linear(feature_size, int(feature_size * increase)),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(int(feature_size * increase), feature_size),
                                   )
        
        # channel concatenation
        self.cat_fc = nn.Sequential(nn.Linear(feature_size * 2, int(feature_size * increase)),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(int(feature_size * increase), feature_size),
                                   )
        self.is_ensemble = False
    
    def downsample(self, x):
        
        x = x.mean(1, keepdim=True)
        y = x.permute(0, 1, 3, 2)
        
        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, 1, -1)
        x = F.interpolate(x, size=(1, self.feature_size//2), mode=self.downsample_mode, align_corners=False)
        
        batch_size, num_channels, height, width = y.shape
        y = y.reshape(batch_size, num_channels, 1, -1)
        y = F.interpolate(y, size=(1, self.feature_size//2), mode=self.downsample_mode, align_corners=False)
        
        return torch.cat((x, y), -1)
                                                      
    def forward(self, degraded, origins=None):
        
        # get restored image
        restored = self.restoration(degraded).clamp_(0, 1)
        
        # get fidelity map
        if 'oracle' in self.mode.lower():
            if 'l1' in self.fidelity_output.lower():
                fidelity_map = (restored-origins).abs()
            elif 'l2' in self.fidelity_output.lower():
                fidelity_map = (restored-origins).square()
            elif 'cos' in self.fidelity_output.lower():
                fidelity_map = self.cos(restored, origins)
                fidelity_map = fidelity_map.unsqueeze(1)
        else:                                            
            if 'degraded' in self.fidelity_input.lower():
                fidelity_map = degraded - self.fidelity(degraded)
            elif 'restored' in self.fidelity_input.lower():
                fidelity_map = restored - self.fidelity(restored)
        
        # normalize
        restored = self.img_normal(restored)
        if 'cos' not in self.fidelity_output.lower():
            fidelity_map = self.fidelity_normal(fidelity_map)
            fidelity_map = 1.0 - fidelity_map
        fidelity_map = fidelity_map.mean(dim=1, keepdim=True)
        
        # spatial multiplication and spatial addition
        restored_feature = restored
        fidelity_feature_bias = self.cnn_layers_bias[0](fidelity_map)
        restored_feature = restored_feature + fidelity_feature_bias
        i = 0
        for child in self.feature_extractor.children():
            if isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                i += 1 
                fidelity_feature_bias = self.cnn_layers_bias[i](fidelity_feature_bias) + fidelity_feature_bias
                fidelity_feature_bias = F.interpolate(fidelity_feature_bias, size=restored_feature.shape[-1], mode=self.downsample_mode, align_corners=False)
                restored_feature = restored_feature + fidelity_feature_bias
            restored_feature = child(restored_feature)
        restored_feature = torch.flatten(restored_feature, 1)
        
        # channel multiplication and channel concatenation:
        fidelity_feature = self.downsample(fidelity_map)
        fidelity_feature = torch.flatten(fidelity_feature, 1)
        restored_feature = restored_feature * 2.0 * torch.sigmoid(self.mul_fc(fidelity_feature))
        feature = self.cat_fc(torch.cat((restored_feature, fidelity_feature), -1))

        return self.classifier(feature)
    
    

class ResidualMechanism(nn.Module):
    """ Original full model without residual mechanism.
    """
    def __init__(self,
                 mode: str,
                 restoration: Module,
                 fidelity_input: str,
                 fidelity_output: str,
                 feature_extractor: Module,
                 feature_size: int,
                 classifier: Module,
                 downsample: str='bilinear',
                 fidelity=None,
                 num_channel=4,
                 increase=0.25,
                 MEAN: list=[0.485, 0.456, 0.406],
                 STD: list=[0.229, 0.224, 0.225]
                ) -> None:
        
        super(ResidualMechanism, self).__init__()
        
        self.mode = mode.lower()
        self.downsample_mode = downsample
        
        # prepare restoration network, feature extractor, classifier
        self.restoration = restoration
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.img_normal = Normalize(MEAN, STD)
        self.feature_size = feature_size
        
        # prepare fidelity map estimator
        self.fidelity_output = fidelity_output
        if fidelity is not None:
            self.fidelity = fidelity
            self.fidelity_input = fidelity_input
        if 'cos' in fidelity_output.lower():
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            in_channels = 1
        else:
            in_channels = 3
            
        # basic trainable module
        basic_cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=num_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=num_channel, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect')
                                 )
        # spatial multiplication and spatial addition
        cnn_layers = nn.ModuleList([copy.deepcopy(basic_cnn)])
        for child in feature_extractor.children():
            if isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                cnn_layers.append(copy.deepcopy(basic_cnn))
        self.cnn_layers_weight = cnn_layers
        self.cnn_layers_bias = copy.deepcopy(cnn_layers)
        
        if 'cos' not in self.fidelity_output.lower():
            sigma = math.sqrt((0.1**2+0.2**2+0.3**2+0.4**2+0.5**2) / (6.0**2 * 2.0))
            if 'l1' in self.fidelity_output.lower():
                mean = sigma * math.sqrt(2.0 / math.pi)
                std = sigma * math.sqrt(1.0- 2.0 / math.pi)
            elif 'l2' in self.fidelity_output.lower():
                mean = sigma**2.0
                std = math.sqrt(2.0) * sigma**2.0
            self.fidelity_normal = Normalize([mean]*in_channels, [std]*in_channels)
        
        # channel multiplication
        self.mul_fc = nn.Sequential(nn.Linear(feature_size, int(feature_size * increase)),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(int(feature_size * increase), feature_size),
                                   )
        
        # channel concatenation
        self.cat_fc = nn.Sequential(nn.Linear(feature_size * 2, int(feature_size * increase)),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(int(feature_size * increase), feature_size),
                                   )
        self.is_ensemble = False
        
    def downsample(self, x):
        
        x = x.mean(1, keepdim=True)
        y = x.permute(0, 1, 3, 2)
        
        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, 1, -1)
        x = F.interpolate(x, size=(1, self.feature_size//2), mode=self.downsample_mode, align_corners=False)
        
        batch_size, num_channels, height, width = y.shape
        y = y.reshape(batch_size, num_channels, 1, -1)
        y = F.interpolate(y, size=(1, self.feature_size//2), mode=self.downsample_mode, align_corners=False)
        
        return torch.cat((x, y), -1)
                                                      
    def forward(self, degraded, origins=None):
        
        # get restored image
        restored = self.restoration(degraded).clamp_(0, 1)
        
        # get fidelity map
        if 'oracle' in self.mode.lower():
            if 'l1' in self.fidelity_output.lower():
                fidelity_map = (restored-origins).abs()
            elif 'l2' in self.fidelity_output.lower():
                fidelity_map = (restored-origins).square()
            elif 'cos' in self.fidelity_output.lower():
                fidelity_map = self.cos(restored, origins)
                fidelity_map = fidelity_map.unsqueeze(1)
        else:                                            
            if 'degraded' in self.fidelity_input.lower():
                fidelity_map = degraded - self.fidelity(degraded)
            elif 'restored' in self.fidelity_input.lower():
                fidelity_map = restored - self.fidelity(restored)
        
        # normalize
        restored = self.img_normal(restored)
        if 'cos' not in self.fidelity_output.lower():
            fidelity_map = self.fidelity_normal(fidelity_map)
            fidelity_map = 1.0 - fidelity_map
        fidelity_map = fidelity_map.mean(dim=1, keepdim=True)
        
        # spatial multiplication and spatial addition
        restored_feature = restored
        fidelity_feature_weight = self.cnn_layers_weight[0](fidelity_map)
        fidelity_feature_bias = self.cnn_layers_bias[0](fidelity_map)
        restored_feature = 2.0 * torch.sigmoid(fidelity_feature_weight) * restored_feature + fidelity_feature_bias
        i = 0
        for child in self.feature_extractor.children():
            if isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                i += 1 
                fidelity_feature_weight = self.cnn_layers_weight[i](fidelity_feature_weight)
                fidelity_feature_bias = self.cnn_layers_bias[i](fidelity_feature_bias)
                fidelity_feature_weight = F.interpolate(fidelity_feature_weight, size=restored_feature.shape[-1], mode=self.downsample_mode, align_corners=False)
                fidelity_feature_bias = F.interpolate(fidelity_feature_bias, size=restored_feature.shape[-1], mode=self.downsample_mode, align_corners=False)
                restored_feature = 2.0 * torch.sigmoid(fidelity_feature_weight) * restored_feature + fidelity_feature_bias
            restored_feature = child(restored_feature)
        restored_feature = torch.flatten(restored_feature, 1)
        
        # channel multiplication and channel concatenation:
        fidelity_feature = self.downsample(fidelity_map)
        fidelity_feature = torch.flatten(fidelity_feature, 1)
        restored_feature = restored_feature * 2.0 * torch.sigmoid(self.mul_fc(fidelity_feature))
        feature = self.cat_fc(torch.cat((restored_feature, fidelity_feature), -1))

        return self.classifier(feature)
    
class SpatialAddition(nn.Module):
    """ Original full model without apatial addition.
    """
    def __init__(self,
                 mode: str,
                 restoration: Module,
                 fidelity_input: str,
                 fidelity_output: str,
                 feature_extractor: Module,
                 feature_size: int,
                 classifier: Module,
                 downsample: str='bilinear',
                 fidelity=None,
                 num_channel=4,
                 increase=0.25,
                 MEAN: list=[0.485, 0.456, 0.406],
                 STD: list=[0.229, 0.224, 0.225]
                ) -> None:
        
        super(SpatialAddition, self).__init__()
        
        self.mode = mode.lower()
        self.downsample_mode = downsample
        
        # prepare restoration network, feature extractor, classifier
        self.restoration = restoration
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.img_normal = Normalize(MEAN, STD)
        self.feature_size = feature_size
        
        # prepare fidelity map estimator
        self.fidelity_output = fidelity_output
        if fidelity is not None:
            self.fidelity = fidelity
            self.fidelity_input = fidelity_input
        if 'cos' in fidelity_output.lower():
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            in_channels = 1
        else:
            in_channels = 3
            
        # basic trainable module
        basic_cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=num_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=num_channel, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect')
                                 )
        # spatial multiplication and spatial addition
        cnn_layers = nn.ModuleList([copy.deepcopy(basic_cnn)])
        for child in feature_extractor.children():
            if isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                cnn_layers.append(copy.deepcopy(basic_cnn))
        self.cnn_layers_weight = cnn_layers
        
        if 'cos' not in self.fidelity_output.lower():
            sigma = math.sqrt((0.1**2+0.2**2+0.3**2+0.4**2+0.5**2) / (6.0**2 * 2.0))
            if 'l1' in self.fidelity_output.lower():
                mean = sigma * math.sqrt(2.0 / math.pi)
                std = sigma * math.sqrt(1.0- 2.0 / math.pi)
            elif 'l2' in self.fidelity_output.lower():
                mean = sigma**2.0
                std = math.sqrt(2.0) * sigma**2.0
            self.fidelity_normal = Normalize([mean]*in_channels, [std]*in_channels)
        
        # channel multiplication
        self.mul_fc = nn.Sequential(nn.Linear(feature_size, int(feature_size * increase)),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(int(feature_size * increase), feature_size),
                                   )
        
        # channel concatenation
        self.cat_fc = nn.Sequential(nn.Linear(feature_size * 2, int(feature_size * increase)),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(int(feature_size * increase), feature_size),
                                   )
        self.is_ensemble = False
    
    def downsample(self, x):
        
        x = x.mean(1, keepdim=True)
        y = x.permute(0, 1, 3, 2)
        
        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, 1, -1)
        x = F.interpolate(x, size=(1, self.feature_size//2), mode=self.downsample_mode, align_corners=False)
        
        batch_size, num_channels, height, width = y.shape
        y = y.reshape(batch_size, num_channels, 1, -1)
        y = F.interpolate(y, size=(1, self.feature_size//2), mode=self.downsample_mode, align_corners=False)
        
        return torch.cat((x, y), -1)
                                                      
    def forward(self, degraded, origins=None):
        
        # get restored image
        restored = self.restoration(degraded).clamp_(0, 1)
        
        # get fidelity map
        if 'oracle' in self.mode.lower():
            if 'l1' in self.fidelity_output.lower():
                fidelity_map = (restored-origins).abs()
            elif 'l2' in self.fidelity_output.lower():
                fidelity_map = (restored-origins).square()
            elif 'cos' in self.fidelity_output.lower():
                fidelity_map = self.cos(restored, origins)
                fidelity_map = fidelity_map.unsqueeze(1)
        else:                                            
            if 'degraded' in self.fidelity_input.lower():
                fidelity_map = degraded - self.fidelity(degraded)
            elif 'restored' in self.fidelity_input.lower():
                fidelity_map = restored - self.fidelity(restored)
        
        # normalize
        restored = self.img_normal(restored)
        if 'cos' not in self.fidelity_output.lower():
            fidelity_map = self.fidelity_normal(fidelity_map)
            fidelity_map = 1.0 - fidelity_map
        fidelity_map = fidelity_map.mean(dim=1, keepdim=True)
        
        # spatial multiplication and spatial addition
        restored_feature = restored
        fidelity_feature_weight = self.cnn_layers_weight[0](fidelity_map)
        restored_feature = 2.0 * torch.sigmoid(fidelity_feature_weight) * restored_feature
        i = 0
        for child in self.feature_extractor.children():
            if isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                i += 1 
                fidelity_feature_weight = self.cnn_layers_weight[i](fidelity_feature_weight) + fidelity_feature_weight
                fidelity_feature_weight = F.interpolate(fidelity_feature_weight, size=restored_feature.shape[-1], mode=self.downsample_mode, align_corners=False)
                restored_feature = 2.0 * torch.sigmoid(fidelity_feature_weight) * restored_feature
            restored_feature = child(restored_feature)
        restored_feature = torch.flatten(restored_feature, 1)
        
        # channel multiplication and channel concatenation:
        fidelity_feature = self.downsample(fidelity_map)
        fidelity_feature = torch.flatten(fidelity_feature, 1)
        restored_feature = restored_feature * 2.0 * torch.sigmoid(self.mul_fc(fidelity_feature))
        feature = self.cat_fc(torch.cat((restored_feature, fidelity_feature), -1))

        return self.classifier(feature)
    
    
class ChannelMultiplication(nn.Module):
    """ Original full model without channel multiplication.
    """
    def __init__(self,
                 mode: str,
                 restoration: Module,
                 fidelity_input: str,
                 fidelity_output: str,
                 feature_extractor: Module,
                 feature_size: int,
                 classifier: Module,
                 downsample: str='bilinear',
                 fidelity=None,
                 num_channel=4,
                 increase=0.25,
                 MEAN: list=[0.485, 0.456, 0.406],
                 STD: list=[0.229, 0.224, 0.225]
                ) -> None:
        
        super(ChannelMultiplication, self).__init__()
        
        self.mode = mode.lower()
        self.downsample_mode = downsample
        
        # prepare restoration network, feature extractor, classifier
        self.restoration = restoration
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.img_normal = Normalize(MEAN, STD)
        self.feature_size = feature_size
        
        # prepare fidelity map estimator
        self.fidelity_output = fidelity_output
        if fidelity is not None:
            self.fidelity = fidelity
            self.fidelity_input = fidelity_input
        if 'cos' in fidelity_output.lower():
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            in_channels = 1
        else:
            in_channels = 3
            
        # basic trainable module
        basic_cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=num_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=num_channel, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect')
                                 )
        # spatial multiplication and spatial addition
        cnn_layers = nn.ModuleList([copy.deepcopy(basic_cnn)])
        for child in feature_extractor.children():
            if isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                cnn_layers.append(copy.deepcopy(basic_cnn))
        self.cnn_layers_weight = cnn_layers
        self.cnn_layers_bias = copy.deepcopy(cnn_layers)
        
        if 'cos' not in self.fidelity_output.lower():
            sigma = math.sqrt((0.1**2+0.2**2+0.3**2+0.4**2+0.5**2) / (6.0**2 * 2.0))
            if 'l1' in self.fidelity_output.lower():
                mean = sigma * math.sqrt(2.0 / math.pi)
                std = sigma * math.sqrt(1.0- 2.0 / math.pi)
            elif 'l2' in self.fidelity_output.lower():
                mean = sigma**2.0
                std = math.sqrt(2.0) * sigma**2.0
            self.fidelity_normal = Normalize([mean]*in_channels, [std]*in_channels)
        
        # channel concatenation
        self.cat_fc = nn.Sequential(nn.Linear(feature_size * 2, int(feature_size * increase)),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(int(feature_size * increase), feature_size),
                                   )
        self.is_ensemble = False
    
    def downsample(self, x):
        
        x = x.mean(1, keepdim=True)
        y = x.permute(0, 1, 3, 2)
        
        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, 1, -1)
        x = F.interpolate(x, size=(1, self.feature_size//2), mode=self.downsample_mode, align_corners=False)
        
        batch_size, num_channels, height, width = y.shape
        y = y.reshape(batch_size, num_channels, 1, -1)
        y = F.interpolate(y, size=(1, self.feature_size//2), mode=self.downsample_mode, align_corners=False)
        
        return torch.cat((x, y), -1)
                                                      
    def forward(self, degraded, origins=None):
        
        # get restored image
        restored = self.restoration(degraded).clamp_(0, 1)
        
        # get fidelity map
        if 'oracle' in self.mode.lower():
            if 'l1' in self.fidelity_output.lower():
                fidelity_map = (restored-origins).abs()
            elif 'l2' in self.fidelity_output.lower():
                fidelity_map = (restored-origins).square()
            elif 'cos' in self.fidelity_output.lower():
                fidelity_map = self.cos(restored, origins)
                fidelity_map = fidelity_map.unsqueeze(1)
        else:                                            
            if 'degraded' in self.fidelity_input.lower():
                fidelity_map = degraded - self.fidelity(degraded)
            elif 'restored' in self.fidelity_input.lower():
                fidelity_map = restored - self.fidelity(restored)
        
        # normalize
        restored = self.img_normal(restored)
        if 'cos' not in self.fidelity_output.lower():
            fidelity_map = self.fidelity_normal(fidelity_map)
            fidelity_map = 1.0 - fidelity_map
        fidelity_map = fidelity_map.mean(dim=1, keepdim=True)
        
        # spatial multiplication and spatial addition
        restored_feature = restored
        fidelity_feature_weight = self.cnn_layers_weight[0](fidelity_map)
        fidelity_feature_bias = self.cnn_layers_bias[0](fidelity_map)
        restored_feature = 2.0 * torch.sigmoid(fidelity_feature_weight) * restored_feature + fidelity_feature_bias
        i = 0
        for child in self.feature_extractor.children():
            if isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                i += 1 
                fidelity_feature_weight = self.cnn_layers_weight[i](fidelity_feature_weight) + fidelity_feature_weight
                fidelity_feature_bias = self.cnn_layers_bias[i](fidelity_feature_bias) + fidelity_feature_bias
                fidelity_feature_weight = F.interpolate(fidelity_feature_weight, size=restored_feature.shape[-1], mode=self.downsample_mode, align_corners=False)
                fidelity_feature_bias = F.interpolate(fidelity_feature_bias, size=restored_feature.shape[-1], mode=self.downsample_mode, align_corners=False)
                restored_feature = 2.0 * torch.sigmoid(fidelity_feature_weight) * restored_feature + fidelity_feature_bias
            restored_feature = child(restored_feature)
        restored_feature = torch.flatten(restored_feature, 1)
        
        # channel multiplication and channel concatenation:
        fidelity_feature = self.downsample(fidelity_map)
        fidelity_feature = torch.flatten(fidelity_feature, 1)
        feature = self.cat_fc(torch.cat((restored_feature, fidelity_feature), -1))

        return self.classifier(feature)
    
    
class ChannelConcatenation(nn.Module):
    """ Original full model without channel concatenation.
    """
    def __init__(self,
                 mode: str,
                 restoration: Module,
                 fidelity_input: str,
                 fidelity_output: str,
                 feature_extractor: Module,
                 feature_size: int,
                 classifier: Module,
                 downsample: str='bilinear',
                 fidelity=None,
                 num_channel=4,
                 increase=0.25,
                 MEAN: list=[0.485, 0.456, 0.406],
                 STD: list=[0.229, 0.224, 0.225]
                ) -> None:
        
        super(ChannelConcatenation, self).__init__()
        
        self.mode = mode.lower()
        self.downsample_mode = downsample
        
        # prepare restoration network, feature extractor, classifier
        self.restoration = restoration
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.img_normal = Normalize(MEAN, STD)
        self.feature_size = feature_size
        
        # prepare fidelity map estimator
        self.fidelity_output = fidelity_output
        if fidelity is not None:
            self.fidelity = fidelity
            self.fidelity_input = fidelity_input
        if 'cos' in fidelity_output.lower():
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            in_channels = 1
        else:
            in_channels = 3
            
        # basic trainable module
        basic_cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=num_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=num_channel, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect')
                                 )
        # spatial multiplication and spatial addition
        cnn_layers = nn.ModuleList([copy.deepcopy(basic_cnn)])
        for child in feature_extractor.children():
            if isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                cnn_layers.append(copy.deepcopy(basic_cnn))
        self.cnn_layers_weight = cnn_layers
        self.cnn_layers_bias = copy.deepcopy(cnn_layers)
        
        if 'cos' not in self.fidelity_output.lower():
            sigma = math.sqrt((0.1**2+0.2**2+0.3**2+0.4**2+0.5**2) / (6.0**2 * 2.0))
            if 'l1' in self.fidelity_output.lower():
                mean = sigma * math.sqrt(2.0 / math.pi)
                std = sigma * math.sqrt(1.0- 2.0 / math.pi)
            elif 'l2' in self.fidelity_output.lower():
                mean = sigma**2.0
                std = math.sqrt(2.0) * sigma**2.0
            self.fidelity_normal = Normalize([mean]*in_channels, [std]*in_channels)
        
        # channel multiplication
        self.mul_fc = nn.Sequential(nn.Linear(feature_size, int(feature_size * increase)),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(int(feature_size * increase), feature_size),
                                   )
        self.is_ensemble = False
    
    def downsample(self, x):
        
        x = x.mean(1, keepdim=True)
        y = x.permute(0, 1, 3, 2)
        
        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, 1, -1)
        x = F.interpolate(x, size=(1, self.feature_size//2), mode=self.downsample_mode, align_corners=False)
        
        batch_size, num_channels, height, width = y.shape
        y = y.reshape(batch_size, num_channels, 1, -1)
        y = F.interpolate(y, size=(1, self.feature_size//2), mode=self.downsample_mode, align_corners=False)
        
        return torch.cat((x, y), -1)
                                                      
    def forward(self, degraded, origins=None):
        
        # get restored image
        restored = self.restoration(degraded).clamp_(0, 1)
        
        # get fidelity map
        if 'oracle' in self.mode.lower():
            if 'l1' in self.fidelity_output.lower():
                fidelity_map = (restored-origins).abs()
            elif 'l2' in self.fidelity_output.lower():
                fidelity_map = (restored-origins).square()
            elif 'cos' in self.fidelity_output.lower():
                fidelity_map = self.cos(restored, origins)
                fidelity_map = fidelity_map.unsqueeze(1)
        else:                                            
            if 'degraded' in self.fidelity_input.lower():
                fidelity_map = degraded - self.fidelity(degraded)
            elif 'restored' in self.fidelity_input.lower():
                fidelity_map = restored - self.fidelity(restored)
        
        # normalize
        restored = self.img_normal(restored)
        if 'cos' not in self.fidelity_output.lower():
            fidelity_map = self.fidelity_normal(fidelity_map)
            fidelity_map = 1.0 - fidelity_map
        fidelity_map = fidelity_map.mean(dim=1, keepdim=True)
        
        # spatial multiplication and spatial addition
        restored_feature = restored
        fidelity_feature_weight = self.cnn_layers_weight[0](fidelity_map)
        fidelity_feature_bias = self.cnn_layers_bias[0](fidelity_map)
        restored_feature = 2.0 * torch.sigmoid(fidelity_feature_weight) * restored_feature + fidelity_feature_bias
        i = 0
        for child in self.feature_extractor.children():
            if isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                i += 1 
                fidelity_feature_weight = self.cnn_layers_weight[i](fidelity_feature_weight) + fidelity_feature_weight
                fidelity_feature_bias = self.cnn_layers_bias[i](fidelity_feature_bias) + fidelity_feature_bias
                fidelity_feature_weight = F.interpolate(fidelity_feature_weight, size=restored_feature.shape[-1], mode=self.downsample_mode, align_corners=False)
                fidelity_feature_bias = F.interpolate(fidelity_feature_bias, size=restored_feature.shape[-1], mode=self.downsample_mode, align_corners=False)
                restored_feature = 2.0 * torch.sigmoid(fidelity_feature_weight) * restored_feature + fidelity_feature_bias
            restored_feature = child(restored_feature)
        restored_feature = torch.flatten(restored_feature, 1)
        
        # channel multiplication and channel concatenation:
        fidelity_feature = self.downsample(fidelity_map)
        fidelity_feature = torch.flatten(fidelity_feature, 1)
        restored_feature = restored_feature * 2.0 * torch.sigmoid(self.mul_fc(fidelity_feature))
        feature = restored_feature

        return self.classifier(feature)