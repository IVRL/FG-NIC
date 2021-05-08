#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : FG-NIC
# @Author       : Xiaoyu LIN
# @File         : classification.py
# @Description  : This file contains funtions for calssificaiton netowks.

from torch.nn import Module
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import random
import math

from util import set_parameter_requires_grad


def initialize_classification(model_name: str, 
                              num_classes: int, 
                              use_pretrained: bool =True
                             ) -> (Module, int): 
    """ Initialize these variables which will be set in this if statement. Each of these
            variables is model specific. The final fully-connected layer will fit the new number 
            of classes. The weights are initialized with the Xavier algorithm. All biases are 
            initialized to 0.
        Args:
            model_name (str): Classification network name in ['vgg', 'alexnet', 'resnet', 'googlenet'].
            num_classes (int): The number of classes in dataset. 
            use_pretrain (bool): If true, load pretrained model on ImageNet.
        Return:
            model (Module): Modified classification network fitting given class number.
            input_size (int): input image size for the classification network.
    """
    model = None
    input_size = None
    
    # VGG-16
    if "vgg" in model_name.lower():
        model = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model, True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        nn.init.xavier_uniform_(model.classifier[6].weight)
        nn.init.zeros_(model.classifier[6].bias)
        input_size = 224
    
    # Alexnet
    elif "alexnet" in model_name.lower():
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        nn.init.xavier_uniform_(model.classifier[6].weight)
        nn.init.zeros_(model.classifier[6].bias)
        input_size = 224
    
     # Resnet-50
    elif "resnet" in model_name.lower():
        if '18' in model_name.lower():
            model = models.resnet18(pretrained=use_pretrained)
        else:
            model = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model, True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)
        input_size = 224
    
    # GoogLeNet
    elif "googlenet" in model_name.lower():    
        model = models.googlenet(pretrained=use_pretrained, aux_logits=True)
        set_parameter_requires_grad(model, True)
        # Handle the auxilary network
        num_ftrs = model.aux1.fc2.in_features
        model.aux1.fc2 = nn.Linear(num_ftrs, num_classes)
        nn.init.xavier_uniform_(model.aux1.fc2.weight)
        nn.init.zeros_(model.aux1.fc2.bias)
        num_ftrs = model.aux2.fc2.in_features
        model.aux2.fc2 = nn.Linear(num_ftrs, num_classes)
        nn.init.xavier_uniform_(model.aux2.fc2.weight)
        nn.init.zeros_(model.aux2.fc2.bias)
        # Handle the primary network
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)
        input_size = 224

    else:
        raise ValueError("Invalid classification network name.")

    return model, input_size


def split_classification(model_name: str, 
                         model: Module
                        ) -> (Module, Module, int): 
    """ Split classification netwrok into two modules: feature extractor and classifier.
        Args:
            model_name (str): Classification network name in ['vgg', 'alexnet', 'resnet', 'googlenet'].
            num_classes (Module): Pytorch Module for the classification network.
        Return:
            feature_extractor (Module):  Pytorch Module for the feature extractor module.
            classifier (Module): Pytorch Module for the classifier module.
            feature_size (int): Length of the extracted feature by feature extractor.
    """
    
    # VGG
    if  "vgg" in model_name.lower():
        classifier = model.classifier
        model.classifier = Identity()
        feature_extractor = model
        feature_size = 25088
    
    # Alexnet
    elif "alexnet" in model_name.lower():
        
        classifier = model.classifier
        model.classifier = Identity()
        feature_extractor = model
        feature_size = 9216
    
    # Resnet
    elif "resnet" in model_name.lower():
        classifier = model.fc
        model.fc = Identity()
        feature_extractor = model
        if '18' in model_name.lower():
            feature_size = 512
        else:
            feature_size = 2048
    
    # GoogLeNet
    elif "googlenet" in model_name.lower():
        classifier = model.fc
        model.fc = Identity()
        model.aux1 = Identity()
        model.aux2 = Identity()
        feature_extractor = model
        feature_size = 1024
    else:
        raise ValueError("Invalid classification network name.")

    return feature_extractor, classifier, feature_size


class Identity(nn.Module):
    """ Identity module, directly give inputs to outputs without any manipulation.
    """
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def learning_rate_warmup(optimizer: optim.Optimizer, 
                         warmup: int, 
                         epoch: int, 
                         lr_init: float, 
                         batch: int, 
                         batch_num: int
                        ) -> None:
    """ Apply batch-wise learning rate warmup.
        Args:
            optimizer (torch.optim.Optimizer): Pytorch optimizer.
            warmup (int): The total number of epochs for learing rate warmup at beginning.
            epoch (int): Current epoch number.
            lr_init: Initial learning rate.
            batch（int):  Current batch number 
            num_epoch (int): Total number of batchs in an epoch.
    """ 
    lr = lr_init * (float(epoch) * float(batch_num) + float(batch)) / (float(warmup) * float(batch_num))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def cosine_learning_rate_decay(optimizer: optim.Optimizer, 
                               epoch: int, 
                               lr_init: float, 
                               num_epochs: int
                              ) -> None:
    """ Apply cosine learning rate decay.
        Args:
            optimizer (torch.optim.Optimizer): Pytorch optimizer.
            epoch (int): Current epoch.
            lr_init (float): Initial learning rate.
            num_epoch (int): Total number of epochs for the whole training process.
    """
    lr = (lr_init / 2.0) * (1.0 + math.cos(epoch*math.pi/num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

class LabelSmoothing(nn.Module):
    """ Apply label smoothing on original label
        Args:
            smoothing (float): label smoothing hyper-parameter epsilon
    """
    def __init__(self, 
                 smoothing: float = 0.1
                ) -> None:
        super(LabelSmoothing, self).__init__()
        
        self.eps = smoothing

    def forward(self, 
                pred: Tensor,
                target: Tensor
               ) -> Tensor:
        """ Label smoothing
            Args:
                pred (Tensor): prediction result from classifier with shape (N x C), where N is batch 
                    size and C is the number of classes.
                target (Tensor): label of each batch element with shape (N) and 
                    takes values from 0 to C.
            Return:
                output (Tensor): smoothed label with shape (N x C).
        """
        
        num_classes = pred.shape[-1]
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        output = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (num_classes - 1)
        
        return output