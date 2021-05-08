#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : FG-NIC
# @Author       : Xiaoyu LIN
# @File         : degradation.py
# @Description  : This file implementation five types of degradtion models, including: 
#                 Additive white Gaussian noise, Salt and Pepper Noise, Gaussian Blur, Motion Blur, Rectangle Crop.

from typing import Union
from torch import Tensor
from torch.nn import Module
import torch
import numpy as np
import random
import copy
import math
from torchvision import transforms

    
def get_degradation(degradation_type: str,
                    level: Union[float, tuple, list],
                    is_discrete = True,
                    vary: Union[str, int, tuple] = None,
                    phase: str = 'test',
                    input_size: int = 224,
                    task: str = 'classification',
                    patch_size: int = 50,
                    stride: int = 25,
                    orientation: Optional[float] = 45,
                   ) -> (object, Union[float, tuple]):
    """ Set up images manipulation to generate degraded images.
    Notes:
        This function take 'vary' input as input for 'patch_num' in rectangle crops.
    Args:
        degradation_type (str): Degardation type in ['awgn', 'gaussian-blur', 'motion-blur', 'salt-pepper', 'rectangle-crop']
        level (Union[float, tuple, list]): Degradation level. If it is float number, this is the uniform degradaion level.
            If it is a tuple, apply spatially varying degradiation levels, ranging from the first value to the second
            value linearly, depending on 'vary'.
            If it is a list, degradation level sampled from the list or range from max(level) to min(level), 
            depending on 'is_discrete'.
        is_discrete (bool): If true, degradation level sampled from set given by list of level;
            If false, degradation level sampled from range max(level) to min(level).
        vary (str or int or tuple): Set 2d or 1d spatially varying or the number of patches for rectangle crops.
        phase (str): Phase for diffierent argument methods, take value in ['train', 'test'].
        input_size (int): Network input images size.
        task (str): The current task, take value in ['classification', 'restoration', 'fidelity', 'model'].
        patch_size (int): Patch size to train restoration network and fidelity map.
        stride (int): Stride between patches.
        orientation (Optional[float]): If given, fixe the oritentation for motion blur.
    Returns:
        data_transforms (object): Data transforms.
        level (float or tuple): Simplified degradation level.
    """

    # Set degradation level
    if isinstance(level, tuple) and level[1] is None:
        level = float(level[0]) if degradation_type != 'motion-blur' else int(level[0])
    
    # Set orientatin for motion blur
    orientation = float(orientation) if orientation is not None else None

    # Data augmentation
    if phase == 'test' or phase =='valid':
        transforms_list = [transforms.Resize(input_size),
                           transforms.CenterCrop(input_size),
                           transforms.ToTensor()
                          ]
    elif phase == 'train':
        transforms_list = [transforms.RandomResizedCrop(input_size),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]
    else: 
        print("Invalid transfrom phase")
        exit()
    
    if 'restoration' in task.lower() or 'fidelity' in task.lower():
        transforms_list.append(ImageToPatch(patch_size=patch_size, stride=stride))
    
    # Degradation model implemention
    # Clean image
    if degradation_type == 'clean':
        transforms_list.append(Clean())
    
    # Additive white Gaussian noise
    elif degradation_type == 'awgn':
        transforms_list.append(AdditiveWhiteGaussianNoise(level, vary, is_discrete))
    
    # Salt and Pepper Noise
    elif degradation_type == 'salt-pepper': 
        transforms_list.append(SaltAndPepperNoise(level, vary))
    
    # Gaussian Blur
    elif degradation_type == 'gaussian-blur':
        transforms_list.append(GaussianBlur(level, vary, stride=5, kernel_size=13))
    
    # Motion Blur
    elif degradation_type == 'motion-blur':
       
        transforms_list.append(MotionBlur(level, vary, stride=5, orientation=orientation))
    
    # Rectangle Crop
    elif degradation_type == 'rectangle-crop':
        transforms_list.append(RectangleCrop(level, patch_num=vary))
        
    else:
        raise ValueError("Invalid degradation type!")

    data_transforms = transforms.Compose(transforms_list)
    
    return data_transforms, level


class ImageToPatch(object):
    """ Crop image into small square patches.
    Args:
        patch_size (int): The size of patch square (side length).
        stride (int): The stride between each crop.
    """
    def __init__(self, 
                 patch_size: int = 50,
                 stride: int = 25
                ) -> None:
        
        self.patch_size = patch_size
        self.stride = stride
    
    def __call__(self, 
                 img: Tensor
                ) -> (Tensor, Tensor):
        """
        Args:
            image (Tensor): Tensor image of size (C, H, W).
        Returns:
            output (Tensor): The stacked image patch (number_of_patches, C, patch_size, patch_size)
        """
        img = img.unsqueeze(0)
        patches = img[ :, :, :, :self.patch_size]
        for i in range(self.stride, img.shape[3] - self.patch_size, self.stride):
            patches = torch.cat((patches, img[:, :, :, i:i+self.patch_size]), 0)
        patches = torch.cat((patches, img[ :, :, :, -self.patch_size:]), 0)
        outputs = patches[ :, :, :self.patch_size, :]
        for i in range(self.stride, img.shape[2] - self.patch_size, self.stride):
            outputs = torch.cat((outputs, patches[ :, :, i:i+self.patch_size, :]), 0)
        outputs = torch.cat((outputs, patches[ :, :, -self.patch_size:, :]), 0)
        
        return outputs

class Clean(object):
    """ Clean images and its copy.
    """
    
    def __call__(self, 
                 img: Tensor
                ) -> (Tensor, Tensor):
        """
        Args:
            image (Tensor): Tensor image of size (C, H, W).
        Returns:
            output (Tensor): The original image.
            origin (Tensor): The original image.
        """
        
        origin = copy.deepcopy(img)
        output = img
        
        return output, origin


class AdditiveWhiteGaussianNoise(object):
    """ Add additive white Gaussiona noise on the origianl image.
    Args:
        level (Union[float, tuple, list]): If it is a float number, this is the uniform degradaion level on the image. 
            If it is a tuple, apply the degradation with varying degradiation levels, ranging from the first value to the second
            value linearly change from a random point to the corner with maximum Euclidean distance to that point, or increases 
            linearly with the number of rows or column in the image.
            If it is a list, degradation level sampled from the list or range from max(level) to min(level), 
            depending on 'is_discrete'.
        is_discrete (bool): If true, degradation level sampled from set given by list of level;
            If false, degradation level sampled from range max(level) to min(level).
        vary (str): take value in ['2d', '1d'] If it is '2d', degradation level linearly change w.r.t Euclidean 
            distance; If it is '1d', degradaion level increases linearly with the number of rows (or column randomly).
    """

    def __init__(self, 
                 level: Union[float, tuple, list],
                 vary: str = '2d',
                 is_discrete: bool = True,
                 change: str = None
                ) -> None:
        
        self.level = level
        self.vary = vary
        self.is_discrete = is_discrete
        self.change = change

    def __call__(self, 
                 img: Tensor
                ) -> (Tensor, Tensor):
        """
        Args:
            image (Tensor): Tensor image of size (C, H, W) or (P, C, H, W), where P is the number of patches per image.
        Returns:
            output (Tensor): The degraded image with additive white Gaussiona noise.
            origin (Tensor): The original image.
        """
        
        origin = copy.deepcopy(img)
        noise = torch.FloatTensor(img.shape).zero_()
        
        # Spatially varying only for inputs (C, H, W)
        if isinstance(self.level, tuple):
            self.level = (float(self.level[0]), float(self.level[1]))
            
            # 2d spatially varying: varying w.r.t Euclidean distance
            if '2d' in self.vary.lower():
                # linearly increase or decrease
                if self.change and 'decrease' in self.change.lower():
                    self.level = self.level
                elif self.change and 'increase' in self.change.lower():
                    self.level = (self.level[1], self.level[0])
                else:
                    self.level = self.level if random.random() < 0.5 else (self.level[1], self.level[0])
                pix_size = torch.Size([img.shape[0]])
                random_point = (np.random.randint(img.shape[1]), np.random.randint(img.shape[2]))
                distance_max = np.sqrt(np.max([(img.shape[1]-1-random_point[0])**2 + (img.shape[2]-1-random_point[1])**2,
                                               (random_point[0])**2 + (img.shape[2]-1-random_point[1])**2,
                                               (img.shape[1]-1-random_point[0])**2 + (random_point[1])**2,
                                               (random_point[0])**2 + (random_point[1])**2]))
                decrease_rate = (self.level[0] - self.level[1]) / (distance_max * 1.0)
                for row in range(img.shape[1]):
                    for column in range(img.shape[2]):
                        degradation_level = self.level[1] + decrease_rate * (
                            np.sqrt((row - random_point[0]) ** 2 + (column - random_point[1]) ** 2) * 1.0)
                        if degradation_level > 0:
                            noise[:, row, column] = torch.FloatTensor(pix_size).normal_(mean=0, std=degradation_level)
            
            # 1d spatially varying: varying w.r.t row or column randomly
            elif '1d' in self.vary.lower():
                # linearly increase or decrease
                self.level = self.level if random.random() < 0.5 else (self.level[1], self.level[0])
                # change from left to right or bottom to up
                (dim, size) = (1, 2) if random.random() < 0.5 else (2, 1)
                dim_size = torch.Size([img.shape[0], img.shape[size]])
                for i in range(img.shape[dim]):
                    degradation_level = self.level[1] + (self.level[0]-self.level[1]) * (i/(img.shape[dim]*1.0-1))
                    if degradation_level > 0:
                        if dim == 1:
                            noise[:,i,:] = torch.FloatTensor(dim_size).normal_(mean=0, std=degradation_level)
                        else:
                            noise[:,:,i] = torch.FloatTensor(dim_size).normal_(mean=0, std=degradation_level)
                            
        # Uniform degradation
        else:
            # original image (C, H, W)
            if img.dim() == 3:
                if isinstance(self.level, float):
                    level = self.level
                elif self.is_discrete:
                    level = np.random.choice(np.array(self.level))
                else:
                    level = np.random.uniform(min(self.level), max(self.level))
                if level > 0:
                    noise = torch.FloatTensor(img.shape).normal_(mean=0, std=float(level))
            
            # patch image (P, C, H, W)
            else:
                if isinstance(self.level, float):
                    stdN = self.level
                elif self.is_discrete:
                    stdN = np.random.choice(np.array(self.level), size=noise.size()[0])
                else:
                    stdN = np.random.uniform(min(self.level), max(self.level), size=noise.size()[0])
                sizeN = noise[0,:,:,:].size() 
                for n in range(noise.size()[0]):
                    if stdN[n] > 0:
                        noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n])
        
        output = img + noise
        output = output.clamp_(0.0, 1.0)
        
        return output, origin

    
class SaltAndPepperNoise(object):
    """ Add salt and pepper noise to the origianl image.
    Args:
        level (float or tuple): If it is a float number, this is the uniform degradaion level on the image. If it is a tuple,
            apply the degradation with varying degradiation levels, ranging from the first value to the second value linearly
            change from a random point to the corner with maximum Euclidean distance to that point, or increases linearly with 
            the number of rows or column in the image.
        vary (str): ['1d', '2d'] If it is '2d', degradation level linearly change w.r.t Euclidean distance; 
            If it is '1d', degradaion level increases linearly with the number of rows or column randomly.
    """

    def __init__(self, 
                 level: Union[float, tuple],
                 vary: str = '2d'
                ) -> None:
        
        self.level = level
        self.vary = vary

    def __call__(self, 
                 img: Tensor
                ) -> (Tensor, Tensor):
        """
        Args:
            image (Tensor): Tensor image of size (C, H, W) or (P, C, H, W).
        Returns:
            output (Tensor): The degraded image with salt and pepper noise.
            origin (Tensor): The original image.
        """
        
        origin = copy.deepcopy(img)
        output = copy.deepcopy(img)
        noise = torch.empty(img.shape).uniform_(0, 1)
        
        # Spatially varying only for inputs (C, H, W)
        if isinstance(self.level, tuple):
            
            # 2d spatially varying: varying w.r.t Euclidean distance
            if '2d' in self.vary.lower():
                # linearly increase or decrease
                self.level = self.level if random.random() < 0.5 else (self.level[1], self.level[0])

                random_point = (np.random.randint(img.shape[1]), np.random.randint(img.shape[2]))
                distance_max = np.sqrt(np.max([(img.shape[1]-1-random_point[0])**2 + (img.shape[2]-1-random_point[1])**2,
                                               (random_point[0])**2 + (img.shape[2]-1-random_point[1])**2,
                                               (img.shape[1]-1-random_point[0])**2 + (random_point[1])**2,
                                               (random_point[0])**2 + (random_point[1])**2]))
                decrease_rate = (self.level[0] - self.level[1]) / (distance_max * 1.0)
                for row in range(img.shape[1]):
                    for column in range(img.shape[2]):
                        degradation_level = self.level[1] + decrease_rate * (
                            np.sqrt((row - random_point[0]) ** 2 + (column - random_point[1]) ** 2) * 1.0)
                        noise[:,row,column] = torch.where(noise[:,row,column]<degradation_level/2.0, 
                                                          torch.zeros_like(noise[:,row,column]), noise[:,row,column])
                        noise[:,row,column] = torch.where(noise[:,row,column]>1.0-degradation_level/2.0,
                                                          torch.ones_like(noise[:,row,column]), noise[:,row,column])
                                
            # 1d spatially varying: varying w.r.t row or column randomly
            elif '1d' in self.vary.lower():
                # linearly increase or decrease
                self.level = self.level if random.random() < 0.5 else (self.level[1], self.level[0])
                # change from left to right or bottom to up
                (dim, size) = (1, 2) if random.random() < 0.5 else (2, 1)
                for i in range(img.shape[dim]):
                    degradation_level = self.level[1] + (self.level[0]-self.level[1]) * (i/(img.shape[dim]*1.0-1))
                    if dim == 1:
                        noise[:,i,:] = torch.where(noise[:,i,:]<degradation_level/2.0, 
                                                   torch.zeros_like(noise[:,i,:]), noise[:,i,:])
                        noise[:,i,:] = torch.where(noise[:,i,:]>1.0-degradation_level/2.0, 
                                                       torch.ones_like(noise[:,i,:]), noise[:,i,:])
                    else:
                        noise[:,:,i] = torch.where(noise[:,:,i]<degradation_level/2.0, 
                                                   torch.zeros_like(noise[:,:,i]), noise[:,:,i])
                        noise[:,:,i] = torch.where(noise[:,:,i]>1.0-degradation_level/2.0,
                                                       torch.ones_like(noise[:,:,i]), noise[:,:,i])
        
        # Uniform degradation
        else:
            # original image (C, H, W)
            if img.dim() == 3:
                level = float(random.choice(self.level)) if isinstance(self.level, list) else self.level
                noise = torch.where(noise<level/2.0, torch.zeros_like(noise), noise)
                noise = torch.where(noise>1.0-level/2.0, torch.ones_like(noise), noise)
            # patch image (P, C, H, W)
            else:
                levelN = np.random.uniform(min(self.level), max(self.level), size=noise.size()[0])
                for n in range(noise.size()[0]):
                    noise[n,:,:,:] = torch.where(noise[n,:,:,:]<levelN[n]/2.0, torch.zeros_like(noise[n,:,:,:]), noise[n,:,:,:])
                    noise[n,:,:,:] = torch.where(noise[n,:,:,:]>1.0-levelN[n]/2.0, torch.ones_like(noise[n,:,:,:]), noise[n,:,:,:])
        
        output = torch.where(noise==1, noise, img)
        output = torch.where(noise==0, noise, output)
        output = output.clamp_(0, 1)
        
        return output, origin
    

def gaussian_kernel(sigma: float, 
                    kernel_size: int, 
                    channels: int = 3
                   ) -> Tensor:
        """ Generate Gaussian kernal for given std (sigma) and kernel size.
        Note:
            Based on 
            https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/2.
        Args:
            sigma (float): Standard deviation for Gaussian distribution.
            kernel_size (int): Kernel size for generated Gaussian kernel, should be odd.
            channels (int): The number of channels for an image, 3 for RGB color image.
        Returns:
            gaussian_kernel (Tensor): Generated Gaussian kernel with size (channels, 1, kernel_size, kernel_size)
        """
        if sigma == 0:
            gaussian_kernel = torch.zeros([channels, 1, kernel_size, kernel_size])
            mean = int((kernel_size - 1)/2)
            gaussian_kernel[:, :, mean, mean] = 1
        else:
            x_cord = torch.arange(kernel_size)
            x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1)

            mean = (kernel_size - 1)/2.0
            variance = sigma**2.0

            # Calculate the 2-dimensional gaussian kernel which is the product of two gaussian 
            #   distributions for two different variables (in this case called x and y)
            gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)/(2.*variance))

            # Make sure sum of values in gaussian kernel equals 1.
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

            # Reshape to 2d depthwise convolutional weight
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
            gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        return gaussian_kernel

    
class GaussianBlur(object):
    """ Apply Gaussiona blur to the origianl image.
    Args:
        level (float or tuple): If it is a float number, this is the uniform degradaion level on the image. If it is a tuple,
            apply the degradation with varying degradiation levels, ranging from the first value to the second value linearly
            change from a random point to the corner with maximum Euclidean distance to that point, or increases linearly with 
            the number of rows or column in the image.
        vary (str): ['1d', '2d'] If it is '2d', degradation level linearly change w.r.t Euclidean distance; 
            If it is '1d', degradaion level increases linearly with the number of rows or column randomly.
        stride (int): The step size for spatially varying condition. 
        kernel_size (int): Kernel size for generated Gaussian kernel, should be odd.
        channels (int): The number of channels for an image, 3 for RGB color image.
    """

    def __init__(self, 
                 level: Union[float, tuple],
                 vary: str = '2d',
                 stride: int = 3,
                 kernel_size: int = 13, 
                 channels: int = 3
                ) -> None:
        
        self.level = level
        if kernel_size % 2 != 1:
            raise ValueError("Please input an odd kernel size!")
        self.kernel_size = kernel_size
        self.channels = channels
        self.stride = stride
        self.vary = vary

    def __call__(self, 
                 img: Tensor
                ) -> (Tensor, Tensor):
        """
        Args:
            image (Tensor): Tensor image of size (C, H, W).
        Returns:
            output (Tensor): The degraded image with Gaussiona blur.
            origin (Tensor): The original image.
        """
        
        original = copy.deepcopy(img)
        self.img = img
        output = img
        
        convolution_layer = torch.nn.Conv2d(in_channels=self.channels, out_channels=self.channels, 
                                            kernel_size=self.kernel_size, groups=self.channels, bias=False,
                                            padding=(self.kernel_size//2, self.kernel_size//2), padding_mode='reflect')
        
        # Spatially varying
        if isinstance(self.level, tuple):
            
            # 2d spatially varying: varying w.r.t Euclidean distance
            if '2d' in self.vary.lower():
                random_point = (np.random.randint(img.shape[1]), np.random.randint(img.shape[2]))
                distance_max = np.sqrt(np.max([(img.shape[1]-1-random_point[0])**2 + (img.shape[2]-1-random_point[1])**2,
                                               (random_point[0])**2 + (img.shape[2]-1-random_point[1])**2,
                                               (img.shape[1]-1-random_point[0])**2 + (random_point[1])**2,
                                               (random_point[0])**2 + (random_point[1])**2]))
                decrease_rate = (self.level[0] - self.level[1]) / (distance_max*1.0)
                            
                for y in range(0, img.shape[2], self.stride):
                    for x in range(0, img.shape[1], self.stride):
                        column = y + self.stride / 2.0 if y + self.stride < img.shape[2] else (y + img.shape[2]) / 2.0
                        row = x + self.stride / 2.0 if x + self.stride < img.shape[1] else (x + img.shape[1]) / 2.0
                        degradation_level = self.level[1] + decrease_rate * (np.sqrt((row-random_point[0])**2
                                                                              +(column-random_point[1])**2)*1.0)
                        if degradation_level > 0:
                            convolution_layer.weight.data = gaussian_kernel(degradation_level, self.kernel_size, self.channels)
                            convolution_layer.weight.requires_grad = False
                            blurred_image = convolution_layer(img.unsqueeze(0))[0]
                            output[:,x:min(img.shape[1], x+self.stride),y:min(img.shape[2], y+self.stride)] = \
                                blurred_image[:,x:min(img.shape[1], x+self.stride),y:min(img.shape[2], y+self.stride)] 
                        
            # 1d spatially varying: varying w.r.t row or column randomly
            elif '1d' in self.vary.lower():
                self.level = self.level if random.random() < 0.5 else (self.level[1], self.level[0])
                dim = 1 if random.random() < 0.5 else 2
                for i in range(0, img.shape[dim], self.stride):
                    center = i + self.stride / 2.0 if i + self.stride < img.shape[dim] else (i + img.shape[dim]) / 2.0
                    degradation_level = self.level[1] + (self.level[0]-self.level[1]) * (center/(img.shape[dim]*1.0-1))
                    if degradation_level > 0:
                        convolution_layer.weight.data = gaussian_kernel(degradation_level, self.kernel_size, self.channels)
                        convolution_layer.weight.requires_grad = False
                        blurred = convolution_layer(img.unsqueeze(0))[0]
                        if dim == 1:
                            output[:,i:min(img.shape[dim], i+self.stride),:] = blurred[:,i:min(img.shape[dim], i+self.stride),:]
                        else:
                            output[:,:,i:min(img.shape[dim], i+self.stride)] = blurred[:,:,i:min(img.shape[dim], i+self.stride)]
                                            
        # Uniform degradation       
        else:
            level = float(random.choice(self.level)) if isinstance(self.level, list) else self.level
            if level > 0:
                convolution_layer.weight.data = gaussian_kernel(level, self.kernel_size, self.channels)
                convolution_layer.weight.requires_grad = False
                output = convolution_layer(img.unsqueeze(0))[0]
        
        output = output.clamp_(0, 1)
        
        return output, original
    

def motion_kernel(orientation: float, 
                  length: int, 
                  channels: int = 3
                 ) -> Tensor:
        """ Generate motion kernal for given orientation and length.
        Args:
            orientation (float): Orientation of motion in [0, 180) degree.
            length (int): Kernel size of the motion kernal, should be odd.
            channels (int): The number of channels for an image, 3 for RGB color image.
        Returns:
            motion_kernel (Tensor): Generated motion kernel with size (channels, 1, length, length)
        """
        length = int(length)
        radius = (length + 1)//2 
        degree = 180 - orientation if orientation > 90 else orientation
        
        if degree > 0:
            x_cord = np.arange(radius)
            x_cord = x_cord * math.tan(degree * math.pi / 180.0)
            x_cord = np.around(x_cord).tolist()
            x_cord = [int(i) for i in x_cord]
            x_cord.append(x_cord[radius-1] if x_cord[radius-1] < radius-1 else radius)
            quarter = torch.zeros([radius, radius])
            for i in range(radius):
                quarter[x_cord[i]:x_cord[i+1], i] = 1

            y_cord = np.arange(radius)
            y_cord = y_cord * 1.0 / math.tan(degree * math.pi / 180.0)
            y_cord = np.around(y_cord).tolist()
            y_cord = [int(i) for i in y_cord]
            y_cord.append(y_cord[radius-1] if y_cord[radius-1] < radius-1 else radius)
            for i in range(radius):
                quarter[i, y_cord[i]:y_cord[i+1]] = 1
        else:
            quarter = torch.zeros([radius, radius])
            quarter[0, :] = 1
        
        motion_kernel = torch.zeros((length,length))
        if orientation < 90:
            motion_kernel[radius-1:,radius-1:] = quarter
            motion_kernel[:radius,:radius] = quarter.flip(0).flip(-1)
        else:
            quarter = quarter.flip(-1)
            motion_kernel[radius-1:,:radius] = quarter
            motion_kernel[:radius,radius-1:] = quarter.flip(0).flip(-1)
        
        # Make sure sum of values in gaussian kernel equals 1.
        motion_kernel = motion_kernel / torch.sum(motion_kernel)

        # Reshape to 2d depthwise convolutional weight
        motion_kernel = motion_kernel.view(1, 1, length, length)
        motion_kernel = motion_kernel.repeat(channels, 1, 1, 1)
        
        return motion_kernel
    

class MotionBlur(object):
    """ Apply motion blur to the origianl image.
    Note:
        For spatially varying situation (both 1D or 2D), the motion orientation will also change randomly, but for uniformly
            degraded situation, the orientation will keep the same.
    Args:
        level (int or tuple): If it is a float number, this is the uniform degradaion level on the image. If it is a tuple,
            apply the degradation with varying degradiation levels, ranging from the first value to the second value linearly
            change from a random point to the corner with maximum Euclidean distance to that point, or increases linearly with 
            the number of rows or column in the image.
        vary (str): ['1d', '2d'] If it is '2d', degradation level (kernel length) linearly change w.r.t Euclidean distance; 
            If it is '1d', degradaion level increases linearly with the number of rows or column randomly.
        orientation (Optional[float]): If given, fixe the oritentation for motion blur.
        stride (int): The step size for spatially varying condition. 
        channels (int): The number of channels for an image, 3 for RGB color image.
    """

    def __init__(self, 
                 level: Union[int, tuple],
                 vary: str = '2d',
                 orientation: Optional[float] = None,
                 stride: int = 3,
                 channels: int = 3,
                ) -> None:
        
        if isinstance(level, tuple) and (level[0] % 2 != 1 or level[1] % 2 != 1) and (level[0]*leve1[1]>0):
            raise ValueError("Please input an odd level!")
        if isinstance(level,int) and level % 2 != 1 and level>0:
            raise ValueError("Please input an odd level!")
        self.level = level
        self.channels = channels
        self.stride = stride
        self.vary = vary
        self.orientation = orientation

    def __call__(self, 
                 img: Tensor
                ) -> (Tensor, Tensor):
        """
        Args:
            image (Tensor): Tensor image of size (C, H, W).
        Returns:
            output (Tensor): The degraded image with motion blur.
            origin (Tensor): The original image.
        """
        
        origin = copy.deepcopy(img)
        self.img = img
        output = img
        
        # Spatially varying
        if isinstance(self.level, tuple):
            
            # 1d spatially varying: varying w.r.t row or column randomly
            if '2d' in self.vary.lower():
                random_point = (np.random.randint(img.shape[1]), np.random.randint(img.shape[2]))
                distance_max = np.sqrt(np.max([(img.shape[1]-1-random_point[0])**2 + (img.shape[2]-1-random_point[1])**2,
                                               (random_point[0])**2 + (img.shape[2]-1-random_point[1])**2,
                                               (img.shape[1]-1-random_point[0])**2 + (random_point[1])**2,
                                               (random_point[0])**2 + (random_point[1])**2]))
                decrease_rate = (self.level[0] - self.level[1]) / (distance_max*1.0)
                            
                for y in range(0, img.shape[2], self.stride):
                    for x in range(0, img.shape[1], self.stride):
                        column = y + self.stride / 2.0 if y + self.stride < img.shape[2] else (y + img.shape[2]) / 2.0
                        row = x + self.stride / 2.0 if x + self.stride < img.shape[1] else (x + img.shape[1]) / 2.0
                        degradation_level = self.level[1] + decrease_rate * (np.sqrt((row-random_point[0])**2
                                                                              +(column-random_point[1])**2)*1.0)
                        degradation_level = int(degradation_level)
                        degradation_level = degradation_level + 1 if degradation_level % 2 == 0 else degradation_level
                        convolution_layer = torch.nn.Conv2d(in_channels=self.channels, out_channels=self.channels, 
                                                            kernel_size=degradation_level, groups=self.channels, bias=False,
                                                            padding=(degradation_level//2, degradation_level//2), 
                                                            padding_mode='reflect')
                        orientation = random.random() * 180.0 if self.orientation is None else self.orientation
                        convolution_layer.weight.data = motion_kernel(orientation, degradation_level, self.channels)
                        convolution_layer.weight.requires_grad = False
                        blurred_image = convolution_layer(img.unsqueeze(0))[0]
                        output[:,x:min(img.shape[1], x+self.stride),y:min(img.shape[2], y+self.stride)] = \
                            blurred_image[:,x:min(img.shape[1], x+self.stride),y:min(img.shape[2], y+self.stride)] 
                        
            # 1d spatially varying: varying w.r.t row or column randomly
            elif '1d' in self.vary.lower():
                self.level = self.level if random.random() < 0.5 else (self.level[1], self.level[0])
                dim = 1 if random.random() < 0.5 else 2
                for i in range(0, img.shape[dim], self.stride):
                    center = i + self.stride / 2.0 if i + self.stride < img.shape[dim] else (i + img.shape[dim]) / 2.0
                    degradation_level = self.level[1] + (self.level[0]-self.level[1]) * (center/(img.shape[dim]*1.0-1))
                    degradation_level = int(degradation_level)
                    degradation_level = degradation_level + 1 if degradation_level % 2 == 0 else degradation_level
                    convolution_layer = torch.nn.Conv2d(in_channels=self.channels, out_channels=self.channels, 
                                                        kernel_size=degradation_level, groups=self.channels, bias=False,
                                                        padding=(degradation_level//2, degradation_level//2), 
                                                        padding_mode='reflect')
                    orientation = random.random() * 180.0 if self.orientation is None else self.orientation
                    convolution_layer.weight.data = motion_kernel(orientation, degradation_level, self.channels)
                    convolution_layer.weight.requires_grad = False
                    blurred = convolution_layer(img.unsqueeze(0))[0]
                    if dim == 1:
                        output[:,i:min(img.shape[dim], i+self.stride),:] = blurred[:,i:min(img.shape[dim], i+self.stride),:]
                    else:
                        output[:,:,i:min(img.shape[dim], i+self.stride)] = blurred[:,:,i:min(img.shape[dim], i+self.stride)]
                        
        # Uniform degradation
        else:
            level = int(random.choice(self.level)) if isinstance(self.level, list) else self.level
            if level > 0:
                convolution_layer = torch.nn.Conv2d(in_channels=self.channels, out_channels=self.channels, 
                                                    kernel_size=level, groups=self.channels, bias=False,
                                                    padding=(level//2, level//2), padding_mode='reflect')

                orientation = random.random() * 180.0 if self.orientation is None else self.orientation
                convolution_layer.weight.data = motion_kernel(orientation, level, self.channels)
                convolution_layer.weight.requires_grad = False
                output = convolution_layer(img.unsqueeze(0))[0]

        output = output.clamp_(0, 1)
        
        return output, origin
    

class RectangleCrop(object):
    """ Remove rectangle patch from the origianl image.
    Note:
        In uniform degradation condation, the croped patch is a square with fixed ratio of side length of croped square 
            to the minimum side length of original image. In spatially varying condation, the same ratio for length and 
            width are independently sampled, so the cropped patch could be a rectangle.
    Args:
        level (float or tuple): If it is a float number, this is the ratio of side length of croped square to the minimum
            side length of original image. If it is a tuple, the ratio will varys from the maxmum tuple value and the minmum
            value, and independtly for length and width.
        patch_num (float or int): If it is a int number, this is the number of patches croped from the image. If it is a 
            tuple, the number of patches will varies from the minmum value to maximum value.
    """

    def __init__(self, 
                 level: Union[float, tuple], 
                 patch_num: Union[int, tuple] = 1
                ) -> None:
        
        self.ratio = level
        self.patch_num = patch_num

    def __call__(self, 
                 img: Tensor
                ) -> (Tensor, Tensor):
        """
        Args:
            image (Tensor): Tensor image of size (C, H, W).
        Returns:
            output (Tensor): The degraded image with rectangle patches croped.
            origin (Tensor): The original image.
        """
        
        origin = copy.deepcopy(img)
        total_patches = random.randint(min(self.patch_num),max(self.patch_num)) if isinstance(self.patch_num, tuple) else int(self.patch_num)
        mask = torch.zeros_like(img)
        
        valid_crop = False
        while not valid_crop:
            for _ in range(total_patches):
                valid_patch = False
                while not valid_patch:
                    point_x, point_y = random.randint(0, img.shape[1]), random.randint(0, img.shape[2])
                    if isinstance(self.ratio, tuple):
                        ratio_x = min(self.ratio) + random.random() * (max(self.ratio)-min(self.ratio))
                        ratio_y = min(self.ratio) + random.random() * (max(self.ratio)-min(self.ratio))
                    else:
                        ratio = float(random.choice(self.ratio)) if isinstance(self.ratio, list) else self.ratio
                        ratio_x, ratio_y = float(ratio), float(ratio)
                    stretch_x = int(min(img.shape[1], img.shape[2]) * ratio_x // 2)
                    stretch_y = int(min(img.shape[1], img.shape[2]) * ratio_y // 2)
                    if point_x - stretch_x >= 0 and point_y - stretch_y >= 0:
                        if point_x + stretch_x <img.shape[1] and point_y + stretch_y < img.shape[2]:
                            valid_patch = True
                            mask[:,point_x-stretch_x:point_x+stretch_x, point_y-stretch_y:point_y+stretch_y] = 1
                        
            mask = mask * (-1) + torch.ones_like(img)
            output = img * mask
            output = output.clamp_(0, 1)
            if (origin - output).sum().item() > 0 or min(ratio_x, ratio_y) == 0:
                valid_crop = True
        
        return output, origin

    
def normalize(tensor: Tensor, 
              mean: List[float], 
              std: List[float], 
              inplace: bool = False
             ) -> Tensor:
    """ Normalize a tensor image with mean and standard deviation.
    Note:
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
        See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


class Normalize(Module):
    """ Normalize a tensor image with mean and standard deviation.
        Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
        channels, this transform will normalize each channel of the input
        ``torch.*Tensor`` i.e.,
        ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    Note:
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
   