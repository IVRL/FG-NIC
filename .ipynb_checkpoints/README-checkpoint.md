# Fidelity Estimation Improves Noisy Image Classification with Pretrained Networks
**Authors**: Xiaoyu Lin, Deblina Bhattacharjee, Majed El Helou and Sabine Süsstrunk

![Python 3.7.9](https://img.shields.io/badge/python-3.7-blue.svg) 
![pytorch 1.6.0](https://img.shields.io/badge/pytorch-1.6.0-orange.svg)
![CUDA 10.2](https://img.shields.io/badge/cuda-10.2-green.svg)


#### [[Paper]](https://github.com/IVRL/FG-NIC) - [[Supplementary]](https://github.com/IVRL/FG-NIC)


> **Abstract:** *Deep learning has achieved significant improvements in recent years for many computer vision tasks, including image classification. This was propelled by large datasets and convolutional networks capable of learning rich feature extractors. However, such methods are developed on clean images and are not robust when handling noisy ones, despite a restoration preprocessing step. While novel methods were recently developed for tackling this problem, they rely on modified feature extractors. In other words, the feature extractor of the classifier needs to be retrained, which is computationally expensive. We propose a method that can be applied to a pretrained classifier. Our method exploits a fidelity map estimate that is fused into the internal feature representations. This fidelity map enables us to adjust the attention of the network, directing it towards features that are faithful to the clean image, and away from those affected by noise and restoration. Our noisy-image classification results improve over the baseline network by significantly large margins, especially at high noise levels, and come close to the fully-retrained approaches. Furthermore, using our oracle fidelity map, we show that our method even outperforms the fully-retrained methods, whether trained on noisy or restored images.*
>

## Degradation model
To explore the effects of degradation types and levels on classification networks, we also implement five types of degradation model: Additive white Gaussian noise, Salt and Pepper Noise, Gaussian Blur, Motion Blur and Rectangle Crop. The instruction of those degradatin models is given in [notebook](synthetic_images.ipynb). 

## Model Training and Testing

### Train Pre-trained models for proposed models
The implemention of classification networks are based on [torchvision](https://pytorch.org/docs/stable/torchvision/models.html), and restoration networks is based on [DnCNN](https://github.com/cszn/KAIR), [MemNet](https://github.com/IVRL/DEU).
- To obtain pre-trained classification neworks:  
`python train.py --task=classification --classifier=resnet50`

- To obtain pre-trained restoration neworks:
`python train.py --task=restoration --classifier=resnet50 --degradation=awgn --restoration=dncnn`

- To obtain pre-trained fidelity map estimator:
`python train.py --task=fidelity --degradation=awgn --restoration=dncnn fidelity_input=degraded --fidelity_output=l1`

### Proposed method
- To train the proposed method:

`python train.py --task=model --classifier=resnet50 --degradation=awgn --restoration=dncnn --mode=endtoend-pretrain`

The `mode` option can be `['endtoend-pretrain', 'pretrain', 'oracle']`

- To test the proposed method:

`python train.py --task=model --classifier=resnet50 --degradation=awgn --restoration=dncnn --mode=endtoend-pretrain`


## Citation

```bibtex

```