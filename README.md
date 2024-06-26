# Fidelity Estimation Improves Noisy-Image Classification with Pretrained Networks
![Python 3.7.9](https://img.shields.io/badge/python-3.7-blue.svg) 
![pytorch 1.6.0](https://img.shields.io/badge/pytorch-1.6.0-orange.svg)
![CUDA 10.2](https://img.shields.io/badge/cuda-10.2-green.svg)

Published in IEEE Signal Processing Letters, 2021. [![DOI](https://zenodo.org/badge/340723861.svg)](https://zenodo.org/doi/10.5281/zenodo.11069080)

#### -- Our frequency-domain insights are based on [SFM](https://github.com/majedelhelou/SFM) and the fidelity concept is inspired by [BUIFD](https://github.com/majedelhelou/BUIFD) and [BIGPrior](https://github.com/majedelhelou/BIGPrior) --

#### [[Paper]](https://arxiv.org/abs/2106.00673) - [[Supplementary]](https://github.com/IVRL/FG-NIC/blob/main/materials/supp.pdf)



> **Abstract:** *Image classification has significantly improved using deep learning. This is mainly due to convolutional neural networks (CNNs) that are capable of learning rich feature extractors from large datasets. However, most deep learning classification methods are trained on clean images and are not robust when handling noisy ones, even if a restoration preprocessing step is applied. 
While novel methods address this problem, they rely on modified feature extractors and thus necessitate retraining. 
We instead propose a method that can be applied on a pretrained classifier. Our method exploits a fidelity map estimate that is fused into the internal representations of the feature extractor, thereby guiding the attention of the network and making it more robust to noisy data. 
We improve the noisy-image classification (NIC) results by significantly large margins, especially at high noise levels, and come close to the fully retrained approaches. Furthermore, as proof of concept, we show that when using our oracle fidelity map we even outperform the fully retrained methods, whether trained on noisy or restored images.*
>

## Table of Contents  
- [Degradation Model](#degradation-model)
- [Requirements](#requirements)
- [Model Training and Testing](#model-training-and-testing)
- [Baseline Methods and Ablation Study](#baseline-methods-and-ablation-study)
- [Results](#results)
- [Citation](#citation)

## Degradation Model
To explore the effects of degradation types and levels on classification networks, we also implement five types of degradation models: Additive white Gaussian noise (AWGN), Salt and Pepper Noise, Gaussian Blur, Motion Blur and Rectangle Crop. The instructions for those degradatin models are given in this [notebook](synthetic_images.ipynb).  

## Requirements
- Python 3.7, PyTorch 2.1.0;
- Other common packages listed in [`requirements.txt`](requirements.txt) or [`environment.yml`](environment.yml).

## Model Training and Testing

### Training procedure
For the DnCNN denoiser, the parameter initialization follows [He et al.](https://ieeexplore.ieee.org/document/7410480). We change the <img src="https://render.githubusercontent.com/render/math?math=\ell_2"> loss function of the original [paper](https://ieeexplore.ieee.org/abstract/document/7839189) to <img src="https://render.githubusercontent.com/render/math?math=\ell_1"> as it achieves better convergence performance. To train the classification networks, we fine-tune models pretrained on the ImageNet dataset. The fully connected layers are modified to fit the number of classes of each dataset (i.e. 257 for Caltech-256). We adopt the same initialization as [He et al.](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html), i.e., the [Xavier algorithm](http://proceedings.mlr.press/v9/glorot10a.html), and the biases are initialized to 0. We use the NAG descent optimizer with an initial learning rate of 0.001, and 120 training epochs. 
We also introduce a batch-step linear learning rate [warmup](https://arxiv.org/abs/1706.02677) for the first 5 epochs and a cosine learning rate [decay](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html), and apply [label smoothing](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html) with <img src="https://render.githubusercontent.com/render/math?math=\varepsilon=0.1">.
We select the model with the highest accuracy on the validation set.

### Train Pretrained Sub-models for the Proposed Models
The implemention of classification networks is taken from [torchvision](https://pytorch.org/docs/stable/torchvision/models.html), and the restoration networks are based on [DnCNN](https://github.com/cszn/KAIR), [MemNet](https://github.com/IVRL/DEU).

- To obtain pretrained classification networks:  
`python train.py --task classification --classification resnet50 --dataset caltech256  --num_class 257`  
    - The `--classification` argument takes value in `'resnet50', 'resnet18', 'alexnet', 'googlenet', 'vgg'`;
    - The `--dataset` and ` --num_class` takes value in `'caltech256', 257` and `'caltech101', 101` respectively.  

- To obtain pretrained restoration networks:  
`python train.py --task=restoration --degradation=awgn --restoration=dncnn --level 0 0.5 --batch_size 256`
    - The `--restoration` argument takes value in `'dncnn', 'memnet'`.  
    
- To obtain retrained classification networks on degraded images:  
`python train.py  --task classification --classification resnet50 --degradation awgn --level 0 0.1 0.2 0.3 0.4 0.5`  
    
- To obtain retrained classification networks on restored images:  
`python train.py  --task classification --classification resnet50 --degradation awgn --level 0 0.1 0.2 0.3 0.4 0.5 --restoration dncnn`

- To obtain our pretrained fidelity map estimator:  
`python train.py  --task fidelity --degradation awgn --restoration dncnn --level 0 0.5 --fidelity_input degraded --fidelity_output l1 --batch_size 256 --num_epochs 60`  
    - The `--fidelity_input` argument takes value in `'degraded', 'restored'`;
    - The `--fidelity_output` argument takes value in `'l1', 'l2', 'cos'`.  

### Proposed Model

- To train the proposed model:  
`python train.py  --task model --mode oracle --classification resnet50 --degradation awgn --restoration dncnn --level 0 0.1 0.2 0.3 0.4 0.5 --fidelity_input degraded --fidelity_output l1 --num_epochs 60  --dataset caltech256 --num_class 257`  
    - The `--mode` argument takes value in `'endtoend-pretrain', 'pretrain', 'oracle'`
    
- To test the proposed model:  
`python test.py --task model --mode oracle --classification resnet50 --degradation awgn --level 0.1 --restoration dncnn --fidelity_input degraded --fidelity_output l1 --is_ensemble True`
    - The `--is_ensemble` argument takes value in `'True', 'False'`

## Baseline Methods and Ablation Study
- We provide four baseline methods for a comprehensive analysis. To train and test the baseline methods:
    - [WaveCNet](https://github.com/LiQiufu/WaveCNet)
        - train: `python train.py  --task wavecnet --classification resnet50`;
        - test: `python test.py  --task wavecnet --classification resnet50 --degradation awgn --level 0.1`;
    - [DeepCorrect](https://github.com/tsborkar/DeepCorrect)
        - train: `python train.py  --task deepcorrect --classification resnet50 --degradation awgn --level 0 0.1 0.2 0.3 0.4 0.5 --num_epochs 60`;
        - test: `python test.py  --task deepcorrect --classification resnet50 --degradation awgn --level 0.1`.

- We also provide some in-depth analysis and ablation study models:
    - To try different fidelty map inputs and outputs, you can use the `--fidelity_input` and `--fidelity_output` arguments;
    - To try different downsampling methods, you can use the `--downsample` argument which takes value in `'bicubic', 'bilinear', 'nearest'`;
    - For ablation study, you can use the `--ablation` argument which takes value in `'spatialmultiplication' 'residualmechanism' 'spatialaddition' 'channelmultiplication' 'channelconcatenation'`;
    - **Note**: For more details on the ablation study models, please refer to our paper.  
    
## Results
Aside from the results in our main paper and supplementary material, we also illustrate the performance of the proposed method on other classification (e.g. AlexNet in the figure below on the left) and restoration networks (e.g. MemNet in the figure below on the right). The performance of the proposed method on other networks parallels that on ResNet-50 and DnCNN in our paper. This shows that the proposed method is model-agnostic and can be used on other networks.

<p align="center">
  <img src="materials/alexnet-dncnn.svg" width="350px"/>
  <img src="materials/resnet50-memnet.svg" width="350px"/>
</p>

**The above figure on the left:** Classification results with the AlexNet classification network and the DnCNN restoration network, on the Caltech-256 dataset, for various setups. The solid lines indicate testing directly on noisy images. The dashed lines indicate testing with the DnCNN restoration preprocessing step.

**The above figure on the right:** Classification results with the ResNet-50 classification network and the MemNet restoration network, on the Caltech-256 dataset, for various setups. The solid lines indicate testing directly on noisy images. The dashed lines indicate testing with the MemNet restoration preprocessing step.

### Extended Experimental Results (CUB-200-2011)

The CUB-200-2011 dataset is an image dataset of 200 bird species. There are 5994 training images and 5794 test images. We randomly chose 20 percent of the training set for validation.
The results are given in the table below.

<table>
    <tr>
        <td rowspan="2">Methods</td>
        <td rowspan="2">Experimental<br>results </td>
        <td colspan="5" align="center">Uniform degradation (sigma)</td>
    </tr>
    <tr>
        <td>0.1</td>
        <td>0.2</td>
        <td>0.3</td>
        <td>0.4</td>
        <td>0.5</td>
    </tr>
    <tr>
        <td rowspan="2">Pretrained</td>
        <td>Test on noisy</td>
        <td>34.89</td>
        <td>08.11</td>
        <td>02.02</td>
        <td>00.89</td>
        <td>00.70</td>
    </tr>
    <tr>
        <td>Test on restored</td>
        <td>56.77</td>
        <td>42.37</td>
        <td>30.97</td>
        <td>23.15</td>
        <td>16.91</td>
    </tr>
    <tr>
        <td rowspan="2">Retrain on<br>noisy</td>
        <td>Test on noisy</td>
        <td>59.91</td>
        <td>55.86</td>
        <td>51.41</td>
        <td>46.94</td>
        <td>42.09</td>
    </tr>
    <tr>
        <td>Test on restored</td>
        <td>58.37</td>
        <td>52.56</td>
        <td>44.97</td>
        <td>37.76</td>
        <td>31.33</td>
    </tr>
    <tr>
        <td rowspan="2">Retrain on<br>restored</td>
        <td>Test on noisy</td>
        <td>52.53</td>
        <td>24.46</td>
        <td>07.18</td>
        <td>01.86</td>
        <td>00.85</td>
    </tr>
    <tr>
        <td>Test on restored</td>
        <td>63.34</td>
        <td>59.51</td>
        <td>54.76</td>
        <td>49.83</td>
        <td>44.63</td>
    </tr>
    <tr>
        <td rowspan="2">FG-NIC<br>(Pretrained)</td>
        <td>Single</td>
        <td>63.75</td>
        <td>56.98</td>
        <td>48.87</td>
        <td>40.55</td>
        <td>32.82</td>
    </tr>
    <tr>
        <td>Ensemble</td>
        <td>64.95</td>
        <td>57.37</td>
        <td>48.74</td>
        <td>40.33</td>
        <td>32.38</td>
    </tr>
    <tr>
        <td rowspan="2">FG-NIC<br>(Oracle)</td>
        <td>Single</td>
        <td>65.10</td>
        <td>60.21</td>
        <td>55.26</td>
        <td>50.77</td>
        <td>46.10</td>
    </tr>
    <tr>
        <td>Ensemble</td>
        <td>65.75</td>
        <td>60.95</td>
        <td>55.75</td>
        <td>51.15</td>
        <td>46.32</td>
    </tr>
</table>


## Citation

```bibtex
@article{lin2021fidelity,
    title={Fidelity Estimation Improves Noisy-Image Classification with Pretrained Networks}, 
    author={Xiaoyu Lin and Deblina Bhattacharjee and Majed El Helou and Sabine Süsstrunk},
    journal={IEEE Signal Processing Letters},
    year={2021},
    publisher={IEEE}
}
```
