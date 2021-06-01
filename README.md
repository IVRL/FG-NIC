# Fidelity Estimation Improves Noisy Image Classification with Pretrained Networks
**Authors**: Xiaoyu Lin, Deblina Bhattacharjee, Majed El Helou and Sabine Süsstrunk

![Python 3.7.9](https://img.shields.io/badge/python-3.7-blue.svg) 
![pytorch 1.6.0](https://img.shields.io/badge/pytorch-1.6.0-orange.svg)
![CUDA 10.2](https://img.shields.io/badge/cuda-10.2-green.svg)


#### [[Paper]](https://github.com/IVRL/FG-NIC) - [[Supplementary]](https://github.com/IVRL/FG-NIC/materials/supp.pdf)


> **Abstract:** *Image classification has significantly improved using deep learning. This is mainly due to convolutional neural networks (CNNs) that are capable of learning rich feature extractors from large datasets. However, most deep learning classification methods are trained on clean images and are not robust when handling noisy ones, even if a restoration preprocessing step is applied. 
While novel methods address this problem, they rely on modified feature extractors and thus necessitate retraining. 
We instead propose a method that can be applied on a pretrained classifier. Our method exploits a fidelity map estimate that is fused into the internal representations of the feature extractor, thereby guiding the attention of the network and making it more robust to noisy data. 
We improve the noisy-image classification (NIC) results by significantly large margins, especially at high noise levels, and come close to the fully retrained approaches. Furthermore, as proof of concept, we show that when using our oracle fidelity map we even outperform the fully retrained methods, whether trained on noisy or restored images.*
>

## Table of Contents  
- [Degradation model](#degradation-model)
- [Requirements](#requirements)
- [Model Training and Testing](#model-training-and-testing)
- [Baseline Methods and Ablation Study](#baseline-methods-and-ablation-study)
- [Results](#results)
- [Citation](#citation)

## Degradation model
To explore the effects of degradation types and levels on classification networks, we also implement five types of degradation model: Additive white Gaussian noise (AWGN), Salt and Pepper Noise, Gaussian Blur, Motion Blur and Rectangle Crop. The instruction of those degradatin models is given in [notebook](synthetic_images.ipynb).  

## Requirements
- Python 3.7, PyTorch 2.1.0;
- Other common packages listed in [`requirements.txt`](requirements.txt) or [`environment.yml`](environment.yml).

## Model Training and Testing

### Training procedure
For the DnCNN denoiser, the parameter initialization follows [He et al.](https://ieeexplore.ieee.org/document/7410480). And we change the <img src="https://render.githubusercontent.com/render/math?math=\ell_2"> loss function of the original [paper](https://ieeexplore.ieee.org/abstract/document/7839189) to <img src="https://render.githubusercontent.com/render/math?math=\ell_1"> as it achieves better convergence performance. To train the classification networks, we fine-tune the pretrained model on the ImageNet dataset. The fully connected layers are modified to fit the number of classes of dataset (i.e. 257 for Caltech-256). We adopt the same initialization as [He et al.](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html), i.e. the [Xavier algorithm](http://proceedings.mlr.press/v9/glorot10a.html), and the biases are initialized to 0. We use the NAG descent optimizer with an initial learning rate of 0.001, and 120 training epochs. 
We also introduce a batch-step linear learning rate [warmup](https://arxiv.org/abs/1706.02677) for the first 5 epochs and a cosine learning rate [decay](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html), and apply [label smoothing](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html) with <img src="https://render.githubusercontent.com/render/math?math=\varepsilon=0.1">.
We select the model with the highest accuracy on the validation set.

### Train Pretrained sub-models for proposed models
The implemention of classification networks are from [torchvision](https://pytorch.org/docs/stable/torchvision/models.html), and restoration networks is based on [DnCNN](https://github.com/cszn/KAIR), [MemNet](https://github.com/IVRL/DEU).

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

- To obtain pretrained fidelity map estimator:  
`python train.py  --task fidelity --degradation awgn --restoration dncnn --level 0 0.5 --fidelity_input degraded --fidelity_output l1 --batch_size 256 --num_epochs 60`  
    - The `--fidelity_input` argument takes value in `'degraded', 'restored'`;
    - The `--fidelity_output` argument takes value in `'l1', 'l2', 'cos'`.  

### Proposed model

- To train the proposed model:  
`python train.py  --task model --mode oracle --classification resnet50 --degradation awgn --restoration dncnn --level 0 0.1 0.2 0.3 0.4 0.5 --fidelity_input degraded --fidelity_output l1 --num_epochs 60  --dataset caltech256 --num_class 257`  
    - The `--mode` argument takes value in `'endtoend-pretrain', 'pretrain', 'oracle'`
    
- To test the proposed model:  
`python test.py --task model --mode oracle --classification resnet50 --degradation awgn --level 0.1 --restoration dncnn --fidelity_input degraded --fidelity_output l1 --is_ensemble True`
    - The `--is_ensemble` argument takes value in `'True', 'False'`

## Baseline Methods and Ablation Study
- We provide four baseline methods for comprehensive analysis. To train and test baseline methods:
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
    - **Note**: For more details of ablation study models, please refer to our paper.  
    
## Results
Except results in our main paper and supplementary materials, we illustrate the performances of the proposed method on other classification network (e.g. AlexNet in figure below on the left) and restoration network (e.g. MemNet in figure below on the right). The performances of the proposed method on other networks are similar to that on ResNet-50 and DnCNN in our paper, which demonstrates the proposed method is model-agnostic which can be used by other networks.

<p align="center">
  <img src="materials/alexnet-dncnn.png" width="350px"/>
  <img src="materials/resnet50-memnet.png" width="350px"/>
</p>

Above figure on the left: Classification results with the AlexNet classification and DnCNN restoration network, on Caltech-256 dataset, for various setups. The solid lines indicate testing directly on noisy images. The dashed lines indicate testing with the DnCNN restoration preprocessing step. Best viewed on screen.  
Above figure on the right: Classification results with the ResNet-50 classification and MemNet restoration network, on Caltech-256 dataset, for various setups. The solid lines indicate testing directly on noisy images. The dashed lines indicate testing with the DnCNN rest
## Citation

```bibtex

```