#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : FG-NIC
# @Author       : Xiaoyu LIN
# @File         : train.py
# @Description  : This file is used to train all models inclusing proposed FG-NIC and baseline methods, 
#                 classification and restoration networks, fidelity estimator.

from typing import TextIO
from torch.nn import Module

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy
import numpy as np
import random
from tqdm import tqdm
import pickle
import math

from classification import *
from util import prepare_parser, set_cwd, get_level, set_parameter_requires_grad, prepare_ablation
from degradation import Normalize, get_degradation
from restoration import initialize_restoration, getPSNR
from thop import profile
from DeepCorrect import DeepCorrect
from baseline.WaveCNet.wavecnet import initialize_wavecnet

def train(model: Module,
          task: str,
          mode: str,
          device: torch.device,
          dataloaders: dict,
          lr: float,
          num_epochs: int,
          warmup: int = 5,
          logs: TextIO = None,
          model_name: str = None,
          smoothing: float = 0.1,
          restoration = None,
          MEAN: list = [0.485, 0.456, 0.406],
          STD: list = [0.229, 0.224, 0.225],
          fidelity_input: str = 'degraded',
          fidelity_output: str = 'l1'
         ) -> None:
    """ Main training function for classifiers
        Args:
            model (Module): Classifier model to train
            device (torch.device): Device where classifier model located.
            datalodaers (dict): Dataloader directory with two keys 'train' and 'valid'.
            lr (float): Initial learning rate.
            num_epochs (int): Total number of epochs for training.
            logs (Text file): Text file to record all training information: loss, accuracy v.s. epochs.
            model_name (str): Name of classifier in ['vgg', 'alexnet', 'resnet', 'googlenet']
            warmup (int): The number of epochs for learing rate warmup in first several batches.
    """
    # Gather the parameters to be optimized/updated in this run.
    logs.write("Params to learn:\n")
    print("Params to learn:")
    param_num = 0
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            param_num += param.nelement()
            logs.write("%s\n" % name)
#             print(name)
    logs.write("Number of Params to learn: {:E}\n".format(param_num))
    print("Number of Params to learn: {:E}".format(param_num))
    
    # Preparetion
    since = time.time()
    valid_metrics, train_metrics, valid_loss, train_loss= [], [], [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metrics = 0.0 if task.lower() != 'fidelity' else -1.0 * float('inf')
    if task.lower() in ('classification', 'model', 'deepcorrect', 'wavecnet'):
        metrics = 'Accuracy'  
    elif 'restoration' in task.lower():
        metrics = 'PSNR improvement'
    else:
        metrics = 'Loss'
    
    # Observe that all parameters are being optimized
    if task.lower() in ('classification', 'model', 'deepcorrect', 'wavecnet'):
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9, nesterov=True)
    else:
        optimizer = optim.Adam(params_to_update, lr=lr)

    # Setup the loss function 
    if smoothing > 0 and task.lower() in ('classification', 'model', 'deepcorrect', 'wavecnet'):
        label_smoothing = LabelSmoothing(smoothing)
        criterion = nn.KLDivLoss(reduction='batchmean')
    elif task.lower() in ('classification', 'model', 'deepcorrect', 'wavecnet'):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss()
        
    # Train for each epoch
    for epoch in range(num_epochs):
        
        # each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            # set model to train mode
            if phase == 'train':
                # fine-tune classification network
                if 'model' not in task.lower():
                    model.train()
                # train our proposed method
                else:
                    if torch.cuda.device_count() > 1:
                        if model.module.is_ensemble:
                            model.module.ensemble.train()
                        else:
                            model.module.cnn_layers_weight.train()
                            model.module.cnn_layers_bias.train()
                            model.module.cat_fc.train()
                            model.module.mul_fc.train()
                            if 'endtoend' in args.mode.lower():
                                model.module.fidelity.train()
                    else:
                        if model.is_ensemble:
                                model.ensemble.train()
                        else:
                            model.cnn_layers_weight.train()
                            model.cnn_layers_bias.train()
                            model.cat_fc.train()
                            model.mul_fc.train()
                            if 'endtoend' in args.mode.lower():
                                model.fidelity.train()
            # set model to evaluate mode
            else:
                model.eval()  

            running_loss = 0.0
            running_metrics = 0
            
            # cosine learning rate decay
            if epoch >= warmup and phase == 'train':
                cosine_learning_rate_decay(optimizer, epoch - warmup, lr, num_epochs - warmup)

            # iterate over data
            batch = 0
            for inputs, origins, labels in tqdm(dataloaders[phase], ncols=70, leave=False, 
                                                unit='b', desc='Epoch {}/{} {}'.format(epoch + 1, num_epochs, phase)):
                # learning rate warmup
                batch += 1
                if epoch < warmup and phase == 'train':
                    learning_rate_warmup(optimizer, warmup, epoch, lr, batch, len(dataloaders[phase]))
                    
                # For restoration network and foidelity map estimator, change size of inputs (N, P, C, H, W) to (NxP, C, H, W)  
                if task.lower() in ('restoration', 'fidelity'):
                    batch_size, patch_size, num_channels, height, width = inputs.shape
                    inputs = inputs.view(-1, num_channels, height, width)
                    origins = origins.view(-1, num_channels, height, width)
                
                inputs = inputs.to(device)
                if task.lower() == 'fidelity':
                    degraded = copy.deepcopy(inputs)
            
                # train network on restored images
                inputs = restoration(inputs).clamp_(0.0, 1.0) if restoration is not None else inputs
                if task.lower() == 'fidelity':
                    restored = copy.deepcopy(inputs)
                    if fidelity_input.lower == 'degraded':
                        inputs = degraded
                # normalize image to train classification network,
                #   for proposed model, this step is integraed in model object.
                inputs = Normalize(MEAN, STD)(inputs) if task.lower() in ('classification', 'deepcorrect', 'wavecnet')  else inputs
                
                origins = origins.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                model.zero_grad()
                optimizer.zero_grad()

                # forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # get model outputs and calculate loss
                    if model_name == 'googlenet' and phase == 'train' and 'classification' == task.lower():
                        outputs, aux1, aux2 = model(inputs)
                        # set up label smoothing
                        if smoothing > 0:
                            labels_before_smoothing = copy.deepcopy(labels)
                            labels = label_smoothing(outputs, labels)
                            aux1 = F.log_softmax(aux1, dim=-1)
                            aux2 = F.log_softmax(aux2, dim=-1)
                            outputs = F.log_softmax(outputs, dim=-1)
                        loss1 = criterion(aux1, labels)
                        loss2 = criterion(aux2, labels)
                        loss3 = criterion(outputs, labels)
                        loss = loss3 + 0.3 * (loss1 + loss2)
                    elif task.lower() in ('classification', 'model', 'deepcorrect', 'wavecnet'):
                        outputs = model(inputs, origins) if task.lower()=='model' and 'oracle' in mode.lower() else model(inputs)
                        # set up label smoothing
                        if smoothing > 0:
                            labels_before_smoothing = copy.deepcopy(labels)
                            labels = label_smoothing(outputs, labels)
                            outputs = F.log_softmax(outputs, dim=-1)
                        loss = criterion(outputs, labels)
                    elif task.lower() in ('fidelity'): 
                        outputs = inputs - model(inputs)
                        if 'l1' in fidelity_output.lower():
                            targets = (restored-origins).abs()
                        elif 'l2' in fidelity_output.lower():
                            targets = (restored-origins).square()
                        loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, origins)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if task.lower() in ('classification', 'model', 'deepcorrect', 'wavecnet'):
                    labels = labels_before_smoothing if smoothing > 0 else labels
                    _, preds = torch.max(outputs, 1)
                    running_metrics += torch.sum(preds == labels.data)
                elif 'restoration' in task.lower():
                    running_metrics += getPSNR(outputs.clamp(0,1), origins).item() * inputs.size(0)\
                                        - getPSNR(inputs, origins).item() * inputs.size(0)
                
            if task.lower() in ('classification', 'model', 'deepcorrect', 'wavecnet'):
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_metrics = running_metrics.double() / len(dataloaders[phase].dataset)
            else:
                epoch_loss = running_loss / (len(dataloaders[phase].dataset) * patch_size)
                epoch_metrics = running_metrics / (len(dataloaders[phase].dataset) * patch_size) \
                                if 'restoration' in task.lower() else -1.0 * epoch_loss
                
            print('Epoch {}: {} Loss: {:.6f} {}: {:.6f}'.format(epoch + 1, phase, epoch_loss, metrics, epoch_metrics))
            logs.write('Epoch {}: {} Loss: {:.6f} {}: {:.6f}\n'.format(epoch + 1, phase, epoch_loss, metrics, epoch_metrics))

            
            # record loss and metrics
            if phase == 'valid':
                valid_metrics.append(epoch_metrics)
                valid_loss.append(epoch_loss)
                # deep copy the model
                if epoch_metrics > best_metrics:
                    best_metrics = epoch_metrics
                    best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_metrics.append(epoch_metrics)
                train_loss.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logs.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val {}: {:6f}'.format(metrics, best_metrics))
    logs.write('Best val {}: {:6f}\n'.format(metrics, best_metrics))
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    # save model and training data
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), 'model.pth')
    else:
        torch.save(model.state_dict(), 'model.pth')
    for data, name in [(valid_metrics, 'valid_metrics'), (valid_loss, 'valid_loss'),
                       (train_metrics, 'train_metrics'), (train_loss, 'train_loss')]:
        with open('{}.pickle'.format(name), 'wb') as file:
            pickle.dump(data, file)
    
    with open('last_batch.pickle', 'wb') as file:
        pickle.dump(inputs, file)
        
    
            
    return model


if __name__ == '__main__':

    args = prepare_parser()
    level = get_level(args.level, args.level_min)
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.dev
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For reproduction
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set current working directory
    project_dir = os.getcwd()
    set_cwd(args)
        
    # Check model
    if os.path.isfile('model.pth'):
        print("There exist a trained model in this directory.")
        exit()

    # Record logs
    logs = open('train-logs.txt', mode='a')
    logs.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    if torch.cuda.is_available():
        print("Using", torch.cuda.device_count(), "GPUs.")
        logs.write("Using " + str(torch.cuda.device_count()) + " GPUs.")
        
    # Initialize and load the classification network
    input_size = args.input_size
    if args.task.lower() in ('classification', 'model', 'deepcorrect'):
        print('Preparing classification network......', end='')
        classification, input_size = initialize_classification(args.classification, args.num_classes,
                                                               use_pretrained='classification' in args.task.lower())
        # train our network task
        if args.task.lower() in ('model', 'deepcorrect'):
           
            classification.load_state_dict(torch.load(os.path.join(project_dir, args.CLASSIFIER_DIR, 
                                                                   '-'.join([args.dataset.lower(), args.classification, 'clean']), 
                                                                    'model.pth')), strict=True)
            feature_extractor, classifier, feature_size = split_classification(args.classification, classification)            
            feature_extractor.eval()
            set_parameter_requires_grad(feature_extractor, False)
            classifier.eval()
            set_parameter_requires_grad(classifier, False)
        
        # train classification network task
        else:
            set_parameter_requires_grad(classification, True)
            model = classification
        print('Done!')
        
    elif 'wavecnet' in args.task.lower():
        model = initialize_wavecnet(classification=args.classification, 
                                    num_classes=args.num_classes, 
                                    wavename='haar', 
                                    pretrained=True)
         
    # Prepare train and validation datasets and pre-processing.
    print('Preparing Datasets and Dataloaders......', end='')
    # data augmentation, implement degradation model
    data_transforms = {x: get_degradation(degradation_type=args.degradation,
                                          level=level,
                                          vary=args.vary,
                                          phase=x,
                                          input_size=input_size,
                                          task=args.task,
                                          patch_size=args.patch_size,
                                          stride=args.stride,
                                          is_discrete=args.task.lower() not in ('restoration', 'fidelity') 
                                         )[0] 
                       for x in ['train', 'valid']}
    # datasets
    if 'caltech256' in args.dataset.lower():
        from data import Caltech256 as Dataset
    elif 'caltech101' in args.dataset.lower():
        from data import Caltech101 as Dataset
    image_datasets = {x: Dataset(root=os.path.join(project_dir, args.DATA_DIR, args.dataset.lower()), 
                                 phase=x, transform=data_transforms[x]) 
                      for x in ['train', 'valid']}
    
    # dataloaders
    if args.task.lower() in ('restoration', 'fidelity'):
        args.batch_size = int(args.batch_size / ((math.floor(input_size/args.stride))**2))
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=8)
                        for x in ['train', 'valid']}
    print('Done!')
    
    # Initialize and load restoration network
    restoration = None
    if args.restoration is not None:
        print('Preparing restoration network......', end='')
        restoration = initialize_restoration(name=args.restoration,
                                             dataset=args.dataset.lower(),
                                             path=os.path.join(project_dir, args.RESTORATION_DIR),
                                             use_pretrain='restoration' not in args.task.lower()
                                            )
        set_parameter_requires_grad(restoration, 'restoration' in args.task.lower())
        # Restoration network task
        if 'restoration' in args.task.lower():
            model = restoration
            restoration = None
        print('Done!')
        
    # Initialize and load fidelity map estimator network
    fidelity = None
    if args.task.lower() in ('fidelity', 'model'):
        print('Preparing fidelity map estimator......', end='')
        fidelity = initialize_restoration(name='dncnn',
                                          dataset=args.dataset.lower(),
                                          path=os.path.join(project_dir, args.RESTORATION_DIR),
                                          use_pretrain = False                                          )
        if 'model' in args.task.lower():
            if 'pretrain' in args.mode.lower():
                fidelity.load_state_dict(torch.load(os.path.join(project_dir,args.FIDELITY_DIR, \
                    '-'.join([args.dataset.lower(), args.fidelity_input, args.fidelity_output, args.restoration]), 'model.pth')), strict=True)
            fidelity.eval()
            set_parameter_requires_grad(fidelity, False)
            if 'endtoend' in args.mode.lower():
                fidelity.train()
                set_parameter_requires_grad(fidelity, True)
        else:
            model = fidelity
            fidelity = None
        print('Done!')
    
    # Initialize our model
    if 'model' in args.task.lower():
        if not args.ablation:
            from model import Model
        else:
            Model = prepare_ablation(args.ablation)
        model = Model(mode=args.mode,
                      restoration=copy.deepcopy(restoration),
                      fidelity_input=args.fidelity_input,
                      fidelity_output=args.fidelity_output,
                      feature_extractor=feature_extractor,
                      feature_size=feature_size,
                      classifier=classifier,
                      downsample=args.downsample,
                      fidelity=None if 'oracle' in args.mode.lower() else fidelity,
                      increase=args.increase,
                      num_channel=args.num_channel,
                      MEAN=args.MEAN,
                      STD=args.STD
                     )
        if not args.ablation:
            set_parameter_requires_grad(model.ensemble, False)
        restoration = None
        
    if 'deepcorrect' in args.task.lower():
        model = DeepCorrect(feature_extractor=feature_extractor,
                            classifier=classifier,
                            MEAN=args.MEAN,
                            STD=args.STD
                           )
        
    # Train and evaluate
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.task.lower() not in 'wavecnet':
        model = nn.DataParallel(model)
    model = model.to(device)
    if restoration is not None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            restoration = nn.DataParallel(restoration)
        restoration = restoration.to(device)

    train(model=model,
          task=args.task,
          mode=args.mode,
          device=device,
          dataloaders=dataloaders_dict,
          lr=args.lr,
          num_epochs=args.num_epochs,
          warmup=args.warmup,
          logs=logs,
          model_name=args.classification,
          smoothing=args.smoothing,
          restoration=restoration,
          MEAN=args.MEAN,
          STD=args.STD,
          fidelity_input=args.fidelity_input,
          fidelity_output=args.fidelity_output
         )
#     inputs = torch.randn(1, 3, 224, 224)
#     if 'model' in args.task.lower():
#         macs, params = profile(model, inputs=(inputs.to(device), inputs.to(device)))
#     else:
#         macs, params = profile(model, inputs=(inputs.to(device), ))
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
    # Train ensemble network
    if 'model' in args.task.lower():
        
        print("Train ensemble network.")
        logs.write("Train ensemble network.")
        if torch.cuda.device_count() > 1:
            set_parameter_requires_grad(model.module, False)        
            set_parameter_requires_grad(model.module.ensemble, True)
            model.module.is_ensemble=True
        else:
            set_parameter_requires_grad(model, False)        
            set_parameter_requires_grad(model.ensemble, True)
            model.is_ensemble=True
        
        train(model=model,
              task=args.task,
              mode=args.mode,
              device=device,
              dataloaders=dataloaders_dict,
              lr=args.lr,
              num_epochs=args.num_epochs//2,
              warmup=args.warmup,
              logs=logs,
              model_name=args.classification,
              smoothing=args.smoothing,
              restoration=restoration,
              MEAN=args.MEAN,
              STD=args.STD,
              fidelity_input=args.fidelity_input,
              fidelity_output=args.fidelity_output
             )
#         inputs = torch.randn(1, 3, 224, 224)
#         macs, params = profile(model, inputs=(inputs.to(device), inputs.to(device)))
#         print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#         print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            
    logs.write('Done.')
    logs.close()
    print('Done.')
