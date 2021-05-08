#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : FG-NIC
# @Author       : Xiaoyu LIN
# @File         : test.py
# @Description  : This file is used to test proposed and baseline methods.

from typing import Optional
from torch.nn import Module
import torch
import torch.nn as nn
import time
import os
import copy
import numpy as np
import random
from tqdm import tqdm
import pickle
from collections import defaultdict

from degradation import get_degradation, Normalize
from util import prepare_parser, set_cwd, get_level, set_parameter_requires_grad, prepare_ablation
from classification import initialize_classification, split_classification
from restoration import initialize_restoration
from ptflops import get_model_complexity_info

from DeepCorrect import DeepCorrect
from baseline.WaveCNet.wavecnet import initialize_wavecnet

def test(model: Module,
         task: str,
         device: torch.device,
         dataloader: torch.utils.data.DataLoader,
         mode: str = None,
         num_round: int = 5,
         restoration: Optional[Callable] = None,
         MEAN: list = [0.485, 0.456, 0.406],
         STD: list = [0.229, 0.224, 0.225]
         ) -> np.ndarray:
    """ Test fine tuned classifier on test dataset with various degradation.
        Args:
            classifier (Module): Classifier model to test.
            device (torch.device): Device where classifier model located.
            datalodaer (torch.utils.data.DataLoader): Dataloader for test data.
            round_num (int): Total number of round for testing, since degradation is introduced on data,
                reture average results.
            restoration (object): Restoration network applied on the degraded image before sending
                it into classifier.
        Return:
            acc_numpy (ndarray): Array of history accuary among all rounds.
    """
    since = time.time()
    
    accuracy = []
    
    # Test for each round
    for rounds in range(num_round):
    
        running_accuracy = 0
        
        # Iterate over data.
        for inputs, origins, labels in tqdm(dataloader, ncols=70, leave=False, unit='b',
                                            desc='Testing round {}/{}'.format(rounds+1, num_round)):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # restoration
            inputs = restoration(inputs).clamp_(0.0, 1.0) if restoration is not None else inputs
            inputs = Normalize(MEAN, STD)(inputs) if task.lower() in ('classification', 'deepcorrect', 'wavecnet')  else inputs
            
            # forward
            if mode and 'oracle' in mode.lower():
                outputs = model(inputs, origins)
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_accuracy += torch.sum(preds == labels.data)

        current_accuracy = running_accuracy.double() / len(dataloader.dataset)
        accuracy.append(current_accuracy)
        print('Round {} Accuracy: {:.6f}'.format(rounds + 1, current_accuracy))
    
    accuracy = np.array([i.cpu().detach().numpy() for i in accuracy])

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    with open('last_test_batch.pickle', 'wb') as file:
        pickle.dump(inputs, file)
    
    return accuracy

    
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
    set_cwd(args, 'test')
    
    # Check model
    if not os.path.isfile('model.pth'):
        print("No trained directory exit. Please fine tune model first!")
        exit()

    # Record logs
    logs = open('test-logs.txt', mode='a')

    # Initialize and load the classification network
    print('Preparing classification network {} ..... '.format(args.classification), end='')
    classification, input_size = initialize_classification(args.classification, args.num_classes, use_pretrained=False)
    if args.task.lower() in ('model', 'deepcorrect'):
        classification.load_state_dict(torch.load(os.path.join(project_dir, args.CLASSIFIER_DIR, 
                                                               '-'.join([args.dataset.lower(), args.classification, 'clean']), 
                                                               'model.pth')))
        feature_extractor, classifier, feature_size = split_classification(args.classification, classification)
        feature_extractor.eval()
        set_parameter_requires_grad(feature_extractor, False)
        classifier.eval()
        set_parameter_requires_grad(classifier, False)
        classification = None
    elif args.task.lower() in 'wavecnet':
        model = initialize_wavecnet(classification=args.classification, 
                                    num_classes=args.num_classes, 
                                    wavename='haar', 
                                    pretrained=False)
        model.load_state_dict(torch.load('model.pth'))
        set_parameter_requires_grad(model, False)
        model.eval()        
    else:
        classification.load_state_dict(torch.load('model.pth'))
        set_parameter_requires_grad(classification, False)
        classification.eval()
        model = classification
    print('Done!')

    # Prepare test datasets and pre-processing.
    print('Preparing Datasets and Dataloaders.....', end='')
    data_transforms, level = get_degradation(degradation_type=args.degradation,
                                             level=level,
                                             vary=args.vary,
                                             phase = 'test',
                                             )
    # datasets
    if 'caltech256' in args.dataset.lower():
        from data import Caltech256 as Dataset
    elif 'caltech101' in args.dataset.lower():
        from data import Caltech101 as Dataset
    test_image_dataset = Dataset(root=os.path.join(project_dir, args.DATA_DIR, args.dataset.lower()), phase='test', transform=data_transforms)
    # Create training and validation dataloaders
    test_dataloader = torch.utils.data.DataLoader(test_image_dataset, 
                                                  batch_size=args.batch_size, 
                                                  shuffle=False,
                                                  num_workers=8
                                                 )
    print('Done!')
    print('Degradation type: {}; Degradation level {} varying {}'.format(args.degradation, str(level), args.vary))
    logs.write('Degradation type: {}; Degradation level {} varying {}. '.format(args.degradation, str(level), args.vary) + '\n')
    
    # Initialize and load the restoration network
    restoration = None
    if args.restoration is not None:
        print('Preparing restoration network {} ...... '.format(args.restoration) , end='')
        logs.write('Restoration model: {}.\n'.format(args.restoration))
        restoration = initialize_restoration(name=args.restoration,
                                             dataset=args.dataset.lower(),
                                             path=os.path.join(project_dir, args.RESTORATION_DIR),
                                             use_pretrain=True
                                            )
        restoration.eval()
        set_parameter_requires_grad(restoration, False)
        print('Done!')
        
    # Load fidelity map
    if 'model' in args.task.lower():
        if not args.ablation:
            from model import Model
        else:
            Model = prepare_ablation(args.ablation)
        print('Preparing fidelity map estimator ...... ', end='')
        fidelity = None
        if args.fidelity_input is not None and 'oracle' not in args.mode.lower():
            fidelity = initialize_restoration(name='dncnn', dataset=args.dataset.lower(), path=args.RESTORATION_DIR, use_pretrain=False)
            fidelity.load_state_dict(torch.load(os.path.join(project_dir, args.FIDELITY_DIR, \
                                '-'.join([args.dataset.lower(), args.fidelity_input, args.fidelity_output, args.restoration]), 'model.pth')))
            fidelity.eval()
            set_parameter_requires_grad(fidelity, False)
        
        # Load our model
        model = Model(mode=args.mode,
                      restoration=copy.deepcopy(restoration),
                      fidelity_input=args.fidelity_input,
                      fidelity_output=args.fidelity_output,
                      feature_extractor=feature_extractor,
                      feature_size=feature_size,
                      classifier=classifier,
                      downsample=args.downsample,
                      fidelity=copy.deepcopy(fidelity),
                      increase=args.increase,
                      num_channel=args.num_channel,
                      MEAN=args.MEAN,
                      STD=args.STD
                     )
        restoration = None
        fidelity = None
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        set_parameter_requires_grad(model, False)
        model.is_ensemble=args.is_ensemble
        print('Done!')
        print('Using ensemble') if args.is_ensemble else print('Using single model')
        logs.write('Ensemble\n') if args.is_ensemble else logs.write('Single model\n')
    
    if 'deepcorrect' in args.task.lower():
        model = DeepCorrect(feature_extractor=feature_extractor,
                            classifier=classifier,
                            MEAN=args.MEAN,
                            STD=args.STD
                           )
        model.load_state_dict(torch.load('model.pth'))
        
    # Send the networks to GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    if restoration is not None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            restoration = nn.DataParallel(restoration)
        restoration = restoration.to(device)
    
    # Train and evaluate
    num_round = 1 if isinstance(level,(int, float)) and level == 0 else args.num_round
    accuracy = test(model=model,
                    task=args.task,
                    mode=args.mode,
                    device=device,
                    dataloader=test_dataloader,
                    num_round=num_round,
                    restoration=restoration,
                    MEAN=args.MEAN,
                    STD=args.STD
                    )

    print("Average Accuracy: {:.6f}; Std: {:.6f}".format(accuracy.mean().item(), accuracy.std().item()))
    logs.write("Average Accuracy: {:.6f}; Std: {:.6f}\n".format(accuracy.mean().item(), accuracy.std().item()))
    
    # Save data
    try:
        results = pickle.load(open('accuracy.pickle', 'rb'))
    except FileNotFoundError:
        results = defaultdict(dict)
    with open('accuracy.pickle', 'wb') as file:
        if args.vary.lower() in ('1d', '2d'):
            level = (level, args.vary)
        setup = args.degradation + '-' + args.restoration if args.restoration else args.degradation
        if args.task == 'model':
            setup = setup + '-ensemble' if args.is_ensemble else setup + '-single'
        results[setup][level] = accuracy
        pickle.dump(results, file)
        
    logs.write('Done.\n')
    logs.close()
    print('Done.')
