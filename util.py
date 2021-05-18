#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : FG-NIC
# @Author       : Xiaoyu LIN
# @File         : util.py
# @Description  : This file contains helper funcitons for both training and testing phases.

from typing import Union, Optional
from torch.nn import Module
import argparse
import os

def str2bool(v: str) -> bool:
    """ Convert string to boolean.
        Args:
            v (string): string
        Return:
            (boolean): True or False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
def prepare_parser() -> argparse.ArgumentParser:
    """ Parser for all scripts.
        Returns:
            args (argparse.ArgumentParser): Arguments read from commond line.
    """
    
    parser = argparse.ArgumentParser()
    
    # classification network
    parser.add_argument('--classification', default=None, type=str, help='Name of classification network') 
    parser.add_argument('--num_classes', default=256 + 1, type=int, help='Number of classes in the dataset')
    parser.add_argument('--dataset', default='caltech256', type=str, help='Name of dataset')
    
    # training
    parser.add_argument('--task', default='classification', type=str, help='Name of task') # ['classification']
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')  # 128 for classification, 256 for restoration
    parser.add_argument('--input_size', default=224, type=int, help='Size of input images')
    parser.add_argument('--num_epochs', default=120, type=int, help='Number of training epoches')  # 120 for classification, 60 for restoration and proposed model
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate') 
    parser.add_argument('--warmup', default=5, type=int, help='Number of epochs for batch-step warmup') 
    parser.add_argument('--smoothing', default=0.1, type=float, help='Epsilon for label smoothing') 
    
    # testing
    parser.add_argument('--num_round', default=10, type=int, help='Number of round to rerun the experiments')  
    parser.add_argument('--is_ensemble', default=True, type=str2bool, help='Flag for ensemble or single model')
    
    # degradaion
    parser.add_argument('--degradation', default='clean', type=str, help='Degradation type')
    parser.add_argument('--level', default=0.0, nargs="+", help='Max degradaion level or fixed degradation level')
    parser.add_argument('--level_min', default=None, type=float, help='Min degradation level for spatially_varying')
    parser.add_argument('--vary', default='uniform', type=str, help='Degradation level change style') 
    
    # restoration
    parser.add_argument('--restoration', default=None, type=str, help='Name of restoration network')
    parser.add_argument('--patch_size', default=50, type=int, help='Size of input patch')
    parser.add_argument('--stride', default=25, type=int, help='Stride to take image patch')
    
    # fidelity map
    parser.add_argument('--fidelity_input', default=None, type=str, help='Fidelity map input')
    parser.add_argument('--fidelity_output', default='l1', type=str, help='Fidelity map output')
    
    # our model
    parser.add_argument('--mode', default=None, type=str, help='Modes of proposed method')
    parser.add_argument('--downsample', default='bilinear', type=str, help='Downsample method')
    parser.add_argument('--increase', default=0.5, type=float)
    parser.add_argument('--num_channel', default=16, type=int)
    parser.add_argument('--ablation', default=None, type=str)
    
    # system
    parser.add_argument('--dev', default='3,2,1,0', type=str)  # Number of GPU to use
    parser.add_argument('--save_dir', default=None, type=str)  # Name of model
    
    # global variable
    parser.add_argument('--DATA_DIR', default="./datasets", type=str)  
    parser.add_argument('--CLASSIFIER_DIR', default="./classification", type=str)  
    parser.add_argument('--RESTORATION_DIR', default="./restorations", type=str)  
    parser.add_argument('--FIDELITY_DIR', default="./fidelity", type=str)  
    parser.add_argument('--RESULT_DIR', default="./result", type=str) 
    parser.add_argument('--MODEL_DIR', default="./saved_model", type=str) 
    parser.add_argument('--MEAN', default=[0.485, 0.456, 0.406], type=list) 
    parser.add_argument('--STD', default=[0.229, 0.224, 0.225], type=list)
    parser.add_argument('--SEED', default=0, type=int)
    
    return parser.parse_args()


def get_level(level_1: Union[float, list],
              level_2: Optional[Union[float, int, str]] = None,
             ) -> Union[float, tuple, list]:
    """ Get degradation level from command line.
    Args:
        level_1 (Union[float, list]): Command line argument '--level'
        level_2 (Optional[Union[float, int]]): Command line argument '--level_min'
    Returns:
        level (Union[float, tuple, list]): float for fixed uniform degradation level,
            tuple for saptially varying degradtion level, list for mixture of degradation level.
    """
    if not isinstance(level_1, list):
        return level_1
    else:
        level_1 = [float(i) for i in level_1]
        
        level_1 = [i/255.0 if i>1 else i for i in level_1]
        level_2 = level_2/255.0 if level_2 and level_2 > 1 else level_2
        
        if len(level_1) == 1:
            return level_1[0] if level_2 is None else (level_1[0], level_2)
        else:
            return level_1


def set_cwd(args: argparse.ArgumentParser,
            phase: str = 'train') -> None:
    """ Set up current working directory.
    Args:
        args (argparse.ArgumentParser): Arguments read from command-line.
        phase (str): Test or train phase.
    """
    if 'classification' in args.task.lower():
        PATH = args.classification + '-' + args.degradation if phase == 'train' else args.classification
        DIR = args.CLASSIFIER_DIR
        PATH = PATH + '-' + args.restoration if args.restoration is not None and phase == 'train' else PATH 
    elif 'restoration' in args.task.lower():
        PATH = args.restoration.lower()
        DIR = args.RESTORATION_DIR
    elif 'fidelity' in args.task.lower():
        PATH = '-'.join([args.fidelity_input, args.fidelity_output, args.restoration])
        DIR = args.FIDELITY_DIR
    elif 'model' in args.task.lower():
        PATH = '-'.join([args.classification, args.restoration, args.mode, args.fidelity_output])
        if args.fidelity_input is not None:
            PATH += '-' + args.fidelity_input
        DIR = args.MODEL_DIR
    elif 'deepcorrect' in args.task.lower():
        PATH = args.classification
        DIR = './baseline/DeepCorrect'
    elif 'wavecnet' in args.task.lower():
        PATH = args.classification
        DIR = './baseline/WaveCNet'
    
    PATH = '-'.join([args.dataset.lower(), PATH])
    if args.save_dir is not None:
        PATH += '-' + args.save_dir
    CWD= os.path.expanduser(os.path.join(DIR, PATH))    
    if not os.path.isdir(CWD):
        os.makedirs(CWD)
    os.chdir(CWD)
    

def set_parameter_requires_grad(model: Module, 
                                requires_grad: bool = False
                               ) -> None:
    """ Setup feature extract or fine tuning.
        Args:
            model (Module): model to be trained
            requires_grad (bool):
    """
    for param in model.parameters():
        param.requires_grad = requires_grad
        
        
def prepare_ablation(ablation: str) -> None:
    
    if 'spatialmultiplication' in ablation.lower():
        from ablation import SpatialMultiplication as Model
    elif 'residualmechanism' in ablation.lower():
        from ablation import ResidualMechanism as Model
    elif 'spatialaddition' in ablation.lower():
        from ablation import SpatialAddition as Model
    elif 'channelmultiplication' in ablation.lower():
        from ablation import ChannelMultiplication as Model
    elif 'channelconcatenation' in ablation.lower():
        from ablation import ChannelConcatenation as Model
    else:
        raise ValueError('Invalid ablation method.')
    
    return Model