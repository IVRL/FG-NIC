#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : FG-NIC
# @Author       : Xiaoyu LIN
# @File         : data.py
# @Description  : This file is used to genterate Pytorch dataset for caltech-256 and caltech-101.

from PIL import Image
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, verify_str_arg
import copy
import gdown
import pickle
import random
import os
import tarfile

class Caltech256(VisionDataset):
    """ Caltech 256 Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        phase (string): ['train', 'valid', 'test'] load data for different phase.
        is_return_origin (bool): If true, return target is label for classification, 
            if false, return target both label and the original image for restoration.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        train_size (int): The number of images in train and validation set per class.
        valid_ratio (float): The ratio of validation image in train and validation set per class.
    """

    def __init__(self,
                 root: str,
                 phase: str = 'train',
                 is_return_origin: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 train_size: int = 60,
                 valid_ratio: float = 0.2,
                ) -> None:
        super(Caltech256, self).__init__(root,
                                         transform=transform,
                                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        
        self.is_return_origin = is_return_origin
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        
        # check previous train and validation indices
        if os.path.isfile(os.path.join(self.root, 'train_dic.pickle')) and os.path.isfile(
                os.path.join(self.root, 'valid_dic.pickle')):
            with open(os.path.join(self.root, 'train_dic.pickle'), 'rb') as file:
                train_dic = pickle.load(file)
            with open(os.path.join(self.root, 'valid_dic.pickle'), 'rb') as file:
                valid_dic = pickle.load(file)
        # if no previous train and validation indices, sample train and validation data
        else:
            train_dic = {}
            valid_dic = {}
            for c in self.categories:
                fileslist = os.listdir(os.path.join(self.root, "256_ObjectCategories", c))
                n = len(list(filter(lambda file: file.endswith(".jpg"), fileslist)))
                # select 60 images randomly as training images per class
                train_index = random.sample(range(1, n + 1), k=train_size)
                valid_index = random.sample(train_index, k=int(train_size * valid_ratio))
                train_index = list(set(train_index).difference(set(valid_index)))
                train_dic[c] = train_index
                valid_dic[c] = valid_index
            with open(os.path.join(self.root, 'train_dic.pickle'), 'wb') as file:
                pickle.dump(train_dic, file)
            with open(os.path.join(self.root, 'valid_dic.pickle'), 'wb') as file:
                pickle.dump(valid_dic, file)
        
        # generate new index, label(y), and map (between label number and text label)
        self.index: List[int] = []
        self.y = []
        self.map = {}
        for (i, c) in enumerate(self.categories):
            if 'train' in phase.lower():
                self.index.extend(train_dic[c])
                self.y.extend(len(train_dic[c]) * [i])
            if 'valid' in phase.lower():
                self.index.extend(valid_dic[c])
                self.y.extend(len(valid_dic[c]) * [i])
            if 'test' in phase.lower():
                fileslist = os.listdir(os.path.join(self.root, "256_ObjectCategories", c))
                n = len(list(filter(lambda file: file.endswith(".jpg"), fileslist)))
                self.index.extend(
                    list(set(range(1, n + 1)).difference(set(train_dic[c])).difference(set(valid_dic[c]))))
                self.y.extend((n - train_size) * [i])
            self.map[i] = c.split('.')[-1]
        

    def __getitem__(self, 
                    index: int
                   ) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class for classification task 
                or the same image for restoration task.
        """
        img = Image.open(os.path.join(self.root,
                                      "256_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "{:03d}_{:04d}.jpg".format(self.y[index] + 1, self.index[index])))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        origin = copy.deepcopy(img)
        target = self.y[index]

        if self.is_return_origin and self.transform is not None:
            img, origin = self.transform(img)
        elif self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_return_origin:
            return img, origin, target,
        else:
            return img, target,

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_root = self.root
        extract_root = download_root
        filename = "256_ObjectCategories.tar"
        url = "https://drive.google.com/uc?id=1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK"
        archive = os.path.join(download_root, filename)
        gdown.download(url, archive, quiet=False)
        
        # extract file
        print("Extracting {} to {}".format(archive, extract_root))
        cwd = os.getcwd()
        tar = tarfile.open(archive, "r")
        os.chdir(extract_root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print("Extraction done!")
        

class Caltech101(VisionDataset):
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
        ``annotation``. Can also be a list to output a tuple with all specified target types.
        ``category`` represents the target class, and ``annotation`` is a list of points
        from a hand-generated outline. Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self,
                 root: str,
                 phase: str = 'train',
                 is_return_origin: bool = True,
                 target_type: Union[List[str], str] = "category",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 train_size: int = 30,
                 valid_ratio: float = 0.2,
                ) -> None:
        super(Caltech101, self).__init__(root,
                                         transform=transform,
                                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        
        self.is_return_origin = is_return_origin
        
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation"))
                            for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {"Faces": "Faces_2",
                    "Faces_easy": "Faces_3",
                    "Motorbikes": "Motorbikes_16",
                    "airplanes": "Airplanes_Side_2"}
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])
            
        # check previous train and validation indices
        if os.path.isfile(os.path.join(self.root, 'train_dic.pickle')) and os.path.isfile(
                os.path.join(self.root, 'valid_dic.pickle')):
            with open(os.path.join(self.root, 'train_dic.pickle'), 'rb') as file:
                train_dic = pickle.load(file)
            with open(os.path.join(self.root, 'valid_dic.pickle'), 'rb') as file:
                valid_dic = pickle.load(file)
        # if no previous train and validation indices, sample train and validation data
        else:
            train_dic = {}
            valid_dic = {}
            for c in self.categories:
                fileslist = os.listdir(os.path.join(self.root, "101_ObjectCategories", c))
                n = len(list(filter(lambda file: file.endswith(".jpg"), fileslist)))
                # select 60 images randomly as training images per class
                train_index = random.sample(range(1, n + 1), k=train_size)
                valid_index = random.sample(train_index, k=int(train_size * valid_ratio))
                train_index = list(set(train_index).difference(set(valid_index)))
                train_dic[c] = train_index
                valid_dic[c] = valid_index
            with open(os.path.join(self.root, 'train_dic.pickle'), 'wb') as file:
                pickle.dump(train_dic, file)
            with open(os.path.join(self.root, 'valid_dic.pickle'), 'wb') as file:
                pickle.dump(valid_dic, file)
        
        # generate new index, label(y), and map (between label number and text label)
        self.index: List[int] = []
        self.y = []
        self.map = {}
        for (i, c) in enumerate(self.categories):
            if 'train' in phase.lower():
                self.index.extend(train_dic[c])
                self.y.extend(len(train_dic[c]) * [i])
            if 'valid' in phase.lower():
                self.index.extend(valid_dic[c])
                self.y.extend(len(valid_dic[c]) * [i])
            if 'test' in phase.lower():
                fileslist = os.listdir(os.path.join(self.root, "101_ObjectCategories", c))
                n = len(list(filter(lambda file: file.endswith(".jpg"), fileslist)))
                self.index.extend(
                    list(set(range(1, n + 1)).difference(set(train_dic[c])).difference(set(valid_dic[c]))))
                self.y.extend((n - train_size) * [i])
            self.map[i] = c.split('.')[-1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(os.path.join(self.root,
                                      "101_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "image_{:04d}.jpg".format(self.index[index])))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(os.path.join(self.root,
                                                     "Annotations",
                                                     self.annotation_categories[self.y[index]],
                                                     "annotation_{:04d}.mat".format(self.index[index])))
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]
        
        
        if self.is_return_origin and self.transform is not None:
            img, origin = self.transform(img)
        elif self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_return_origin:
            return img, origin, target
        else:
            return img, target


    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_root = self.root
        extract_root = download_root
        filename = "101_ObjectCategories.tar"
        url = "https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp"
        archive = os.path.join(download_root, filename)
        gdown.download(url, archive, quiet=False)
        
        # extract file
        print("Extracting {} to {}".format(archive, extract_root))
        cwd = os.getcwd()
        tar = tarfile.open(archive, "r")
        os.chdir(extract_root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print("Extraction done!")
        
    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)