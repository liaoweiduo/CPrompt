import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from PIL import Image
import torch

from utils.CFST_data import CGQA, COBJ
from timm.data import create_transform
from torchvision.transforms.functional import crop, InterpolationMode


def build_cgqa_transform(img_size=(224, 224)):
    _train_transform = create_transform(
        input_size=img_size,
        is_training=True,
        color_jitter=0.3,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )
    # replace RandomResizedCropAndInterpolation with Resize, for not cropping img and missing concepts
    _train_transform.transforms[0] = transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC)

    return _train_transform.transforms


def build_default_transform(image_size=(224, 224), is_train=True, normalize=True):
    """
    Default transforms borrowed from MetaShift.
    Imagenet normalization.
    """
    _train_transform = [
        transforms.Resize(image_size),  # allow reshape but not equal scaling
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]
    _eval_transform = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
    if normalize:
        _train_transform.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ))
        _eval_transform.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ))

    # _default_train_transform = transforms.Compose(_train_transform)
    # _default_eval_transform = transforms.Compose(_eval_transform)

    if is_train:
        return _train_transform
    else:
        return _eval_transform


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCGQA(iData):
    use_path = False
    train_trsf = build_cgqa_transform()
    test_trsf = build_default_transform(is_train=False)
    common_trsf = []

    class_order = np.arange(100).tolist()

    # # big first task
    # labels_order = [
    #     26, 86, 2, 55, 75, 93, 16, 73, 54, 95,
    #     53, 92, 78, 13, 7, 30, 22, 24, 33, 8,
    #     43, 62, 3, 71, 45, 48, 6, 99, 82, 76,
    #     60, 80, 90, 68, 51, 27, 18, 56, 63, 74,
    #     1, 61, 42, 41, 4, 15, 17, 40, 38, 5,
    #     91, 59, 0, 34, 28, 50, 11, 35, 23, 52,
    #     10, 31, 66, 57, 79, 85, 32, 84, 14, 89,
    #     19, 29, 49, 97, 98, 69, 20, 94, 72, 77,
    #     25, 37, 81, 46, 39, 65, 58, 12, 88, 70,
    #     87, 36, 21, 83, 9, 96, 67, 64, 47, 44]
    # classes_per_task = [10 * (10 - args.num_tasks + 1), 10]

    def download_data(self):

        train_dataset = CGQA('../datasets', train=True)
        test_dataset = CGQA('../datasets', train=False)

        self.train_data, self.train_targets = train_dataset.train_set.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.test_set.data, np.array(test_dataset.targets)


class iCOBJ(iData):
    use_path = False
    train_trsf = build_cgqa_transform()
    test_trsf = build_default_transform(is_train=False)
    common_trsf = []

    class_order = np.arange(30).tolist()

    def download_data(self):

        train_dataset = COBJ('../datasets', train=True)
        test_dataset = COBJ('../datasets', train=False)

        self.train_data, self.train_targets = train_dataset.train_set.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.test_set.data, np.array(test_dataset.targets)


class iDomainnetCIL(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        rootdir = "[DATA-PATH]"
        train_txt = './data/datautils/domainnet/train.txt'
        test_txt = './data/datautils/domainnet/test.txt'

        train_images = []
        train_labels = []
        with open(train_txt, 'r') as dict_file:
            for line in dict_file:
                (value, key) = line.strip().split(' ')
                train_images.append(os.path.join(rootdir, value))
                train_labels.append(int(key))
        
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = []
        test_labels = []
        with open(test_txt, 'r') as dict_file:
            for line in dict_file:
                (value, key) = line.strip().split(' ')
                test_images.append(os.path.join(rootdir, value))
                test_labels.append(int(key))
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        self.train_data = train_images
        self.train_targets = train_labels
        
        self.test_data = test_images
        self.test_targets = test_labels


class iImageNetR(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        rootdir = "[DATA-PATH]"
        train_txt = './data/datautils/imagenet-r/coda_train.txt'
        test_txt = './data/datautils/imagenet-r/coda_test.txt'

        train_images = []
        train_labels = []
        with open(train_txt, 'r') as dict_file:
            for line in dict_file:
                
                (value, key) = line.strip().split(' ')
                train_images.append(os.path.join(rootdir, value))
                train_labels.append(int(key))
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        test_images = []
        test_labels = []
        with open(test_txt, 'r') as dict_file:
            for line in dict_file:
                (value, key) = line.strip().split(' ')
                test_images.append(os.path.join(rootdir, value))
                test_labels.append(int(key))
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        self.train_data = train_images
        self.train_targets = train_labels
        self.test_data = test_images
        self.test_targets = test_labels
        

class iCIFAR100_vit(iData):
    use_path = False
    train_trsf = [
        transforms.Resize(256),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()
   
    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)
        

class iStanford_cars(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
    ]

    class_order = np.arange(45).tolist()

    def download_data(self):
        rootdir = "[DATA-PATH]"
        train_txt = './data/datautils/stanfordcars/train.txt'
        test_txt = './data/datautils/stanfordcars/test.txt'
        
        train_images = []
        train_labels = []
        with open(train_txt, 'r') as dict_file:
            for line in dict_file:
                # (key, value) = line.strip().split('\t')
                (value, key) = line.strip().split(' ')
                train_images.append(os.path.join(rootdir, value))
                train_labels.append(int(key))
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        test_images = []
        test_labels = []
        with open(test_txt, 'r') as dict_file:
            for line in dict_file:
                # (key, value) = line.strip().split('\t')
                (value, key) = line.strip().split(' ')
                test_images.append(os.path.join(rootdir, value))
                test_labels.append(int(key))
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        self.train_data = train_images
        self.train_targets = train_labels
        self.test_data = test_images
        self.test_targets = test_labels
