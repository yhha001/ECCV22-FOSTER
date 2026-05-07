import numpy as np
import os
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import torch
from .autoaugment import CIFAR10Policy, ImageNetPolicy


def _resolve_imagenet100_roots(data_path):
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    val_in_class_dirs = True

    raw_train_dir = os.path.join(data_path, "ILSVRC2012_image_train")
    raw_val_dir = os.path.join(data_path, "ILSVRC2012_image_val")
    if os.path.isdir(raw_train_dir) and os.path.isdir(raw_val_dir):
        train_dir = raw_train_dir
        val_dir = raw_val_dir
        val_in_class_dirs = False

    return train_dir, val_dir, val_in_class_dirs


def _load_imagenet100_subset(data_path, split_root):
    train_split = os.path.join(split_root, "train.txt")
    val_split = os.path.join(split_root, "eval.txt")
    train_dir, val_dir, val_in_class_dirs = _resolve_imagenet100_roots(data_path)

    train_data, train_targets, class_to_idx = _read_imagenet100_split(
        train_dir, train_split
    )
    val_data, val_targets, _ = _read_imagenet100_split(
        val_dir, val_split, class_to_idx, include_class_subdir=val_in_class_dirs
    )

    return train_data, train_targets, val_data, val_targets, class_to_idx


def _read_imagenet100_split(base_dir, split_file, class_to_idx=None, include_class_subdir=True):
    images, labels = [], []
    current_class = None

    if class_to_idx is None:
        class_to_idx = {}

    with open(split_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Class marker lines do not contain a filename extension.
            if "/" not in line:
                current_class = line
                if current_class not in class_to_idx:
                    class_to_idx[current_class] = len(class_to_idx)
                continue

            if current_class is None:
                raise ValueError(
                    "Invalid ImageNet-100 split file: image path before class name."
                )

            relative_path = line if include_class_subdir else os.path.basename(line)
            images.append(os.path.join(base_dir, relative_path))
            labels.append(class_to_idx[current_class])

    return np.array(images), np.array(labels), class_to_idx


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

    def __init__(self, data_path=None):
        self.data_path = data_path


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63/255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(
            './data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(
            './data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(
            './data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(
            './data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets)


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255),
        ImageNetPolicy(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        data_path = self.data_path or os.environ.get("FOSTER_IMAGENET1000_ROOT", "")
        assert data_path, "please specify the data path "
        train_dir = os.path.join(data_path, "train")
        test_dir = os.path.join(data_path, "val")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(
            train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255),
        ImageNetPolicy()
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        data_path = self.data_path or os.environ.get("FOSTER_IMAGENET100_ROOT", "")
        assert data_path, "please specify the data path "
        split_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "imagenet-sub")
        )
        (
            self.train_data,
            self.train_targets,
            self.test_data,
            self.test_targets,
            class_to_idx,
        ) = _load_imagenet100_subset(data_path, split_root)
        self.class_order = np.arange(len(class_to_idx)).tolist()
