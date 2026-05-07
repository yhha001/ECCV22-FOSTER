import logging
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000


class DataManager(object):
    def __init__(
        self,
        dataset_name,
        shuffle,
        seed,
        init_cls,
        increment,
        data_path=None,
        forget_classes_path=None,
        move_forget_classes_to_end=False,
        exclude_forget_classes_from_schedule=False,
    ):
        self.dataset_name = dataset_name
        self._custom_increments = None
        self._setup_data(
            dataset_name,
            shuffle,
            seed,
            data_path,
            init_cls=init_cls,
            increment=increment,
            forget_classes_path=forget_classes_path,
            move_forget_classes_to_end=move_forget_classes_to_end,
            exclude_forget_classes_from_schedule=exclude_forget_classes_from_schedule,
        )
        assert init_cls <= len(self._class_order), "No enough classes."
        if self._custom_increments is not None:
            self._increments = self._custom_increments
        else:
            self._increments = [init_cls]
            while sum(self._increments) + increment < len(self._class_order):
                self._increments.append(increment)
            offset = len(self._class_order) - sum(self._increments)
            if offset > 0:
                self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(
                    set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(
            val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(
        self,
        dataset_name,
        shuffle,
        seed,
        data_path=None,
        init_cls=None,
        increment=None,
        forget_classes_path=None,
        move_forget_classes_to_end=False,
        exclude_forget_classes_from_schedule=False,
    ):
        idata = _get_idata(dataset_name, data_path)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order

        forget_classes = _load_class_ids(forget_classes_path)
        if forget_classes:
            if not move_forget_classes_to_end:
                raise ValueError(
                    "forget_classes_path was provided, but move_forget_classes_to_end is false."
                )
            forget_set = set(forget_classes)
            num_classes = len(order)
            invalid = sorted(c for c in forget_set if c < 0 or c >= num_classes)
            if invalid:
                raise ValueError(
                    "forget_classes_path contains labels outside 0..{}: {}".format(
                        num_classes - 1, invalid[:10]
                    )
                )

            base_order = list(order)
            order = [
                original_class
                for new_label, original_class in enumerate(base_order)
                if new_label not in forget_set
            ] + [
                original_class
                for new_label, original_class in enumerate(base_order)
                if new_label in forget_set
            ]
            logging.info(
                "Moved {} forget classes to the end of the class order.".format(
                    len(forget_set)
                )
            )

            if exclude_forget_classes_from_schedule:
                self._custom_increments = _retained_task_increments(
                    num_classes,
                    init_cls,
                    increment,
                    forget_set,
                )
                logging.info(
                    "Retained-class task increments after exclusion: {}".format(
                        self._custom_increments
                    )
                )

        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(
            self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(
                y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _load_class_ids(path):
    if not path:
        return []
    with open(path, "r") as handle:
        tokens = [tok for tok in re.split(r"[\s,]+", handle.read().strip()) if tok]
    return [int(tok) for tok in tokens]


def _default_increments(num_classes, init_cls, increment):
    increments = [init_cls]
    while sum(increments) + increment < num_classes:
        increments.append(increment)
    offset = num_classes - sum(increments)
    if offset > 0:
        increments.append(offset)
    return increments


def _retained_task_increments(num_classes, init_cls, increment, forget_set):
    if init_cls is None or increment is None:
        raise ValueError("init_cls and increment are required to build retained task increments.")
    increments = []
    start = 0
    for task_size in _default_increments(num_classes, init_cls, increment):
        task_labels = range(start, start + task_size)
        retained = sum(1 for label in task_labels if label not in forget_set)
        if retained > 0:
            increments.append(retained)
        start += task_size
    if not increments:
        raise ValueError("All classes were excluded from the retained training schedule.")
    return increments


def _get_idata(dataset_name, data_path=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10(data_path)
    elif name == "cifar100":
        return iCIFAR100(data_path)
    elif name == "imagenet1000":
        return iImageNet1000(data_path)
    elif name == "imagenet100":
        return iImageNet100(data_path)
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
