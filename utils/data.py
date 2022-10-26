"""
Data loading and augmentation utilities
"""

import torch
import random
import numpy as np
import pandas as pd

from torch.utils.data.dataset import Dataset
from transforms3d.axangles import axangle2mat
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

import utils.utils as utils

log = utils.get_logger()


class RandomSwitchAxis:
    """
    Randomly switch the three axises for the raw files
    Input size: 3 * FEATURE_SIZE
    """

    def __call__(self, sample):
        # print(sample.shape)
        # 3 * FEATURE
        x = sample[0, :]
        y = sample[1, :]
        z = sample[2, :]

        choice = random.randint(1, 6)

        if choice == 1:
            sample = torch.stack([x, y, z], dim=0)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=0)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=0)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=0)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=0)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=0)
        # print(sample.shape)
        return sample


class RotationAxis:
    """
    Rotation along an axis
    """

    def __call__(self, sample):
        # 3 * FEATURE_SIZE
        sample = np.swapaxes(sample, 0, 1)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        sample = np.matmul(sample, axangle2mat(axis, angle))
        sample = np.swapaxes(sample, 0, 1)
        return sample


class NormalDataset(Dataset):
    def __init__(self,
                 X,
                 y=None,
                 pid=None,
                 name="",
                 is_labelled=False,
                 transform=False,
                 transpose_channels_first=True):

        if transpose_channels_first:
            X = np.transpose(X, (0, 2, 1))

        self.X = torch.from_numpy(X)
        if y is not None:
            self.y = torch.tensor(y)
        self.isLabel = is_labelled
        self.pid = pid
        if transform:
            self.transform = transforms.Compose([RandomSwitchAxis(), RotationAxis()])
        else:
            self.transform = None
        log.info(name + " set sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]

        if self.isLabel:
            y = self.y[idx]
        else:
            y = np.NaN

        if self.pid is not None:
            pid = self.pid[idx]
        else:
            pid = np.NaN

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, y, pid


def load_data(cfg, cv='GroupKFold'):
    X = np.load(cfg.data.X_path, mmap_mode='r')
    y = np.load(cfg.data.Y_path)
    pid = np.load(cfg.data.PID_path, allow_pickle=True)  # participant IDs
    time = np.load(cfg.data.time_path, allow_pickle=True)
    source = np.load(cfg.data.processed_data+'/Y_source.npy', allow_pickle=True)

    log.info('X shape: %s', X.shape)
    log.info('Y shape: %s', y.shape)
    log.info('Label distribution:\n%s', pd.Series(y).value_counts())

    # TODO: Manage transformation given missing data with label -1
    #y = utils.le.transform(y)

    input_size = cfg.data.winsec * cfg.data.sample_rate
    if X.shape[1] == input_size:
        log.info("No need to downsample")
    else:
        X = utils.resize(X, input_size)

    X = X.astype(
        "f4"
    )  # PyTorch defaults to float32

    if cv == 'GroupKFold':
        if len(np.unique(source))> 1:
            # generate train/test splits
            # Stratify based on data source
            folds = StratifiedGroupKFold(
                cfg.num_folds
            ).split(X, source, groups=pid)
        else:
            folds = StratifiedGroupKFold(
                cfg.num_folds
            ).split(X, y, groups=pid)
    else:
        folds = GroupShuffleSplit(
            cfg.num_folds, test_size=0.2, random_state=42
        ).split(X, y, groups=pid)

    return {fold: split_data(X, y, pid, time, train_idx, test_idx, fold) 
                for fold, (train_idx, test_idx) in enumerate(folds)}

def split_data(X, y, pid, time, train_idx, test_idx, fold):
    x_test = X[test_idx]
    y_test = y[test_idx]
    time_test = time[test_idx]
    group_test = pid[test_idx]

    # further split train into train/val
    X = X[train_idx]
    y = y[train_idx]
    pid = pid[train_idx]
    time = time[train_idx]

    # Remove unlabelled training data identified with -1
    lablled_mask = y != -1
    
    X = X[lablled_mask]
    y = y[lablled_mask]
    pid = pid[lablled_mask]
    time = time[lablled_mask]

    folds = GroupShuffleSplit(
        1, test_size=0.125, random_state=41+fold
    ).split(X, y, groups=pid)
    train_idx, val_idx = next(folds)

    x_train = X[train_idx]
    x_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    time_train = time[train_idx]
    time_val = time[val_idx]

    group_train = pid[train_idx]
    group_val = pid[val_idx]

    return (
        x_train, y_train, group_train, time_train,
        x_val, y_val, group_val, time_val,
        x_test, y_test, group_test, time_test,
    )

def get_inverse_class_weights(y):
    """ Return a list with inverse class frequencies in y """
    import collections

    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    log.info("Inverse class weights: ")
    log.info(weights)

    return weights
