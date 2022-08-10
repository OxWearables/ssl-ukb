import torch
import logging
import os
import utils.utils as utils
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn import preprocessing

log = logging.getLogger('hmm')


def load_data(cfg):
    X = np.load(cfg.data.X_path, mmap_mode='r')
    Y = np.load(cfg.data.Y_path)
    P = np.load(cfg.data.PID_path)  # participant IDs
    time = np.load(cfg.data.time_path)

    log.info('X shape: %s', X.shape)
    log.info('Y shape: %s', Y.shape)
    log.info('Label distribution: ')
    log.info(pd.Series(Y).value_counts())

    le = preprocessing.LabelEncoder()
    le.fit(Y)
    y = le.transform(Y)
    labels = le.classes_

    group_train = np.load(os.path.join(cfg.pretrained_model_root, 'group_train.npy'))
    group_train_rf = np.load(os.path.join(cfg.pretrained_model_root, 'rf_group_train.npy'))
    group_val = np.load(os.path.join(cfg.pretrained_model_root, 'group_val.npy'))
    group_test = np.load(os.path.join(cfg.pretrained_model_root, 'group_test.npy'))
    group_test_rf = np.load(os.path.join(cfg.pretrained_model_root, 'rf_group_test.npy'))

    train_idx = np.isin(P, np.unique(group_train)).nonzero()[0]
    train_idx_rf = np.isin(P, np.unique(group_train_rf)).nonzero()[0]
    val_idx = np.isin(P, np.unique(group_val)).nonzero()[0]
    test_idx = np.isin(P, np.unique(group_test)).nonzero()[0]
    test_idx_rf = np.isin(P, np.unique(group_test_rf)).nonzero()[0]

    # p_idx = {}
    # for person in np.unique(P):
    #     p_idx[person] = np.isin(P, person).nonzero()[0]

    input_size = cfg.data.input_size
    if X.shape[1] == input_size:
        log.info("No need to downsample")
        x_downsampled = X
    else:
        x_downsampled = utils.resize(X, input_size)

    x_downsampled = x_downsampled.astype(
        "f4"
    )  # PyTorch defaults to float32
    # channels first: (N,M,3) -> (N,3,M). PyTorch uses channel first format
    x_transposed = np.transpose(x_downsampled, (0, 2, 1))

    x_train = x_transposed[train_idx]
    x_train_rf = x_downsampled[train_idx_rf]
    y_train = y[train_idx]
    y_train_rf = y[train_idx_rf]
    time_train = time[train_idx]
    time_train_rf = time[train_idx_rf]

    x_val = x_transposed[val_idx]
    y_val = y[val_idx]
    time_val = time[val_idx]

    x_test = x_transposed[test_idx]
    x_test_rf = x_downsampled[test_idx_rf]
    y_test = y[test_idx]
    y_test_rf = y[test_idx_rf]
    time_test = time[test_idx]
    time_test_rf = time[test_idx_rf]

    return (
        x_train, y_train, group_train, time_train,
        x_train_rf, y_train_rf, group_train_rf, time_train_rf,
        x_val, y_val, group_val, time_val,
        x_test, y_test, group_test, time_test,
        x_test_rf, y_test_rf, group_test_rf, time_test_rf,
        le
    )


class NormalDataset(Dataset):
    def __init__(
        self,
        X,
        y=None,
        pid=None,
        name="",
        is_labelled=False,
    ):
        """
        Y needs to be in one-hot encoding
        X needs to be in N * Width
        Args:
            data_path (string): path to data
            files_to_load (list): subject names
            currently all npz format should allow support multiple ext

        """

        self.X = torch.from_numpy(X)
        self.y = y
        self.isLabel = is_labelled
        self.pid = pid
        log.info(name + " set sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]

        y = None
        if self.isLabel:
            y = self.y[idx]

        pid = None
        if self.pid is not None:
            pid = self.pid[idx]

        return sample, y, pid
