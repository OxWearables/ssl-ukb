"""
Perform inference with pretrained SSL+HMM on a UKB accelerometer file.
Requires a pretrained SSL and HMM (generated by hmm_train.py).

Arguments:
    Input file, in the format {eid}_*.cwa.gz

Example usage:
    python inference.py /data/ukb-accelerometer/group1/4027057_90001_0_0.cwa.gz

Output:
    Prediction DataFrame in {eid}.parquet format, stored in ukb_output_root/ (see conf/config.yaml).
    If the input file is stored in a groupX folder, output will be in ukb_output_root/groupX/

    An {eid}_info.csv file will be saved alongside the parquet file with the actipy info dict.
"""

import actipy
import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils.dataloader import NormalDataset
from datetime import datetime

from models.hmm import HMM
import utils.utils as utils

DEVICE_HZ = 30  # Hz
WINDOW_SEC = 30  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks

log = utils.get_logger()

start_time = datetime.now()


def vectorized_stride_v2(acc, time, window_size, stride_size):
    """
    Numpy vectorised windowing with stride (super fast!). Will discard the last window.

    :param np.ndarray acc: Accelerometer data array, shape (nsamples, nchannels)
    :param np.ndarray time: Time array, shape (nsamples, )
    :param int window_size: Window size in n samples
    :param int stride_size: Stride size in n samples
    :return: Windowed data and time arrays
    :rtype: (np.ndarray, np.ndarray)
    """
    start = 0
    max_time = len(time)

    sub_windows = (start +
                   np.expand_dims(np.arange(window_size), 0) +
                   # Create a rightmost vector as [0, V, 2V, ...].
                   np.expand_dims(np.arange(max_time + 1, step=stride_size), 0).T
                   )[:-1]  # drop the last one

    return acc[sub_windows], time[sub_windows]


def df_to_windows(df):
    """
    Convert a time series dataframe (e.g.: from actipy) to a windowed Numpy array.

    :param pd.DataFrame df: A dataframe with DatetimeIndex and x, y, z columns
    :return: Data array with shape (nwindows, WINDOW_LEN, 3), Time array with shape (nwindows, )
    :rtype: (np.ndarray, np.ndarray)
    """

    acc = df[['x', 'y', 'z']].to_numpy()
    time = df.index.to_numpy()

    # convert to windows
    x, t = vectorized_stride_v2(acc, time, WINDOW_LEN, WINDOW_STEP_LEN)

    # drop the whole window if it contains a NaN
    na = np.isnan(x).any(axis=1).any(axis=1)
    x = x[~na]
    t = t[~na]

    return x, t[:, 0]  # only return the first timestamp for each window


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='SSL UKB', usage='Apply the SSL+HMM model on a UKB cwa file.')
    parser.add_argument('input_file', type=str, help='input cwa file')
    args = parser.parse_args()

    input_file = args.input_file
    input_path = Path(input_file)

    # get pid and group from input string
    pid = input_path.stem.split('_')[0]
    group = input_path.parent.stem if 'group' in input_path.parent.stem else None

    log.info(input_file)
    log.info('%s %s', group, pid)

    np.random.seed(42)
    torch.manual_seed(42)

    # load config
    cfg = OmegaConf.load("conf/config.yaml")

    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    elif cfg.multi_gpu is True:
        my_device = "cuda:0"  # use the first GPU as master
    else:
        my_device = "cpu"

    # load data and construct dataloader
    data, info = actipy.read_device(input_file,
                                    lowpass_hz=None,
                                    calibrate_gravity=True,
                                    detect_nonwear=True,
                                    resample_hz=DEVICE_HZ)
    log.info(data.head(1))
    log.info(info)
    info = pd.DataFrame(info, index=[1])

    # store original start/end times for reindexing later
    data_start = data.index[0]
    data_end = data.index[-1]

    # prepare dataset
    log.info('Windowing')
    X, T = df_to_windows(data)
    del data  # free up memory

    # transpose for pytorch channel first format
    dataset = NormalDataset(np.transpose(X, (0, 2, 1)), name=pid)

    dataloader = DataLoader(
        dataset,
        batch_size=120,
        shuffle=False,
        num_workers=0,
    )

    # load pretrained SSL model and weights
    if cfg.torch_path:
        # use repo from disk (for offline use)
        log.info('Using %s', cfg.torch_path)
        sslnet: nn.Module = torch.hub.load(cfg.torch_path, 'harnet30', source='local', class_num=4, pretrained=False)
    else:
        # download repo from github
        repo = 'OxWearables/ssl-wearables'
        torch.hub.set_dir(cfg.torch_cache)
        sslnet: nn.Module = torch.hub.load(repo, 'harnet30', class_num=4, pretrained=False)

    # load pretrained weights
    model_dict = torch.load(cfg.sslnet.weights, map_location=my_device)
    sslnet.load_state_dict(model_dict)
    sslnet.eval()
    sslnet.to(my_device)

    sslnet.labels_ = utils.labels_
    sslnet.classes_ = utils.classes_

    # load pretrained HMM
    hmm_ssl = HMM(sslnet.classes_, uniform_prior=cfg.hmm.uniform_prior)
    hmm_ssl.load(cfg.hmm.weights_ssl)

    # do inference
    log.info('SSL inference')
    _, y_prob, _ = utils.mlp_predict(
        sslnet, dataloader, my_device, cfg, output_logits=True
    )

    y_pred = np.argmax(y_prob, axis=1)

    log.info('HMM smoothing')
    y_pred_hmm = hmm_ssl.viterbi(y_pred)

    # construct dataframe
    df = utils.raw_to_df(X, y_prob, T, sslnet.labels_, label_proba=True, reindex=False)

    dtype = pd.CategoricalDtype(categories=sslnet.labels_)
    df['label'] = pd.Series(sslnet.labels_[y_pred], index=df.index, dtype=dtype)
    df['label_hmm'] = pd.Series(sslnet.labels_[y_pred_hmm], index=df.index, dtype=dtype)

    # reindex for missing values
    newindex = pd.date_range(data_start, data_end, freq='{s}S'.format(s=WINDOW_SEC))
    df = df.reindex(newindex)
    log.info('Done')

    # save dataframe
    log.info('Saving dataframe')

    if group:
        path = os.path.join(cfg.ukb_output_root, group)
    else:
        path = cfg.ukb_output_root

    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_parquet(os.path.join(path, pid + '.parquet'), engine='pyarrow')
    info.to_csv(os.path.join(path, pid + '_info.csv'))

    end_time = datetime.now()
    log.info('Duration: %s', end_time - start_time)
