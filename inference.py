import actipy
import argparse
import os
import logging
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils.dataloader import NormalDataset
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed

from models.hmm import HMM
import utils.utils as utils

DEVICE_HZ = 30  # Hz
WINDOW_SEC = 30  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%

log = logging.getLogger('hmm')
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
fmt = logging.Formatter(fmt='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(fmt)
log.addHandler(handler)

start_time = datetime.now()


def vectorized_stride_v2(acc, time, window_size, stride_size):
    start = 0
    max_time = len(time)

    sub_windows = (start +
                   np.expand_dims(np.arange(window_size), 0) +
                   # Create a rightmost vector as [0, V, 2V, ...].
                   np.expand_dims(np.arange(max_time + 1, step=stride_size), 0).T
                   )[:-1]  # drop the last one

    return acc[sub_windows], time[sub_windows]


def is_good_quality(w):
    """ Window quality check """

    if w.isna().any().any():
        return False

    if len(w) != WINDOW_LEN:
        return False

    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(WINDOW_SEC, 's')
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
        return False

    return True


def df_to_windows(data):
    """
    Convert the dataframe to a numpy array of windows

    :param pd.DataFrame data: Input data from actipy
    :return: window array, timestamp Series
    :rtype: (np.ndarray, pd.Series)
    """

    def window(i, data):
        w = data.iloc[i:i + WINDOW_LEN]
        if is_good_quality(w):
            acc = w[['x', 'y', 'z']].values
            return acc, w.index[0]
        else:
            return np.empty((WINDOW_LEN, 3)), np.nan

    x, t = zip(*Parallel(n_jobs=8)(
        delayed(window)(i, data)
        for i in tqdm(range(0, len(data), WINDOW_STEP_LEN))
    ))

    t = pd.Series(t)
    na = t.isna()
    t = t.loc[~na]

    x = np.stack(x)
    x = x[~na]

    return x, t


def df_to_windows2(data):
    """
    Numpy vectorised version of df_to_windows (much faster)
    """
    acc = data[['x', 'y', 'z']].values
    time = data.index.values

    # convert to windows, shape: (rows, WINDOW_LEN, 3)
    x, t = vectorized_stride_v2(acc, time, WINDOW_LEN, WINDOW_STEP_LEN)

    # drop the whole window if it contains a NaN
    na = np.isnan(x).any(axis=1).any(axis=1)
    x = x[~na]
    t = t[~na]

    return x, t[:, 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='SSL UKB', usage='Apply the SSL+HMM model on a UKB cwa file.')
    parser.add_argument('input_file', type=str, help='input cwa file')
    args = parser.parse_args()

    input_file = args.input_file
    input_path = Path(input_file)

    pid = input_path.stem.split('_')[0]
    group = input_path.parent.stem if 'group' in input_path.parent.stem else None

    log.info(input_file)
    log.info('%s %s', group, pid)

    np.random.seed(42)
    torch.manual_seed(42)

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

    data_start = data.index[0]
    data_end = data.index[-1]

    log.info('Windowing')
    X, T = df_to_windows2(data)
    del data

    dataset = NormalDataset(np.transpose(X, (0, 2, 1)), name=pid)

    dataloader = DataLoader(
        dataset,
        batch_size=120,
        shuffle=False,
        num_workers=0,
    )

    # load pretrained SSL model and weights
    if cfg.torch_path:
        log.info('Using %s', cfg.torch_path)
        sslnet: nn.Module = torch.hub.load(cfg.torch_path, 'harnet30', source='local', class_num=4, pretrained=False)
    else:
        repo = 'OxWearables/ssl-wearables'
        torch.hub.set_dir(cfg.torch_cache)
        sslnet: nn.Module = torch.hub.load(repo, 'harnet30', class_num=4, pretrained=False)

    model_dict = torch.load(os.path.join(cfg.pretrained_model_root, 'state_dict.pt'), map_location=my_device)
    sslnet.load_state_dict(model_dict)
    sslnet.eval()
    sslnet.to(my_device)

    sslnet.labels_ = np.array(['light', 'moderate-vigorous', 'sedentary', 'sleep'])
    sslnet.classes_ = np.array([0, 1, 2, 3])

    # load pretrained HMM
    hmm_ssl = HMM(sslnet.classes_, uniform_prior=cfg.hmm.uniform_prior)
    hmm_ssl.load(cfg.hmm.path_ssl)

    # inference
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
    df['label'] = sslnet.labels_[y_pred]
    df['label'] = df['label'].astype(dtype)
    df['label_hmm'] = sslnet.labels_[y_pred_hmm]
    df['label_hmm'] = df['label_hmm'].astype(dtype)

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
