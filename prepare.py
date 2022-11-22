"""
Prepare the training data
"""

import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
from pathlib import Path
from joblib import Parallel, delayed

from omegaconf import OmegaConf

# own module imports
import utils.utils as utils

log = utils.get_logger()


def read_csv(filename):
    """ Data loader """

    data = pd.read_csv(
        filename, 
        parse_dates=['timestamp'], 
        index_col='timestamp',
        dtype={
            'x': 'f4', 
            'y': 'f4', 
            'z': 'f4',
            'mag': 'f4',
            'annotation': 'Int64',
            'task_code': str,
            'tremor': 'Int64',
            'bradykinesia': 'Int64',
            'dyskinesia': 'Int64'
        }
    )
    return data


def resize(x, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """
    from scipy.interpolate import interp1d

    length_orig = x.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    x = interp1d(t_orig, x, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )
    return x


def is_good_quality(w, sample_rate, winsec, tolerance=0.01):
    """ Window quality check """

    if w.isna().any().any():
        return False

    if len(w) != sample_rate*winsec:
        return False

    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(winsec, 's')
    if np.abs(w_duration - target_duration) > tolerance * target_duration:
        return False

    return True


def make_windows(datafile, sample_rate, winsec, resample_rate, step_threshold, overlapsec=0, source_name='', annot_type='steps'):
    X, Ys, T, P, S = [], [], [], [], []
    accel_cols = ['x', 'y', 'z']

    data = read_csv(datafile)
    
    annotation_cols = data.columns.difference(accel_cols)
    p = os.path.basename(datafile).replace(".csv", "").replace("_wrist100", "") # PXX, PYY, XX_AAA, YY_BBB

    for i in range(0, len(data), sample_rate*(winsec-overlapsec)):
        w = data.iloc[i:i + sample_rate*winsec]

        if not is_good_quality(w, sample_rate, winsec):
            continue

        t = w.index[0].to_datetime64()
        x = w[accel_cols].values

        if annot_type == 'steps':
            ys = {'is_walk': 1 if w['annotation'].sum() >= step_threshold else 0,
                  'steps': w['annotation'].sum()}
        else:
            ys = {**{'is_walk': w['annotation'].mode(dropna=False).iloc[0]},
                  **{label: w[label].mode(dropna=False).iloc[0] 
                        for label in annotation_cols.difference(['annotation'])}}
        X.append(x)
        Ys.append(ys)
        T.append(t)
        P.append(p)
        S.append(source_name)

    X = np.asarray(X)
    Ys = pd.DataFrame(Ys)
    T = np.asarray(T)
    P = np.asarray(P)
    S = np.asarray(S)

    if sample_rate != resample_rate:
        X = resize(X, int(resample_rate * winsec))

    return X, Ys, T, P, S


def prepare_data(cfg):
    if cfg.data.overwrite or not os.path.exists(cfg.data.processed_data+"/config.txt"):
        log.info("Processing raw data.")
        files = [(source_name, source_info, filename) for source_name, source_info in cfg.data.sources.items()
                                for filename in glob("{}/{}/*.csv".format(cfg.data.raw_data, source_name))]

        X, Ys, T, P, S = zip(*Parallel(n_jobs=cfg.num_workers)(
            delayed(make_windows)(filename, source_info['raw_sample_rate'], cfg.data.winsec, 
                                  cfg.data.sample_rate, cfg.data.step_threshold, source_name=source_name,
                                  annot_type=source_info['annot_type']) for source_name, source_info, filename in tqdm(files)))
        
        X = np.vstack(X)
        T = np.hstack(T)
        P = np.hstack(P)
        S = np.hstack(S)
        Ys = pd.concat(Ys)

        Path(cfg.data.processed_data).mkdir(parents=True, exist_ok=True)

        np.save(cfg.data.processed_data+"/X.npy", X)
        np.save(cfg.data.processed_data+"/T.npy", T)
        np.save(cfg.data.processed_data+"/pid.npy", P)
        np.save(cfg.data.processed_data+"/source.npy", S)
    
        for col in Ys.columns:
            np.save("{}/Y_{}.npy".format(cfg.data.processed_data, col), np.array(Ys[col]))
        
        with open(cfg.data.processed_data+"/config.txt", "w") as f:
            f.write(str({
                'Data Source(s)': cfg.data.sources,
                'Sample Rate': "{}Hz".format(cfg.data.sample_rate),
                'Window Size': "{}s".format(cfg.data.winsec),
                'Step Walking Threshold': "{} step(s) per window".format(cfg.data.step_threshold)
            }))

    else:
        with open(cfg.data.processed_data+"/config.txt", "rt") as f:
            config = f.read()
            log.info(config)
        log.info("Using already processed data.")


if __name__ == "__main__":
    np.random.seed(42)

    cfg = OmegaConf.load("conf/config.yaml")
    log.info(str(OmegaConf.to_yaml(cfg)))

    prepare_data(cfg)
