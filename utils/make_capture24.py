import re
import glob
import os
import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm
from joblib import Parallel, delayed

# change these to reflect the folders on your system
DATA_DIR = '/data/UKBB/capture24/data/'  # location of Capture-24 (the .csv.gz files should be in this folder)
OUT_DIR = '/data/UKBB/capture24_30hz_w30_o0/'  # output location of the processed dataset

# number of CPU cores to use, need ~1.5GB memory per worker
# don't set this higher than the number of physical cores
NUM_WORKERS = 4

# don't edit below this line (unless deliberately changing parameters)
DEVICE_HZ = 100  # Hz
RESAMPLE_HZ = 30  # Hz
WINDOW_SEC = 30  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
DATAFILES = os.path.join(DATA_DIR, 'P*.csv.gz')
ANNOLABELFILE = os.path.join(DATA_DIR, 'annotation-label-dictionary.csv')
LABEL = 'label:Walmsley2020'

annolabel = pd.read_csv(ANNOLABELFILE, index_col='annotation')


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


def is_good_quality(w):
    """ Window quality check """

    if w.isna().any().any():
        return False

    if len(w) != WINDOW_LEN:
        return False

    if len(w['annotation'].unique()) > 1:
        return False

    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(WINDOW_SEC, 's')
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
        return False

    return True


def make(datafile):
    X, Y, T, P, = [], [], [], []

    data = pd.read_csv(datafile, parse_dates=['time'], index_col='time',
                       dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'str'})

    p = re.search(r'(P\d{3})', datafile, flags=re.IGNORECASE).group()

    for i in range(0, len(data), WINDOW_STEP_LEN):
        w = data.iloc[i:i + WINDOW_LEN]

        if not is_good_quality(w):
            continue

        t = w.index[0].to_datetime64()
        x = w[['x', 'y', 'z']].values
        y = annolabel.loc[w['annotation'][0], LABEL]

        X.append(x)
        Y.append(y)
        T.append(t)
        P.append(p)

    X = np.asarray(X)
    Y = np.asarray(Y)
    T = np.asarray(T)
    P = np.asarray(P)

    if DEVICE_HZ != RESAMPLE_HZ:
        X = resize(X, int(RESAMPLE_HZ * WINDOW_SEC))

    return X, Y, T, P


if __name__ == '__main__':
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    x, y, t, p = zip(
        *Parallel(n_jobs=NUM_WORKERS)(
            delayed(make)(datafile)
            for datafile in tqdm(glob.glob(DATAFILES))
        )
    )

    X = np.vstack(x)
    Y = np.hstack(y)
    T = np.hstack(t)
    P = np.hstack(p)

    np.save(os.path.join(OUT_DIR, 'X'), X)
    np.save(os.path.join(OUT_DIR, 'Y_Walmsley'), Y)
    np.save(os.path.join(OUT_DIR, 'time'), T)
    np.save(os.path.join(OUT_DIR, 'pid'), P)

    print(f"Saved in {OUT_DIR}")
    print("X shape:", X.shape)
    print("Y distribution:")
    print(pd.Series(Y).value_counts())
