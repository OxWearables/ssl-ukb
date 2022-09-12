import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal

from joblib import Parallel, delayed
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier

import utils.utils as utils

log = utils.get_logger()


def get_rf(num_workers=1, oob_score=True):
    """
    Return an untrained Random Forest.

    :param int num_workers: Set >1 for multiprocessing during training.
    :param bool oob_score: Calculate out-of-bag accuracy scores.
    :rtype: BalancedRandomForestClassifier
    """

    return BalancedRandomForestClassifier(
        n_estimators=3000,
        replacement=True,
        sampling_strategy="not minority",
        n_jobs=num_workers,
        random_state=42,
        oob_score=oob_score
    )


def extract_features(data, sample_rate, num_workers=1):
    """
    Extract handcrafted features from xyz data.

    :param np.ndarray data: Input data with shape (num_windows, window_len, 3).
    :param int sample_rate: Data sample rate.
    :param int num_workers: Set >1 for multiprocessing.
    :return: Feature matrix of shape (num_windows, num_features).
    :rtype: np.ndarray
    """

    x_feats = Parallel(n_jobs=num_workers)(
        delayed(_handcraft_features)(x, sample_rate=sample_rate) for x in tqdm(data)
    )
    x_feats = pd.DataFrame(x_feats).to_numpy()

    return x_feats


def _handcraft_features(xyz, sample_rate):
    """Our baseline handcrafted features. xyz is a window of shape (N,3)"""

    feats = {}
    feats["xMean"], feats["yMean"], feats["zMean"] = np.mean(xyz, axis=0)
    feats["xStd"], feats["yStd"], feats["zStd"] = np.std(xyz, axis=0)
    feats["xRange"], feats["yRange"], feats["zRange"] = np.ptp(xyz, axis=0)

    x, y, z = xyz.T

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # ignore div by 0 warnings
        feats["xyCorr"] = np.nan_to_num(np.corrcoef(x, y)[0, 1])
        feats["yzCorr"] = np.nan_to_num(np.corrcoef(y, z)[0, 1])
        feats["zxCorr"] = np.nan_to_num(np.corrcoef(z, x)[0, 1])

    m = np.linalg.norm(xyz, axis=1)

    feats["mean"] = np.mean(m)
    feats["std"] = np.std(m)
    feats["range"] = np.ptp(m)
    feats["mad"] = stats.median_abs_deviation(m)
    feats["kurt"] = stats.kurtosis(m)
    feats["skew"] = stats.skew(m)
    feats["enmomean"] = np.mean(np.abs(m - 1))

    # Spectrum using Welch's method with 3s segment length
    # First run without detrending to get the true spectrum
    freqs, powers = signal.welch(
        m,
        fs=sample_rate,
        nperseg=3 * sample_rate,
        noverlap=2 * sample_rate,
        detrend=False,
        average="median",
    )

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # ignore div by 0 warnings
        feats["pentropy"] = np.nan_to_num(stats.entropy(powers + 1e-16))

    # Spectrum using Welch's method with 3s segment length
    # Now do detrend to find dominant freqs
    freqs, powers = signal.welch(
        m,
        fs=sample_rate,
        nperseg=3 * sample_rate,
        noverlap=2 * sample_rate,
        detrend="constant",
        average="median",
    )

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]
    if len(peaks) >= 2:
        feats["f1"] = peak_freqs[peak_ranks[0]]
        feats["f2"] = peak_freqs[peak_ranks[1]]
    elif len(peaks) == 1:
        feats["f1"] = feats["f2"] = peak_freqs[peak_ranks[0]]
    else:
        feats["f1"] = feats["f2"] = 0

    return feats
