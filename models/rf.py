import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal

from joblib import Parallel, delayed
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier

from models.utils import butterfilt
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
        delayed(_handcraft_features)(x, sample_rate=sample_rate) for x in tqdm(data, mininterval=60)
    )
    x_feats = pd.DataFrame(x_feats).fillna(0).to_numpy()

    return x_feats

def _handcraft_features(xyz, sample_rate):
    ''' Extract commonly used HAR time-series features. xyz is a window of shape (N,3) '''
    feats = {}

    v = np.linalg.norm(xyz, axis=1)
    v = v - 1  # detrend: "remove gravity"
    v = np.clip(v, -2, 2)  # clip abnormaly high values

    # Moments features
    feats.update(moments_features(v, sample_rate))

    # Quantile features
    feats.update(quantile_features(v, sample_rate))

    # Autocorrelation features
    feats.update(autocorr_features(v, sample_rate))

    # Spectral features
    feats.update(spectral_features(v, sample_rate))

    # FFT features
    feats.update(fft_features(v, sample_rate))

    # Peak features
    feats.update(peaks_features(v, sample_rate))

    return feats


def moments_features(v, sample_rate=None):
    """ Statistical moments """
    feats = {
        'avg': np.mean(v),
        'std': np.std(v),
        'skew': stats.skew(v),
        'kurt': stats.kurtosis(v),
    }
    return feats


def quantile_features(v, sample_rate=None):
    """ Quantiles (min, 25th, med, 75th, max) """
    feats = {}
    feats['min'], feats['q25'], feats['med'], feats['q75'], feats['max'] = np.quantile(v, (0, .25, .5, .75, 1))
    return feats


def autocorr_features(v, sample_rate):
    """ Autocorrelation (0.5, 1 and 2 seconds lag) """
    feats = {}
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        onesec = int(sample_rate)
        halfsec = int(sample_rate // 2)
        twosec = int(2 * sample_rate)
        feats['autocorr_halfsec'] = np.nan_to_num(np.corrcoef(v[:-halfsec], v[halfsec:]))[0, 1]
        feats['autocorr_onesec'] = np.nan_to_num(np.corrcoef(v[:-onesec], v[onesec:]))[0, 1]
        feats['autocorr_twosec'] = np.nan_to_num(np.corrcoef(v[:-twosec], v[twosec:]))[0, 1]
    return feats


def spectral_features(v, sample_rate):
    """ Spectral entropy, average power, dominant frequencies """

    feats = {}

    freqs, powers = signal.periodogram(v, fs=sample_rate, detrend='constant')

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        feats['pentropy'] = np.nan_to_num(stats.entropy(powers + 1e-16))

    feats['avgpow'] = np.mean(powers)

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]

    TOPN = 3
    feats = {}
    feats.update({f"f{i + 1}": 0 for i in range(TOPN)})
    feats.update({f"p{i + 1}": 0 for i in range(TOPN)})
    for i, j in enumerate(peak_ranks[:TOPN]):
        feats[f"f{i + 1}"] = peak_freqs[j]
        feats[f"p{i + 1}"] = peak_powers[j]

    return feats


def fft_features(v, sample_rate, nfreqs=5):
    """ Power of frequencies 0Hz, 1Hz, 2Hz, ... using Welch's method """

    _, powers = signal.welch(
        v, fs=sample_rate,
        nperseg=sample_rate,
        noverlap=sample_rate // 2,
        detrend='constant',
        average='median'
    )

    feats = {f"fft{i}": powers[i] for i in range(nfreqs + 1)}

    return feats


def peaks_features(v, sample_rate):
    """ Features of the signal peaks """

    feats = {}
    u = butterfilt(v, 5, fs=sample_rate)  # lowpass 5Hz
    peaks, peak_props = signal.find_peaks(u, distance=0.2 * sample_rate, prominence=0.25)
    feats['npeaks'] = len(peaks) / (len(v) / sample_rate)  # peaks/sec
    if len(peak_props['prominences']) > 0:
        feats['peaks_avg_promin'] = np.mean(peak_props['prominences'])
        feats['peaks_min_promin'] = np.min(peak_props['prominences'])
        feats['peaks_max_promin'] = np.max(peak_props['prominences'])
    else:
        feats['peaks_avg_promin'] = feats['peaks_min_promin'] = feats['peaks_max_promin'] = 0

    return feats
