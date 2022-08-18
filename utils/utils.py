import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.autograd import Variable


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


def raw_to_df(data, labels, time, classes, label_proba=False, freq='30S', reindex=True):
    label_matrix = np.zeros((len(time), len(classes)), dtype=np.float32)
    a_matrix = np.zeros(len(time), dtype=np.float32)

    for i, data in enumerate(data):
        if not label_proba:
            label = labels[i]
            label_matrix[i, label] = 1

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        # a = (np.sqrt(np.mean(np.abs(x)) ** 2 + np.mean(np.abs(y)) ** 2 + np.mean(np.abs(z)) ** 2) - 0.5) * 1000
        a = (np.sqrt(np.mean(x ** 2) + np.mean(y ** 2) + np.mean(z ** 2)) - 1) * 1000
        a_matrix[i] = a

    if label_proba:
        datadict = {
            'time': time,
            'acc': a_matrix,
            classes[0]: labels[:, 0],
            classes[1]: labels[:, 1],
            classes[2]: labels[:, 2],
            classes[3]: labels[:, 3]
        }
    else:
        datadict = {
            'time': time,
            'acc': a_matrix,
            classes[0]: label_matrix[:, 0],
            classes[1]: label_matrix[:, 1],
            classes[2]: label_matrix[:, 2],
            classes[3]: label_matrix[:, 3],
        }
    df = pd.DataFrame(datadict)
    df = df.set_index('time')
    # df = df.tz_localize('Europe/London', ambiguous='NaT', nonexistent='NaT')
    if reindex:
        newindex = pd.date_range(df.index[0], df.index[-1], freq=freq)
        df = df.reindex(newindex)

    return df


def ukb_df_to_accplot(data, label_col):
    df = data.copy()
    dtype = pd.CategoricalDtype(categories=['light', 'moderate-vigorous', 'sedentary', 'sleep'])
    labels = pd.get_dummies(df[label_col].astype(dtype))
    df[labels.columns] = labels[labels.columns]
    df = df.drop(['label', 'label_hmm'], axis=1)
    return df


def mlp_predict(model, data_loader, my_device, cfg, output_logits=False):
    predictions_list = []
    true_list = []
    pid_list = []
    model.eval()
    if my_device == 'cpu':
        torch.set_flush_denormal(True)
    for i, (my_X, my_Y, my_PID) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            logits = model(my_X)
            true_list.append(my_Y)
            if output_logits:
                predictions_list.append(logits.cpu())
            else:
                pred_y = torch.argmax(logits, dim=1)
                predictions_list.append(pred_y.cpu())
            pid_list.extend(my_PID)
    true_list = torch.cat(true_list)
    predictions_list = torch.cat(predictions_list)

    if output_logits:
        return (
            torch.flatten(true_list).numpy(),
            predictions_list.numpy(),
            np.array(pid_list),
        )
    else:
        return (
            torch.flatten(true_list).numpy(),
            torch.flatten(predictions_list).numpy(),
            np.array(pid_list),
        )


def handcraft_features(xyz, sample_rate):
    import scipy.stats as stats
    import scipy.signal as signal
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


def classification_scores(Y_test, Y_test_pred):
    import sklearn.metrics as metrics

    cohen_kappa = metrics.cohen_kappa_score(Y_test, Y_test_pred)
    precision = metrics.precision_score(
        Y_test, Y_test_pred, average="macro", zero_division=0
    )
    recall = metrics.recall_score(
        Y_test, Y_test_pred, average="macro", zero_division=0
    )
    f1 = metrics.f1_score(
        Y_test, Y_test_pred, average="macro", zero_division=0
    )

    return cohen_kappa, precision, recall, f1


def save_report(
    precision_list, recall_list, f1_list, cohen_kappa_list, report_path
):
    data = {
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "kappa": cohen_kappa_list,
    }

    df = pd.DataFrame(data)
    df.to_csv(report_path, index=False)

    return df


def classification_report(results, report_path):
    # logger is a tf logger
    # Collate metrics
    cohen_kappa_list = [result[0] for result in results]
    precision_list = [result[1] for result in results]
    recall_list = [result[2] for result in results]
    f1_list = [result[3] for result in results]

    return save_report(
        precision_list, recall_list, f1_list, cohen_kappa_list, report_path
    )


def write_cluster_cmds(ukb_data_dir: str, output_file: str, group_file: str):
    import os
    from glob import glob
    from pathlib import Path

    cmd = 'python inference.py {input}'

    with open(group_file, 'w') as g:
        with open(output_file, 'w') as f:
            files = glob(os.path.join(ukb_data_dir, '**/*.cwa.gz'))
            for file in files:
                f.write(cmd.format(input=file) + '\n')

                path = Path(file)
                group = path.parent.name
                subject = path.stem.replace('.cwa', '').split('_')[0]

                g.write('{g};{s}'.format(g=group, s=subject) + '\n')
