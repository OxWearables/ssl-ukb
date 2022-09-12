import numpy as np
import pandas as pd

# Capture24 class labels
labels = np.array(['light', 'moderate-vigorous', 'sedentary', 'sleep'])
classes = np.array([0, 1, 2, 3])


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


def raw_to_df(data, labels, time, classes, label_proba=False, reindex=True, freq='30S'):
    """
    Construct a DataFrome from the raw data, prediction labels and time Numpy arrays.

    :param data: Numpy acc data, shape (rows, window_len, 3)
    :param labels: Either a scalar label array with shape (rows, ),
                    or the probabilities for each class if label_proba==True with shape (rows, len(classes)).
    :param time: Numpy time array, shape (rows, )
    :param classes: Array with the categorical class labels.
                    The index of this array should correspond to the labels value when label_proba==False.
    :param label_proba: If True, assume 'labels' contains the raw class probabilities.
    :param reindex: Reindex the dataframe to fill missing values
    :param freq: Reindex frequency
    :return: Dataframe
        Index: DatetimeIndex
        Columns: acc, classes
    :rtype: pd.DataFrame
    """
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


def ukb_df_to_series(df: pd.DataFrame, label_col: str):
    """
    Convert a dataframe generated by inference.py to timeseries format with one-hot encoded labels.
    Columns: index (time), acc, labels (onehot)

    :param df: input dataframe
    :param label_col: column name used for the labels. This will be either 'label' (SSLNET) or 'label_hmm' (SSLNET+HMM).
    :rtype: pd.DataFrame
    :return: The processed dataframe
    """
    df = df.copy()
    labels = pd.get_dummies(df[label_col], dummy_na=True)
    df[labels.columns] = labels
    df.loc[labels[np.nan].astype('boolean')] = np.nan
    df = df.drop(['label', 'label_hmm', np.nan], axis=1, errors='ignore')
    return df


def classification_scores(y_test, y_test_pred):
    import sklearn.metrics as metrics

    cohen_kappa = metrics.cohen_kappa_score(y_test, y_test_pred)
    precision = metrics.precision_score(
        y_test, y_test_pred, average="macro", zero_division=0
    )
    recall = metrics.recall_score(
        y_test, y_test_pred, average="macro", zero_division=0
    )
    f1 = metrics.f1_score(
        y_test, y_test_pred, average="macro", zero_division=0
    )

    return cohen_kappa, precision, recall, f1


def save_report(precision_list, recall_list, f1_list, cohen_kappa_list, report_path):
    log = get_logger()

    data = {
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "kappa": cohen_kappa_list,
    }

    df = pd.DataFrame(data)
    df.to_csv(report_path, index=False)

    log.info('Report saved to %s', report_path)

    return df


def classification_report(results, report_path):
    # Collate metrics
    cohen_kappa_list = [result[0] for result in results]
    precision_list = [result[1] for result in results]
    recall_list = [result[2] for result in results]
    f1_list = [result[3] for result in results]

    return save_report(
        precision_list, recall_list, f1_list, cohen_kappa_list, report_path
    )


def write_cluster_cmds(ukb_data_dir: str, output_file: str, group_file: str):
    """
    Write commands to a file for cluster processing in an array job.

    :param ukb_data_dir: Path to UKB accelerometer files
    :param output_file: Commands output file
    :param group_file: Extra output file with group;pid data
    """
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


def plot(data):
    """
    Simple wrapper function that takes a prediction dataframe and plots it. For data inspection and debugging.

    :param str | pd.DataFrame data: Path to parquet file, or the preloaded dataframe object.
    """
    from accelerometer.accPlot import plotTimeSeries

    if isinstance(data, str):
        df = pd.read_parquet(data)
    else:
        df = data

    df = ukb_df_to_series(df, 'label_hmm')

    fig = plotTimeSeries(df)
    fig.show()


def get_logger():
    """
    Return a shared logger for the package.
    """
    import logging

    log = logging.getLogger('ssl-ukb')
    log.setLevel(logging.DEBUG)

    if not log.hasHandlers():
        handler = logging.StreamHandler()
        fmt = logging.Formatter(fmt='%(asctime)s %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(fmt)
        log.addHandler(handler)

    return log
