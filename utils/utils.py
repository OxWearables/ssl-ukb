import numpy as np
import pandas as pd
import torch
import math
from tqdm import tqdm
import sklearn.metrics as metrics
from scipy.interpolate import interp1d
from torch.autograd import Variable


def resize(x, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """

    length_orig = x.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    x = interp1d(t_orig, x, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )
    return x


def raw_to_df(data, labels, time, classes):
    label_matrix = np.zeros((len(time), len(classes)))
    a_matrix = np.zeros(len(time))

    for i, data in enumerate(data):
        label = labels[i]
        label_matrix[i, label] = 1

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        a = math.fabs(math.sqrt(np.mean(x) ** 2 + np.mean(y) ** 2 + np.mean(z) ** 2) - 1) * 500
        a_matrix[i] = a

    datadict = {
        'time': time,
        'acc': a_matrix,
        classes[0]: label_matrix[:, 0],
        classes[1]: label_matrix[:, 1],
        classes[2]: label_matrix[:, 2],
        classes[3]: label_matrix[:, 3],
    }
    df = pd.DataFrame(datadict)
    df.set_index('time', inplace=True)
    df = df.tz_localize('Europe/London')
    newindex = pd.date_range(df.index[0], df.index[-1], freq='30S')
    df = df.reindex(newindex)

    return df.copy()


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

            # true_y = my_Y.to(my_device, dtype=torch.long)
            logits = model(my_X)
            pred_y = torch.argmax(logits, dim=1)

            # true_list.append(true_y.cpu())
            true_list.append(my_Y)
            if output_logits:
                predictions_list.append(logits.cpu())
            else:
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


def classification_scores(Y_test, Y_test_pred):
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
