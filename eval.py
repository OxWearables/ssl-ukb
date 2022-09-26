"""
Evaluate the following models: RF, SSL, RF+HMM, SSL+HMM
Requires the pretrained RF, SSL and HMMs (trained with train.py)

Output:
- A report in .csv for each model with per-subject classification performance.
- Per-subject time series plots for RF+HMM and SSL+HMM in the 'plots' folder
- Confusion matrix plots in the 'plots' folder
"""

import joblib
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
import seaborn as sns

from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from accelerometer.accPlot import plotTimeSeries
from tqdm import tqdm
from joblib import Parallel, delayed
from imblearn.ensemble import BalancedRandomForestClassifier

# own module imports
import utils.utils as utils
import models.sslnet as ssl
import models.rf as rf
from models.hmm import HMM
from utils.data import NormalDataset, load_data

log = utils.get_logger()

if __name__ == '__main__':
    cfg = OmegaConf.load("conf/config.yaml")

    np.random.seed(42)
    torch.manual_seed(42)
    log.info(str(OmegaConf.to_yaml(cfg)))

    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    else:
        my_device = "cpu"

    # load pretrained SSL model
    sslnet = ssl.get_sslnet(my_device, cfg, load_weights=True)
    hmm_ssl = HMM(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
    hmm_ssl.load(cfg.hmm.weights_ssl)

    if cfg.rf.enabled:
        # load pretrained RF
        rfmodel: BalancedRandomForestClassifier = joblib.load(cfg.rf.path)
        log.info('Loaded RF from %s', cfg.rf.path)
        hmm_rf = HMM(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
        hmm_rf.load(cfg.hmm.weights_rf)

    # load raw data
    (
        _, _, _, _,
        _, _, _, _,
        x_test, y_test, group_test, time_test,
    ) = load_data(cfg)

    le = utils.le  # label encoder

    # SSL data loader
    test_dataset = NormalDataset(x_test, y_test, pid=group_test, name="test", is_labelled=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )

    # get test predictions
    log.info('Get SSL test predictions')
    y_test, y_test_pred, pid_test = ssl.predict(
        sslnet, test_loader, my_device, output_logits=False
    )
    y_test_pred_hmm = hmm_ssl.viterbi(y_test_pred)

    if cfg.rf.enabled:
        log.info('Extract RF features')
        x_feats = rf.extract_features(x_test, sample_rate=cfg.data.sample_rate, num_workers=cfg.num_workers)

        log.info('Get RF test predictions')
        y_test_pred_rf = rfmodel.predict(x_feats)
        y_test_pred_hmm_rf = hmm_rf.viterbi(y_test_pred_rf)

    # save performance scores and plots for every single subject
    my_pids = np.unique(pid_test)

    def score(name, current_pid, pid, y, y_pred, y_pred_hmm):
        subject_filter = current_pid == pid
        subject_true = y[subject_filter]
        subject_pred = y_pred[subject_filter]
        subject_pred_hmm = y_pred_hmm[subject_filter]

        result = utils.classification_scores(subject_true, subject_pred)
        result_hmm = utils.classification_scores(subject_true, subject_pred_hmm)

        cmatrix = metrics.confusion_matrix(subject_true, subject_pred, labels=utils.classes)
        cmatrix_hmm = metrics.confusion_matrix(subject_true, subject_pred_hmm, labels=utils.classes)

        # plot subject predictions
        df_true = utils.raw_to_df(x_test[subject_filter], subject_true, time_test[subject_filter], le.classes_)
        df_pred = utils.raw_to_df(x_test[subject_filter], subject_pred, time_test[subject_filter], le.classes_)
        df_pred_hmm = utils.raw_to_df(x_test[subject_filter], subject_pred_hmm, time_test[subject_filter], le.classes_)

        fig = plotTimeSeries(df_true)
        fig.savefig('plots/{pid}_true.png'.format(pid=current_pid), dpi=200)
        plt.close()

        fig = plotTimeSeries(df_pred)
        fig.savefig('plots/{pid}_{model}_pred.png'.format(pid=current_pid, model=name), dpi=200)
        plt.close()

        fig = plotTimeSeries(df_pred_hmm)
        fig.savefig('plots/{pid}_{model}_pred_hmm.png'.format(pid=current_pid, model=name), dpi=200)
        plt.close()

        return result, result_hmm, cmatrix, cmatrix_hmm

    log.info('Process results')
    # Use joblib lazy parallel cause plotting is slow
    # SSL results
    results, results_hmm, cmatrix, cmatrix_hmm = zip(*Parallel(n_jobs=cfg.num_workers)(
        delayed(score)('SSL', current_pid, pid_test, y_test, y_test_pred, y_test_pred_hmm)
        for current_pid in tqdm(my_pids)
    ))

    results = np.array(results)
    results_hmm = np.array(results_hmm)

    cmatrix = pd.DataFrame(np.sum(cmatrix, axis=0), index=le.classes_, columns=le.classes_)
    cmatrix_hmm = pd.DataFrame(np.sum(cmatrix_hmm, axis=0), index=le.classes_, columns=le.classes_)

    if cfg.rf.enabled:
        # RF results
        results_rf, results_hmm_rf, cmatrix_rf, cmatrix_hmm_rf = zip(*Parallel(n_jobs=cfg.num_workers)(
            delayed(score)('RF', current_pid, pid_test, y_test, y_test_pred_rf, y_test_pred_hmm_rf)
            for current_pid in tqdm(my_pids)
        ))

        results_rf = np.array(results_rf)
        results_hmm_rf = np.array(results_hmm_rf)

        cmatrix_rf = pd.DataFrame(np.sum(cmatrix_rf, axis=0), index=le.classes_, columns=le.classes_)
        cmatrix_hmm_rf = pd.DataFrame(np.sum(cmatrix_hmm_rf, axis=0), index=le.classes_, columns=le.classes_)

        # confusion matrix plots
        plots = {
            'matrix_ssl': cmatrix,
            'matrix_ssl_hmm': cmatrix_hmm,
            'matrix_rf': cmatrix_rf,
            'matrix_rf_hmm': cmatrix_hmm_rf,
        }
    else:
        plots = {
            'matrix_ssl': cmatrix,
            'matrix_ssl_hmm': cmatrix_hmm
        }

    log.info('Class list: \n %s', le.classes_)

    for title in plots:
        matrix: pd.DataFrame = plots[title]
        matrix = matrix.div(matrix.sum(axis=1), axis=0).round(2)  # normalise
        plt.figure()
        sns.heatmap(matrix, annot=True, vmin=0, vmax=1)
        plt.title(title)
        plt.xticks(rotation=40, ha='right')
        # plt.ylabel('true', rotation=0), plt.xlabel('predicted')
        plt.tight_layout()
        plt.savefig('plots/{title}.png'.format(title=title), dpi=200)
        plt.close()

        log.info('Confusion %s\n%s\n', title, matrix)

    # save SSL reports
    dfr = utils.classification_report(results, os.path.join(cfg.ukb_output_path, 'report_ssl.csv'))
    dfr_hmm = utils.classification_report(results_hmm, os.path.join(cfg.ukb_output_path, 'report_ssl_hmm.csv'))

    log.info('Results SSL:\n%s', dfr.mean())

    log.info('Results SSL-HMM:\n%s\n', dfr_hmm.mean())

    if cfg.rf.enabled:
        # save RF reports
        dfr_rf = utils.classification_report(results_rf, os.path.join(cfg.ukb_output_path, 'report_rf.csv'))
        dfr_hmm_rf = utils.classification_report(results_hmm_rf, os.path.join(cfg.ukb_output_path, 'report_rf_hmm.csv'))

        log.info('Results RF:\n%s', dfr_rf.mean())

        log.info('Results RF-HMM:\n%s', dfr_hmm_rf.mean())
