import joblib
import torch
import torch.nn as nn
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader
from scipy.special import softmax
from omegaconf import OmegaConf
from accelerometer.accPlot import plotTimeSeries
from tqdm import tqdm
from joblib import Parallel, delayed
from imblearn.ensemble import BalancedRandomForestClassifier

# own module imports
import utils.utils as utils
from models.hmm import HMM
from utils.dataloader import NormalDataset, load_data

log = logging.getLogger('hmm')
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    cfg = OmegaConf.load("conf/config.yaml")

    np.random.seed(42)
    torch.manual_seed(42)
    log.info(str(OmegaConf.to_yaml(cfg)))

    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    elif cfg.multi_gpu is True:
        my_device = "cuda:0"  # use the first GPU as master
    else:
        my_device = "cpu"

    # load pretrained SSL model and weights
    repo = 'OxWearables/ssl-wearables'
    sslnet: nn.Module = torch.hub.load(repo, 'harnet30', class_num=4, pretrained=True)
    sslnet.to(my_device)

    model_dict = torch.load(os.path.join(cfg.pretrained_model_root, 'state_dict.pt'), map_location=my_device)
    sslnet.load_state_dict(model_dict)

    # load pretrained RF
    rf: BalancedRandomForestClassifier = joblib.load(os.path.join(cfg.pretrained_model_root, 'rf.joblib'))

    # load raw data
    (
        x_train, y_train, group_train, time_train,
        x_train_rf, y_train_rf, group_train_rf, time_train_rf,
        x_val, y_val, group_val, time_val,
        x_test, y_test, group_test, time_test,
        x_test_rf, y_test_rf, group_test_rf, time_test_rf,
        le
    ) = load_data(cfg)

    # load pretrained HMM
    hmm_ssl = HMM(le.transform(le.classes_), uniform_prior=cfg.hmm.uniform_prior)
    hmm_ssl.load(cfg.hmm.path_ssl)

    hmm_rf = HMM(le.transform(rf.classes_), uniform_prior=cfg.hmm.uniform_prior)
    hmm_rf.load(cfg.hmm.path_rf)

    # data loader
    test_dataset = NormalDataset(x_test, y_test, pid=group_test, name="test", is_labelled=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=6,
    )

    log.info('Get SSL test predictions')
    y_test, y_test_pred, pid_test = utils.mlp_predict(
        sslnet, test_loader, my_device, cfg, output_logits=False
    )

    log.info('Extract RF features')
    x_feats = Parallel(n_jobs=20, verbose=0)(
        delayed(utils.handcraft_features)(x, sample_rate=cfg.data.sample_rate) for x in tqdm(x_test_rf)
    )

    log.info('Get RF test predictions')
    x_feats = pd.DataFrame(x_feats).to_numpy()

    y_test_pred_rf = le.transform(rf.predict(x_feats))

    # HMM smoothed predictions
    log.info('Get HMM smoothed predictions')
    y_test_pred_hmm = hmm_ssl.viterbi(y_test_pred)
    y_test_pred_hmm_rf = hmm_rf.viterbi(y_test_pred_rf)

    # save performance scores for every single subject
    my_pids = np.unique(pid_test)

    def score(name, current_pid, pid, y, y_pred, y_pred_hmm):
        subject_filter = current_pid == pid
        subject_true = y[subject_filter]
        subject_pred = y_pred[subject_filter]
        subject_pred_hmm = y_pred_hmm[subject_filter]

        result = utils.classification_scores(subject_true, subject_pred)
        result_hmm = utils.classification_scores(subject_true, subject_pred_hmm)

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

        return result, result_hmm

    log.info('Process results')
    results, results_hmm = zip(*Parallel(n_jobs=8, verbose=0)(
        delayed(score)('SSL', current_pid, pid_test, y_test, y_test_pred, y_test_pred_hmm) for current_pid in
        tqdm(my_pids)
    ))

    results = np.array(results)
    results_hmm = np.array(results_hmm)

    results_rf, results_hmm_rf = zip(*Parallel(n_jobs=8, verbose=0)(
        delayed(score)('RF', current_pid, pid_test, y_test_rf, y_test_pred_rf, y_test_pred_hmm_rf) for current_pid in
        tqdm(my_pids)
    ))

    results_rf = np.array(results_rf)
    results_hmm_rf = np.array(results_hmm_rf)

    # save reports
    dfr = utils.classification_report(results, 'report_ssl.csv')
    dfr_hmm = utils.classification_report(results_hmm, 'report_ssl_hmm.csv')

    log.info('Results SSL: ')
    log.info(dfr.mean())

    log.info('\nResults SSL-HMM: ')
    log.info(dfr_hmm.mean())

    # save reports
    dfr_rf = utils.classification_report(results_rf, 'report_rf.csv')
    dfr_hmm_rf = utils.classification_report(results_hmm_rf, 'report_rf_hmm.csv')

    log.info('\nResults RF: ')
    log.info(dfr_rf.mean())

    log.info('\nResults RF-HMM: ')
    log.info(dfr_hmm_rf.mean())
