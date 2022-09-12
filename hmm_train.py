"""
Train 2 HMM models.

RF-HMM: Trained with out-of-bag predictions of a RF with handcrafted features.
        Will extract features and train the RF from scratch, and joblib dump the trained RF model for later use.

SSL-HMM: Trained on the predictions of the validation fold of a pretrained SSLNet.

Output:
A Numpy archive (.npz) with the HMM model matrices (see models.HMM class)
"""

import os
import joblib
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from scipy.special import softmax
from omegaconf import OmegaConf

# own module imports
import utils.utils as utils
import models.sslnet as ssl
import models.rf as rf
from models.hmm import HMM
from utils.dataloader import NormalDataset, load_data

log = utils.get_logger()

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = OmegaConf.load("conf/config.yaml")
    log.info(str(OmegaConf.to_yaml(cfg)))

    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    elif cfg.multi_gpu is True:
        my_device = "cuda:0"  # use the first GPU as master
    else:
        my_device = "cpu"

    # load raw data
    (
        x_train, y_train, group_train, time_train,
        x_train_rf, y_train_rf, group_train_rf, time_train_rf,
        x_val, y_val, group_val, time_val,
        x_test, y_test, group_test, time_test,
        x_test_rf, y_test_rf, group_test_rf, time_test_rf,
        le
    ) = load_data(cfg)

    # train RF
    rfmodel = rf.get_rf(num_workers=cfg.num_workers)

    log.info('Extract RF features')
    x_feats = rf.extract_features(x_train_rf, sample_rate=cfg.data.sample_rate, num_workers=cfg.num_workers)

    log.info('Training RF')
    rfmodel.fit(x_feats, y_train_rf)
    joblib.dump(rfmodel, cfg.rf.path)
    log.info('RF saved to %s', cfg.rf.path)

    # load pretrained SSL model and weights
    sslnet = ssl.get_sslnet(my_device, cfg, eval=True, load_weights=True)

    # construct dataloader for SSL inference
    val_dataset = NormalDataset(x_val, y_val, pid=group_val, name="val", is_labelled=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )

    # HMM training (SSL)
    log.info('Training SSL-HMM')
    log.info('Getting SSLNet validation predictions')
    y_val, y_val_pred, pid_val = ssl.predict(
        sslnet, val_loader, my_device, output_logits=True
    )

    # softmax logits
    y_val_pred_sf = softmax(y_val_pred, axis=1)

    hmm_ssl = HMM(le.transform(le.classes_), uniform_prior=cfg.hmm.uniform_prior)
    hmm_ssl.train(y_val_pred_sf, y_val)
    hmm_ssl.save(cfg.hmm.weights_ssl)

    log.info(hmm_ssl)
    log.info(le.classes_)
    log.info('SSL-HMM saved to %s', cfg.hmm.weights_ssl)

    # HMM training (RF)
    log.info('Training RF-HMM')
    hmm_rf = HMM(rfmodel.classes_, uniform_prior=cfg.hmm.uniform_prior)
    hmm_rf.train(rfmodel.oob_decision_function_, y_train_rf)
    hmm_rf.save(cfg.hmm.weights_rf)

    log.info(hmm_rf)
    log.info(rfmodel.classes_)
    log.info('RF-HMM saved to %s', cfg.hmm.weights_rf)
