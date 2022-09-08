"""
Train 2 HMM models.

RF-HMM: Trained with out-of-bag predictions of a RF with handcrafted features.
        Will extract features and train the RF, and joblib dump the full RF model for later use.

SSL-HMM: Trained on the predictions of the validation fold of the SSL. Will use the pretrained SSL.

Output:
A Numpy archive (.npz) with the HMM model matrices (see models.HMM class)
"""

import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from scipy.special import softmax
from omegaconf import OmegaConf
from joblib import Parallel, delayed
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier

# own module imports
import utils.utils as utils
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

    # load pretrained SSL model and weights
    repo = 'OxWearables/ssl-wearables'
    sslnet: nn.Module = torch.hub.load(repo, 'harnet30', class_num=4, pretrained=False)

    model_dict = torch.load(os.path.join(cfg.pretrained_model_root, 'state_dict.pt'), map_location=my_device)
    sslnet.load_state_dict(model_dict)
    sslnet.eval()
    sslnet.to(my_device)

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
    rf = BalancedRandomForestClassifier(
        n_estimators=3000,
        replacement=True,
        sampling_strategy="not minority",
        n_jobs=cfg.num_workers,
        random_state=42,
        oob_score=True
    )

    log.info('Extract RF features')
    x_feats = Parallel(n_jobs=cfg.num_workers)(
        delayed(utils.handcraft_features)(x, sample_rate=cfg.data.sample_rate) for x in tqdm(x_train_rf)
    )
    x_feats = pd.DataFrame(x_feats).to_numpy()

    log.info('Training RF')
    rf.fit(x_feats, y_train_rf)
    joblib.dump(rf, cfg.rf.path)

    # construct dataloader for SSL inference
    val_dataset = NormalDataset(x_val, y_val, pid=group_val, name="val", is_labelled=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=6,
    )

    # HMM training (SSL)
    log.info('Training SSL-HMM')
    log.info('Getting SSLNET validation predictions')
    y_val, y_val_pred, pid_val = utils.mlp_predict(
        sslnet, val_loader, my_device, cfg, output_logits=True
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
    hmm_rf = HMM(rf.classes_, uniform_prior=cfg.hmm.uniform_prior)
    hmm_rf.train(rf.oob_decision_function_, y_train_rf)
    hmm_rf.save(cfg.hmm.weights_rf)

    log.info(hmm_rf)
    log.info(rf.classes_)
    log.info('RF-HMM saved to %s', cfg.hmm.weights_rf)
