"""
Train the following models:

Random Forest (RF): Extract handcrafted features and train an RF. Joblib dump the whole RF for later use.
RF-HMM: Hidden Markov Model (HMM) trained with out-of-bag predictions of the RF.

Self-Supervised Net (SSLNet): The pretrained self-supervised model, fine-tuned on the training data.
SSL-HMM: HMM trained on the predictions of the validation fold of the fine-tuned SSLNet.

Output (saved to disk, see config for paths):
- A joblib dump pickle with the RF (cfg.rf.path)
- The fine-tuned SSLNet weights (cfg.sslnet.weights)
- A Numpy archive (.npz) with the RF-HMM and SSL-HMM model matrices (cfg.hmm.weights)
"""

import joblib
import torch
import numpy as np

from torch.utils.data import DataLoader
from scipy.special import softmax
from omegaconf import OmegaConf

# own module imports
import utils.utils as utils
import models.sslnet as ssl
import models.rf as rf
from models.hmm import HMM
from utils.data import NormalDataset, load_data, get_inverse_class_weights

log = utils.get_logger()

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = OmegaConf.load("conf/config.yaml")
    log.info(str(OmegaConf.to_yaml(cfg)))

    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    else:
        my_device = "cpu"

    # load raw data
    (
        x_train, y_train, group_train, time_train,
        x_train_rf, y_train_rf, group_train_rf, time_train_rf,
        x_val, y_val, group_val, time_val,
        x_test, y_test, group_test, time_test,
        x_test_rf, y_test_rf, group_test_rf, time_test_rf
    ) = load_data(cfg)

    # RF training
    rfmodel = rf.get_rf(num_workers=cfg.num_workers)

    log.info('Extract RF features')
    x_feats = rf.extract_features(x_train_rf, sample_rate=cfg.data.sample_rate, num_workers=cfg.num_workers)

    log.info('Training RF')
    rfmodel.fit(x_feats, y_train_rf)
    joblib.dump(rfmodel, cfg.rf.path)
    log.info('RF saved to %s', cfg.rf.path)

    # SSLNet training
    # load SSL model with self-supervised pre-trained weights
    sslnet = ssl.get_sslnet(my_device, cfg, load_weights=False, pretrained=True)

    if cfg.multi_gpu:
        sslnet = torch.nn.DataParallel(sslnet, output_device=my_device, device_ids=cfg.gpu_ids)

    # construct train and validation dataloaders
    train_dataset = NormalDataset(x_train, y_train, name="train", is_labelled=True, transform=cfg.sslnet.augmentation)
    val_dataset = NormalDataset(x_val, y_val, name="val", is_labelled=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1000,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )

    log.info('SSLNet training')
    ssl.train(sslnet, train_loader, val_loader, cfg, my_device, get_inverse_class_weights(y_train))

    # load trained SSLNet weights (best weights prior to early-stopping)
    model_dict = torch.load(cfg.sslnet.weights, map_location=my_device)

    if cfg.multi_gpu:
        sslnet.module.load_state_dict(model_dict)
    else:
        sslnet.load_state_dict(model_dict)

    # HMM training (SSL)
    log.info('Training SSL-HMM')
    log.info('Getting SSLNet validation predictions')
    y_val, y_val_pred, pid_val = ssl.predict(
        sslnet, val_loader, my_device, output_logits=True
    )

    # softmax logits
    y_val_pred_sf = softmax(y_val_pred, axis=1)

    hmm_ssl = HMM(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
    hmm_ssl.train(y_val_pred_sf, y_val)
    hmm_ssl.save(cfg.hmm.weights_ssl)

    log.info(hmm_ssl)
    log.info('SSL-HMM saved to %s', cfg.hmm.weights_ssl)

    # HMM training (RF)
    log.info('Training RF-HMM')
    hmm_rf = HMM(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
    hmm_rf.train(rfmodel.oob_decision_function_, y_train_rf)
    hmm_rf.save(cfg.hmm.weights_rf)

    log.info(hmm_rf)
    log.info('RF-HMM saved to %s', cfg.hmm.weights_rf)
