"""
Train the following models with 5 fold cross validation:

Self-Supervised Net (SSLNet): The pretrained self-supervised model, fine-tuned on the training data.
SSL-HMM: HMM trained on the predictions of the validation fold of the fine-tuned SSLNet.
SSL-HMM-learn: HMM-learn trained on the predictions of the validation fold of the fine-tuned SSLNet.

Output for each fold (saved to disk, see config for paths):
- The fine-tuned SSLNet weights (cfg.sslnet.weights)
- A Numpy archive (.npz) with the SSL-HMM model matrices (cfg.hmm.weights)
- A Numpy archive (.npz) with the SSL-HMM-learn model matrices (cfg.hmmlearn.weights)
"""

import torch
import numpy as np

from omegaconf import OmegaConf

# own module imports
import utils.utils as utils
from utils.data import load_data
from train import train_model
from eval import evaluate_model, evaluate_folds
from prepare import prepare_data

log = utils.get_logger()

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = OmegaConf.load("conf/config.yaml")
    log.info(str(OmegaConf.to_yaml(cfg)))

    prepare_data(cfg)

    data_dict = load_data(cfg)

    for fold, fold_data in data_dict.items():        
        train_model(fold_data, cfg, str(fold))

        evaluate_model(fold_data, cfg, str(fold))

    evaluate_folds(cfg, folds=len(data_dict))    
