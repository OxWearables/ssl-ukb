import torch
import torch.nn as nn
import numpy as np
import logging
import os

from torch.utils.data import DataLoader
from scipy.special import softmax
from omegaconf import OmegaConf

# own module imports
import utils.utils as utils
from models.hmm import HMM
from utils.dataloader import NormalDataset, load_data

log = logging.getLogger('hmm')
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def main():
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
    sslnet: nn.Module = torch.hub.load(repo, 'harnet30', class_num=4, pretrained=True)
    sslnet.to(my_device)

    model_dict = torch.load(os.path.join(cfg.pretrained_model_root, 'state_dict.pt'), map_location=my_device)
    sslnet.load_state_dict(model_dict)

    # load raw data
    (
        x_train, y_train, group_train, time_train,
        x_val, y_val, group_val, time_val,
        x_test, y_test, group_test, time_test,
        le
    ) = load_data(cfg)

    # construct dataloader
    val_dataset = NormalDataset(x_val, y_val, pid=group_val, name="val", is_labelled=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=6,
    )

    # HMM training
    hmm = HMM(le.transform(le.classes_), uniform_prior=cfg.hmm.uniform_prior)

    log.info('Getting validation predictions')
    y_val, y_val_pred, pid_val = utils.mlp_predict(
        sslnet, val_loader, my_device, cfg, output_logits=True
    )

    # softmax logits
    y_val_pred_sf = softmax(y_val_pred, axis=1)

    log.info('Training HMM')
    hmm.train(y_val_pred_sf, y_val)
    hmm.save(cfg.hmm.path)

    log.info(hmm)
    log.info(le.classes_)
    log.info('HMM saved to %s', cfg.hmm.path)


if __name__ == "__main__":
    main()
