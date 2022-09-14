import torch
import torch.nn as nn
import numpy as np

import utils.utils as utils

log = utils.get_logger()


def get_sslnet(device, cfg, eval=True, load_weights=True, pretrained=False):
    """
    Load and return the SSLNet.

    :param str device: pytorch map location
    :param cfg: config object
    :param bool eval: Put the model in evaluation mode.
    :param bool load_weights: Load pretrained (fine-tuned) weights.
    :param bool pretrained: Initialise the model with self-supervised pretrained weights.
    :return: pytorch SSLNet model
    :rtype: nn.Module
    """

    if cfg.ssl_repo_path:
        # use repo from disk (for offline use)
        log.info('Using local %s', cfg.ssl_repo_path)
        sslnet: nn.Module = torch.hub.load(cfg.ssl_repo_path, 'harnet30', source='local', class_num=4,
                                           pretrained=pretrained)
    else:
        # download repo from github
        repo = 'OxWearables/ssl-wearables'
        sslnet: nn.Module = torch.hub.load(repo, 'harnet30', trust_repo=True, class_num=4, pretrained=pretrained)

    if load_weights:
        # load pretrained weights
        model_dict = torch.load(cfg.sslnet.weights, map_location=device)
        sslnet.load_state_dict(model_dict)
        log.info('Loaded SSLNet weights from %s', cfg.sslnet.weights)

    if eval:
        sslnet.eval()

    sslnet.to(device)

    sslnet.labels = utils.labels
    sslnet.classes = utils.classes

    return sslnet


def predict(model, data_loader, my_device, output_logits=False):
    """
    Iterate over the dataloader and do inference with a pytorch model.

    :param nn.Module model: pytorch Module
    :param data_loader: pytorch dataloader
    :param str my_device: pytorch map device
    :param bool output_logits: When True, output the raw outputs (logits) from the last layer (before classification).
                                When False, argmax the logits and output a classification scalar.
    :return: true labels, model predictions, pids
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """

    from tqdm import tqdm

    predictions_list = []
    true_list = []
    pid_list = []
    model.eval()
    if my_device == 'cpu':
        torch.set_flush_denormal(True)
    for i, (my_X, my_Y, my_PID) in enumerate(tqdm(data_loader)):
        with torch.inference_mode():
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
