import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

import utils.utils as utils

log = utils.get_logger()


class EarlyStopping:
    """Early stops the training if validation loss
    doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time v
                            alidation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            msg = "Validation loss decreased"
            msg = msg + f" ({self.val_loss_min:.6f} --> {val_loss:.6f}). "
            msg = msg + "Saving model ..."
            self.trace_func(msg)
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_sslnet(device, ssl_repo_path=None, ssl_weights_path=None, pretrained=False):
    """
    Load and return the SSLNet.

    :param str device: pytorch map location
    :param ssl_repo_path: the path of downloaded resnet model
    :param ssl_weights_path: the path of the pretrained (fine-tuned) weights.
    :param bool pretrained: Initialise the model with self-supervised pretrained weights.
    :return: pytorch SSLNet model
    :rtype: nn.Module
    """

    if ssl_repo_path:
        # use repo from disk (for offline use)
        log.info('Using local %s', ssl_repo_path)
        sslnet: nn.Module = torch.hub.load(ssl_repo_path, 'harnet10', source='local', class_num=2,
                                           pretrained=pretrained)
    else:
        # download repo from github
        repo = 'OxWearables/ssl-wearables'
        sslnet: nn.Module = torch.hub.load(repo, 'harnet10', trust_repo=True, 
                                           class_num=2, pretrained=pretrained)

    if ssl_weights_path:
        # load pretrained weights
        model_dict = torch.load(ssl_weights_path, map_location=device)
        sslnet.load_state_dict(model_dict)
        log.info('Loaded SSLNet weights from %s', ssl_weights_path)

    sslnet.to(device)

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
    for i, (x, y, pid) in enumerate(tqdm(data_loader)):
        with torch.inference_mode():
            x = x.to(my_device, dtype=torch.float)
            logits = model(x)
            true_list.append(y)
            if output_logits:
                predictions_list.append(logits.cpu())
            else:
                pred_y = torch.argmax(logits, dim=1)
                predictions_list.append(pred_y.cpu())
            pid_list.extend(pid)
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


def train(model, train_loader, val_loader, cfg, my_device, weights, 
            weights_path):
    """
    Iterate over the training dataloader and train a pytorch model.
    After each epoch, validate model and early stop when validation loss function bottoms out.

    Trained model weights will be saved to disk (cfg.sslnet.weights).

    :param nn.Module model: pytorch model
    :param train_loader: training data loader
    :param val_loader: validation data loader
    :param cfg: config object.
    :param str my_device: pytorch map device.
    :param weights: training class weights
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.sslnet.learning_rate, amsgrad=True
    )

    if cfg.sslnet.weighted_loss_fn:
        weights = torch.FloatTensor(weights).to(my_device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(
        patience=cfg.sslnet.patience, path=weights_path, verbose=True, trace_func=log.info
    )

    for epoch in range(cfg.sslnet.num_epoch):
        model.train()
        train_losses = []
        train_acces = []
        for i, (x, y, _) in enumerate(tqdm(train_loader)):
            x.requires_grad_(True)
            x = x.to(my_device, dtype=torch.float)
            true_y = y.to(my_device, dtype=torch.long)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, true_y)
            loss.backward()
            optimizer.step()

            pred_y = torch.argmax(logits, dim=1)
            train_acc = torch.sum(pred_y == true_y)
            train_acc = train_acc / (pred_y.size()[0])

            train_losses.append(loss.cpu().detach())
            train_acces.append(train_acc.cpu().detach())

        val_loss, val_acc = _validate_model(model, val_loader, my_device, loss_fn)

        epoch_len = len(str(cfg.sslnet.num_epoch))
        print_msg = (
            f"[{epoch:>{epoch_len}}/{cfg.sslnet.num_epoch:>{epoch_len}}] | "
            + f"train_loss: {np.mean(train_losses):.3f} | "
            + f"train_acc: {np.mean(train_acces):.3f} | "
            + f"val_loss: {val_loss:.3f} | "
            + f"val_acc: {val_acc:.2f}"
        )

        early_stopping(val_loss, model)
        log.info(print_msg)

        if early_stopping.early_stop:
            log.info('Early stopping')
            log.info('SSLNet weights saved to %s', weights_path)
            break

    return model


def _validate_model(model, val_loader, my_device, loss_fn):
    """ Iterate over a validation data loader and return mean model loss and accuracy. """
    model.eval()
    losses = []
    acces = []
    for i, (x, y, _) in enumerate(val_loader):
        with torch.inference_mode():
            x = x.to(my_device, dtype=torch.float)
            true_y = y.to(my_device, dtype=torch.long)

            logits = model(x)
            loss = loss_fn(logits, true_y)

            pred_y = torch.argmax(logits, dim=1)

            val_acc = torch.sum(pred_y == true_y)
            val_acc = val_acc / (list(pred_y.size())[0])

            losses.append(loss.cpu().detach())
            acces.append(val_acc.cpu().detach())
    losses = np.array(losses)
    acces = np.array(acces)
    return np.mean(losses), np.mean(acces)
