import joblib
from imblearn.ensemble import BalancedRandomForestClassifier
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from models.hmm import HMM
from models.peak_count import PeakCounter
import models.sslnet as ssl
import models.rf as rf
from utils.data import NormalDataset
from utils.utils import make_windows


class WalkingDetector():
    def __init__(
        self,
        walking_model,
        model_type,
        hmm = None,
        winsec = None, 
        sample_rate = None, 
        num_workers = None, 
        batch_size = None,
        device = None,
        cfg = None
    ):
        self.model_type = model_type
        self.device = device
        self.load_walking_model(walking_model, cfg)
        self.load_hmm(hmm)
        self.winsec = winsec
        self.sample_rate = sample_rate
        self.num_workers = num_workers
        self.batch_size = batch_size

    def load_walking_model(self, model, cfg):
        if isinstance(model, str):
            if self.model_type == 'rf':
                self.walking_model: BalancedRandomForestClassifier = joblib.load(model)

            elif self.model_type == 'ssl':
                self.walking_model: nn.Module = ssl.get_sslnet(self.device, cfg, model)
            
            else:
                raise Exception("Invalid walking model type")

        else:
            self.walking_model = model

    def load_hmm(self, hmm):
        if hmm:
            if isinstance(hmm, str):
                self.hmm = HMM()
                self.hmm.load(hmm)
        
            elif isinstance(hmm, HMM):
                self.hmm = hmm
            
            else:
                raise Exception("Invalid HMM")
        
        else:
            self.hmm = None
    
    def predict(self, X, T=None):
        if self.model_type == 'rf':
            x_feats = rf.extract_features(X, self.sample_rate, self.num_workers)
            y = self.walking_model.predict(x_feats)
        
        elif self.model_type == 'ssl':
            data_loader = DataLoader(
                NormalDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

            _, y, _ = ssl.predict(
                self.walking_model, data_loader, self.device, output_logits=False
            )

        else:
            raise Exception("Invalid walking model type")
        
        if self.hmm is not None:
            y = self.hmm.predict(y, T, self.winsec)

        return y


class StepCounter():
    def __init__(
        self,
        peak_counter,
        walking_detector,
        walking_model_type,
        hmm = None,
        winsec=10,
        sample_rate=30,
        **kwargs
    ):
        self.winsec = winsec
        self.sample_rate = sample_rate
        self.load_peak_counter(peak_counter)
        self.walking_detector = WalkingDetector(walking_detector, walking_model_type, 
                                                hmm, winsec, sample_rate, **kwargs)
        

    def load_peak_counter(self, peak_counter):
        if isinstance(peak_counter, str):
            self.peak_counter = PeakCounter(self.winsec, self.sample_rate)
            self.peak_counter.load(peak_counter)
        
        elif isinstance(peak_counter, PeakCounter):
            self.peak_counter = peak_counter

        else:
            raise Exception("Invalid peak counter")

    def predict(self, X, T=None, return_walk=False):
        # check X quality
        whr_ok = ~(np.asarray([np.isnan(x).any() for x in X]))

        X_ = X[whr_ok]
        T_ = T[whr_ok] if T is not None else None
        W_ = self.walking_detector.predict(X_, T_).astype('bool')
        Y_ = np.zeros_like(W_, dtype='float')
        Y_[W_] = self.peak_counter.predict(X_[W_], return_sum=False)

        Y = np.full(len(X), fill_value=np.nan)
        Y[whr_ok] = Y_

        if return_walk:
            W = np.full(len(X), fill_value=np.nan)
            W[whr_ok] = W_
            return Y, W
        
        return Y

    def predict_from_frame(self, data, return_sum=False):

        def fn(chunk):
            """ Process the chunk. Apply padding if length is not enough. """
            n = len(chunk)
            window_len = self.sample_rate*self.winsec
            x = chunk[['x', 'y', 'z']].to_numpy()
            if n > window_len:
                x = x[:window_len]
            if n < window_len:
                m = window_len - n
                x = np.pad(x, ((0, m), (0, 0)), mode='wrap')
            return x

        X, T = make_windows(data, self.winsec, fn=fn, return_index=True)
        X = np.asarray(X)
        Y = self.predict(X, T)
        Y = pd.Series(Y, index=T)

        if return_sum:
            return Y.sum()
        else:
            return Y

