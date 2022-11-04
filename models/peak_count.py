import numpy as np
import os
from models.utils import butterfilt
from scipy.signal import find_peaks
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error


class PeakCounter():
    def __init__(
        self,
        window_sec = 10,
        sample_rate = 30
    ):
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.peak_params = None
    
    def save(self, path):
        """
        Save model parameters to a Numpy npz file.

        :param str path: npz file location
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.peak_params)

    def load(self, path):
        """
        Load model parameters from a Numpy npz file.

        :param str path: npz file location
        """
        d = np.load(path, allow_pickle=True)
        self.peak_params = d.item()

    def train(self, X, y_walk, y_steps, groups):
        """
        Train model parameters.

        :param X: Accelerometry signal in windows
        :param y_walk: Predicted walking labels
        :param y_steps: True steps labels
        :param groups: Particpant labels
        """
        def mae(x):
            step_count_true, step_count_pred = \
                zip(*filter(None, (count_participant_peaks(X, y_walk, y_steps, groups, pid, 
                                                           self.sample_rate, to_params(x)) 
                                    for pid in np.unique(groups))))
            return mean_absolute_error(step_count_true, step_count_pred)

        def to_params(x):
            params = {
                "distance": x[0],
                "max_width": x[1],
                "prominence": x[2],
                "lowpass_hz": x[3]
            }
            return params
                    
        res = minimize(
            mae,
            x0=[.5, .5, .5, 4],
            bounds=[
                (.2, 2),  # 0.2s to 2s (4Hz - 0.5Hz)
                (.01, 1), # 10ms to 1s
                (.1, 1),  # .1g to 1g
                (3, 5),   # 3Hz to 5Hz
            ],
            method='Nelder-Mead'
        )

        self.peak_params = to_params(res.x)
    
    def predict(self, X, y_walk):
        return batch_count_peaks(X[y_walk==1], self.sample_rate, self.peak_params).sum()


def count_participant_peaks(X, y_walk, y_steps, groups, pid, sample_rate, params):
    subject_x = X[groups==pid]
    subject_walk = y_walk[groups==pid]
    subject_step_count = y_steps[groups==pid].sum()
    if not np.isnan(subject_step_count):
        predicted_step_count = batch_count_peaks(subject_x[subject_walk==1], 
                                                 sample_rate,
                                                 params).sum()
        return subject_step_count, predicted_step_count 


def batch_count_peaks(X, sample_rate, params):
    """ Count number of peaks for an array of signals """
    V = toV(X, sample_rate, params["lowpass_hz"])
    return batch_count_peaks_from_V(V, sample_rate, params)


def batch_count_peaks_from_V(V, sample_rate, params):
    """ Count number of peaks for an array of signals """
    Y = np.asarray([
        len(find_peaks(
            v,
            distance=params["distance"] * sample_rate,
            prominence=params["prominence"],
            width=(1, params["max_width"] * sample_rate)
        )[0]) for v in V
    ])
    return Y


def toV(x, sample_rate, lowpass_hz):
    V = np.linalg.norm(x, axis=-1)
    V = V - 1
    V = np.clip(V, -2, 2)
    V = butterfilt(V, lowpass_hz, sample_rate, axis=-1)
    return V

