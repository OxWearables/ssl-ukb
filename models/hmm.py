import numpy as np
import os
from models.utils import check_for_time_values_error, \
    restore_labels_after_gaps, calculate_transition_matrix

class HMM:
    """
    Implement a basic HMM model with parameter saving/loading.
    """

    def __init__(self, labels=None, uniform_prior=True):
        self.prior = None
        self.emission = None
        self.transition = None
        self.labels = labels
        self.uniform_prior = uniform_prior

    def __str__(self):
        return "prior: {prior}\n" \
               "emission: {emission}\n" \
               "transition: {transition}\n" \
               "labels: {labels}".format(prior=self.prior, emission=self.emission,
                                         transition=self.transition, labels=self.labels)

    def train(self, y_prob, y_true, t=None, interval=None):
        """ https://en.wikipedia.org/wiki/Hidden_Markov_model
        :param y_prob: Observation probabilities
        :param y_true: Ground truth labels
        """

        if self.labels is None:
            self.labels = np.unique(y_true)

        prior = np.mean(y_true.reshape(-1, 1) == self.labels, axis=0)

        emission = np.vstack(
            [np.mean(y_prob[y_true == label], axis=0) for label in self.labels]
        )

        transition = calculate_transition_matrix(y_true, t, interval)

        self.prior = prior
        self.emission = emission
        self.transition = transition

    def predict(self, y_obs, t=None, interval=None, uniform_prior=None):        
        check_for_time_values_error(y_obs, t, interval)

        y_smooth = self.viterbi(y_obs, uniform_prior)
        
        if t is not None:
            y_smooth = restore_labels_after_gaps(y_obs, y_smooth, t, interval)
                
        return y_smooth

    def viterbi(self, y_obs, uniform_prior=None):
        """ Perform HMM smoothing over observations via Viteri algorithm
        https://en.wikipedia.org/wiki/Viterbi_algorithm
        :param y_obs: Predicted observation
        :param bool uniform_prior: Assume uniform priors.

        :return: Smoothed sequence of activities
        :rtype: np.ndarray
        """

        def log(x):
            return np.log(x + 1e-16)

        prior = np.ones(len(self.labels)) / len(self.labels) if (self.uniform_prior or uniform_prior) else self.prior
        emission = self.emission
        transition = self.transition
        labels = self.labels

        nobs = len(y_obs)
        n_labels = len(labels)

        y_obs = np.where(y_obs.reshape(-1, 1) == labels)[1]  # to numeric

        probs = np.zeros((nobs, n_labels))
        probs[0, :] = log(prior) + log(emission[:, y_obs[0]])
        for j in range(1, nobs):
            for i in range(n_labels):
                probs[j, i] = np.max(
                    log(emission[i, y_obs[j]]) +
                    log(transition[:, i]) +
                    probs[j - 1, :])  # probs already in log scale
        viterbi_path = np.zeros_like(y_obs)
        viterbi_path[-1] = np.argmax(probs[-1, :])
        for j in reversed(range(nobs - 1)):
            viterbi_path[j] = np.argmax(
                log(transition[:, viterbi_path[j + 1]]) +
                probs[j, :])  # probs already in log scale

        viterbi_path = labels[viterbi_path]  # to labels

        return viterbi_path

    def save(self, path):
        """
        Save model parameters to a Numpy npz file.

        :param str path: npz file location
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 prior=self.prior, emission=self.emission,
                 transition=self.transition, labels=self.labels)

    def load(self, path):
        """
        Load model parameters from a Numpy npz file.

        :param str path: npz file location
        """
        d = np.load(path, allow_pickle=True)
        self.prior = d['prior']
        self.emission = d['emission']
        self.transition = d['transition']
        self.labels = d['labels']
