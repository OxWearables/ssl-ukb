import numpy as np
import hmmlearn.hmm as hmm

from models.hmm import HMM
from models.utils import check_for_time_values_error, restore_labels_after_gaps

class HMMLearn(HMM):
    """
    Implement a HMM_learn model with parameter saving/loading.
    Note that this is saving the initial parameters of the HMM
    """
    def __str__(self):
        return "prior: {prior}\n" \
               "emission: {emission}\n" \
               "labels: {labels}".format(prior=self.prior, emission=self.emission,
                                         labels=self.labels)

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

        self.prior = prior
        self.emission = emission
    
    def predict(self, y_obs, t=None, interval=None, uniform_prior=None):
        check_for_time_values_error(y_obs, t, interval)

        model = hmm.MultinomialHMM(n_components=len(self.labels), random_state=42, n_iter=1000,
                                   params="st", init_params="")
        
        model.startprob_ = np.ones(len(self.labels)) / len(self.labels) if (self.uniform_prior or uniform_prior) else self.prior
        model.emissionprob_ = self.emission

        model.fit(y_obs.reshape(-1,1))

        y_smooth = model.predict(y_obs.reshape(-1,1))

        if t is not None:
            y_smooth = restore_labels_after_gaps(y_obs, y_smooth, t, interval)
        
        return y_smooth
