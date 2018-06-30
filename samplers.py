"""
This file contains code for inference, training, and probability estimation for the iHMM
class defined in `ihmm.py`.
"""
import numpy as np
import scipy.stats as stats


# - - - - - Inference Code
class BeamSampler(object):
    """
    Beam sampling, for inference of hidden sequence on an infinite HMM given an emission sequence.
    """
    def __init__(self, ihmm):
        """
        Args:
        * ihmm: instance of InfiniteHMM.
        * (...TBD...)
        """
        self.ihmm = ihmm
        # TBD

    def inference(self, observations):
        """
        Return the most likely sequence of states given an array of observations.
        
        Args:
        * observations: a 1d np.array with dtype=float64 representing a sequence of
        emissions from the infinite HMM.
        """
        return np.random.rand(observations.shape[0]) # FIX THIS --- TODO


class GibbsSampler(object):
    """
    Gibbs sampling for inference of hidden sequence on an infinite HMM given an emission sequence.

    N.B.: this sampler is deprecated in favor of the Beam sampler, which is shown to typically be
    faster/more efficient in the second paper referenced above.
    """
    def __init__(self, ihmm):
        """TODO"""
        return None # TODO

    def inference(self, observations):
        """TODO"""
        return None # TODO


class ParticleFilter(object):
    """
    Particle-filtering class that computes likelihoods for sequences of observations, given an iHMM.
    This does not change the internal parameters of the iHMM.
    """
    def __init__(self, ihmm):
        """TODO"""
        return None # TODO

    def likelihood(self, observations):
        """Return likelihood of observations given an infinite HMM."""
        return None # TODO


# - - - - - Training Code
def train_ihmm(states, obs, ihmm, n_epochs):
    """
    Return the best hyperparameters (t_alpha, t_beta, t_gamma, e_beta, e_gamma) for an iHMM
    given a sequence of states and their corresponding observations.

    This function is bayesian; all hyperparameters have a vague Gamma-distributed prior and
    the posteriors follow the results in section 4.2 of the paper.

    # 
    """
    # ...
    return None # TODO
