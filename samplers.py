"""
This file contains code for inference, training, and likelihood estimation for the iHMM
class defined in `ihmm.py`.

(N.B.: Gibbs sampling is deprecated in favor of the beam sampling, which is shown to typically be
faster/more efficient; hence we won't include this.)

Many of the algorithms here take their influence from J. Van Gael's original MATLAB code, a fork of
which can be found at: `https://github.com/tmills/ihmm`
"""
import numpy as np
import scipy.stats as stats
from ihmm import InfiniteHMM

# ----- beam sampling helper functions:
def _forward_slice_sample(states, transition_hdp):
    """
    Args:
    * states: sequence of integer states of length T.
    * transition_hdp: a hierarchical dirichlet process (instance of HierarchicalDirichletProcess)
      representing the transition probabilities from a given iHMM.
    
    Returns:
    * u_slices: samples u_t ~ Uniform(0, transition_proba(s_{t-1} -> s_t)) for t=1:T.
    """
    return None # TODO

def _backward_dynamic_resample(slices, obs):
    """
    Sample new state sequence given a set of observations and limiting slices.

    Args:
    * slices: (...)
    * obs: (...)
    
    Returns: a slice-sampled state sequence `states`.
    """
    return None # TODO

def _resample_parameters(ihmm, states):
    """
    Resample the underlying parameters of an iHMM given a state sequence; updates the HMM parameters
    in-place.
    """
    return None # TODO


# ----- main beam-sampling function:
def beam_sample(observations, init_states=None, init_ihmm=None):
    """
    Beam sampling, for inference of a hidden sequence on an infinite HMM given an emission sequence.
    
    Args:
    * observations: a sequence of integers repesenting the output of an infinite HMM with unknown values.
    * init_states

    Returns: a tuple (states, ihmm) where:
    * states: a sequence of integers representing the state sequence corresponding to `observations`.
    * ihmm: the infinite hidden markov model (an instance of InfiniteHMM) that generated (observations, states).
    """
    # --- 1. initialize an iHMM and an initial "hypothesis" hidden state sequence:
    if (init_ihmm is None):
        t_alpha0 = None # TODO
        t_beta0 = None # TODO
        t_gamma0 = None # TODO
        e_beta0 = None # TODO
        e_gamma0 = None # TODO
        ihmm = InfiniteHMM(t_alpha, t_beta, t_gamma, e_beta, e_gamma)
    else:
        ihmm = init_ihmm
    states = np.zeros(observations.shape) if (init_states is None) else init_states
    
    # --- 2. loop through observations:
    while (condn): # FIGURE OUT CONDITION HERE
        # forward pass to get slices:
        uniform_slices = _forward_slice_sample(states, ihmm.t_hdp)
        # backwards pass through finite slices:
        states = _backward_dynamic_resample(uniform_slices, observations)
        # update iHMM given new estimated state sequence:
        _resample_parameters(ihmm, states)

    return (states, ihmm)


# ----- joint log-likelihood of an (observations, states) pair:
def joint_log_likelihood(observations, states, ihmm):
    """
    Compute the log-likelihood of observing an (observations,states) pair coming from an infinite HMM
    with specific transition & emission HDPs and hyperparameters.

    Args:
    * (...)
    
    Returns:
    * (...)
    """
    return None # TODO
