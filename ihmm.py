"""
Proof-of-concept implementation of an infinite Hidden Markov Model.

[----- TODO: provide more information on what these things are here -----]

References:
* "The Infinite Hidden Markov Model", M. Beal, Z. Ghahramani, C. Rasmussen
* "Beam Sampling for the Infinite Hidden Markov Model", J. Van Gael, Y. Saatci, Y. Teh, Z. Ghahramani
"""
import numpy as np
import scipy.stats as stats
from collections import defaultdict


class Matrix(object):
    """
    A matrix of mutable size, used to track counts between nodes in state-transition/emissions
    dirichlet processes; essentially just a wrapper class around a defaultdict where key-value
    mappings are of form [(Int x Int) => Int].

    The first value of a key-pair always represents the row of the matrix, while the second
    value represents the column. We overload the indexing operator (`[]`) to support indexes,
    slices, etc.

    Usage:
    mat = Matrix()
    mat[i,j] # returns an integer; represents observed i->j counts when used in HDPs
    mat[i,j] += 1
    mat[i:i+5,j] # returns a list [ mat[i,j], mat[i+1,j], ..., mat[i+4,j] ]
    # (note that slices with steps != 1 (e.g. mat[i:i+5:2, j]) are not supported)
    """
    def __init__(self):
        # count_table represents the counts themselves; keys are always tuples:
        self.count_table = defaultdict(int)
        # mutable; max row size seen so far:
        self.max_rows = 0
        # mutable; max column size seen so far:
        self.max_cols = 0

    def __getitem__(self, pair):
        row, col = pair
        # both row and col are integers:
        if (not isinstance(row, slice)) and (not isinstance(col,slice)):
            return self.count_table[(row,col)]
        # row is slice, col is integer:
        elif isinstance(row,slice) and (not isinstance(col,slice)):
            return [self.count_table[(r,col)] for r in self.__parse_slice(row, self.max_rows)]
        # row is integer, col is slice:
        elif (not isinstance(row,slice)) and isinstance(col,slice):
            return [self.count_table(row,c) for c in self.__parse_slice(col, self.max_cols)]
        # both row and col are slices:
        elif isinstance(row,slice) and isinstance(col,slice):
            return [[self.count_table[(r, c)] for c in self.__parse_slice(col, self.max_cols)]
                    for r in self.__parse_slice(row, self.max_rows)]
                    
        # raise exception if pair is of invalid type:
        else:
            raise Exception("[Matrix.__getitem__] key of type {} not recognized".format(type(pair)))

    def __parse_slice(self, sl, upper_lim):
        """Helper function to parse a slice into a list of integers."""
        if sl.step is not None:
            raise Exception("[Matrix.__parse_slice] step != 1 not supported")
        start = 0 if (sl.start is None) else min(upper_lim, sl.start)
        stop = upper_lim if (sl.stop is None) else min(upper_lim, sl.stop)
        return range(start,stop)

    def __setitem__(self, pair, val):
        self.max_rows = max(pair[0], self.max_rows)
        self.max_cols = max(pair[1], self.max_cols)
        self.count_table[pair] = val


class HierarchicalDirichletProcess(object):
    """Implementation of a 2-level hierarchical dirichlet process (HDP) for Infinite HMM state transitions/emissions."""
    def __init__(self, alpha, beta, gamma):
        """        
        * alpha: float; self-transition hyperparameter.
        * beta: float; innovation parameter that decides how probable the oracle DP is.
        * gamma: float; new-state pseudocount parameter for the oracle DP.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_states = 0 # keeps track of number of states visited so far
        self.base_counts = Matrix()
        self.oracle_counts = Matrix() # there should only be one column in the oracle counts

    def sample(self, state):
        """
        Return a new state j given that we're currently in a state i.
        Running this method updates the underlying count tables (self.{base,oracle}_counts); use
        `HDP.probas(state)` to get the probability over all visited states `j`.
        """
        # get probabilities for next state over all states observed so far, plus oracle DP in final index:
        base_probas = self.probas(state)
        # sample one of the states (or oracle query):
        next_state = np.random.choice(range(len(base_probas)), p=base_probas)
        # update tables and return state if our choice is not oracle:
        if next_state != base_probas.shape[0]:
            self.base_counts[state,next_state] += 1
            return next_state
        # otherwise if we choose final state, sample from oracle (also updating count tables/num_states):
        else:
            oracle_probas = self.oracle_probas()
            next_oracle_state = np.random.choice(range(len(oracle_probas)), p=oracle_probas)
            # update both counts:
            self.base_counts[state,next_oracle_state] += 1
            self.oracle_counts[next_oracle_state,0] += 1
            # update num_states if new state seen:
            if (next_oracle_state ==  oracle_probas.shape[0]):
                self.num_states += 1
            # return:
            return next_oracle_state

    def probas(self, state):
        """
        Returns probability over all states `j` observed so far, given that we're in state `i`.
        Returns a 1d numpy array of type np.float of size `self.num_states+1`, with the final
        value representing the probability that we make a query to the oracle DP.
        """
        # raw counts from state `i` to each state `j` that we've already seen before:
        # [[TODO: uncomment the first line once Matrix slicing is better-supported]]
        #n_ijs = np.array(self.base_counts[state,0:self.num_states], dtype=np.float)
        n_ijs = np.array([self.base_counts[state,j] for j in range(self.num_states)], dtype=np.float)
        n_ijs[state] += self.alpha
        # convert to probabilities:
        denominator = np.reciprocal(np.sum(n_ijs) + self.beta + self.alpha)
        state_probas = n_ijs * denominator
        # compute leftover remaining probability mass (probability of oracle query):
        oracle_proba = self.beta * denominator
        # join together and return:
        return np.concatenate((state_probas, [oracle_proba]), axis=0)

    def oracle_probas(self):
        """
        Return an array of probabilities based on current configuration of `self.oracle_counts`.
        Returned 1d array of type np.float is of size `self.num_states+1`, representing probabiltiies
        for returning an existing state, plus probabilities for returning 
        """
        n_js = np.array([self.oracle_counts[j,0] for j in range(self.num_states)], dtype=np.float)
        denominator = np.reciprocal(np.sum(n_js) + self.gamma)
        new_state_proba = self.gamma * denominator
        existing_state_probas = n_js * denominator
        return np.concatenate((existing_state_probas, [new_state_probas]), axis=0)


class InfiniteHMM(object):
    """
    An infinite HMM; state space distributions are dirichlet process mixtures.
    Inference done by using the BeamSampler class.

    State transitions and emissions are both governed by HDPs; the latter
    HDP has an alpha hyperparameter set to 0.0, to indicate the lack of self-transitions.
    """
    def __init__(self, t_alpha, t_beta, t_gamma, e_beta, e_gamma):
        """
        Args:
        * t_alpha: float; initial value for transition HDP alpha parameter.
        * t_beta: float; initial value for transition HDP beta parameter.
        * t_gamma: float; initial value for transition HDP gamma parameter.
        * e_beta: float; initial value for emissions HDP beta parameter.
        * e_gamma float; initial value for emissions HDP gamma parameter.
        """
        # set up transition process:
        self.t_alpha = t_alpha
        self.t_beta = t_beta
        self.t_gamma = t_gamma
        self.t_hdp = HierarchicalDirichletProcess(t_alpha, t_beta, t_gamma)

        # set up emission process:
        self.e_beta = e_beta
        self.e_gamma = e_gamma
        self.e_hdp = HierarchicalDirichletProcess(0., e_beta, e_gamma) # alpha clamped to 0.0

    def sample(self, num_steps):
        """Sample a random path through the state space and a corresponding sequence of observations."""
        # first timestep starts off at state 0:
        states = [self.t_hdp.sample(0)]
        observations = [self.e_hdp.sample(states[-1])]
        # keep sampling next-states and observations for `num_steps`:
        for _ in range(1,num_steps):
            sampled_state = self.t_hdp.sample(states[-1])
            sampled_obs = self.e_hdp.sample(sampled_state)
            states.append(sampled_state)
            observations.append(sampled_obs)
        return (states, observations)


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
    
    N.B.: this sampler is deprecated in favor of the Beam sampler, which is shown to be more efficient
    in a paper referenced above.
    """
    def __init__(self, ihmm):
        """TODO"""
        return None # TODO

    def inference(self, observations):
        """TODO"""
        return None # TODO


class ParticleFilter(object):
    """
    Particle-filtering class that computes likelihoods for sequences of observations.
    """
    def __init__(self, ihmm):
        """TODO"""
        return None # TODO

    def likelihood(self, observations):
        """Return likelihood of observations given an infinite HMM."""
        return None # TODO
