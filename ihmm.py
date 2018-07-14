"""
Proof-of-concept implementation of an infinite Hidden Markov Model.

This file contains all classes needed to build an infinite HMM; for inference and training code, see
`samplers.py`.

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
    def __init__(self, init_matrix=None):
        # count_table represents the counts themselves; keys are always tuples:
        self.count_table = defaultdict(int)
        # mutable; max row size seen so far:
        self.max_rows = 0
        # mutable; max column size seen so far:
        self.max_cols = 0
        # if we want an initial matrix, update count_table and max_{rows,cols}:
        if (init_matrix is not None):
            assert (len(init_matrix.shape) == 2), "[Matrix] init_matrix must be a 2D numpy array"
            self.max_rows = init_matrix.shape[0]
            self.max_cols = init_matrix.shape[1]
            for r,c in zip(range(init_matrix.shape[0]), range(init_matrix.shape[1])):
                self.count_table[r,c] = init_matrix[r,c]

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
            return [self.count_table[(row,c)] for c in self.__parse_slice(col, self.max_cols)]
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
        self.max_rows = max(pair[0]+1, self.max_rows)
        self.max_cols = max(pair[1]+1, self.max_cols)
        self.count_table[pair] = val

    def numpy(self):
        """Return underlying matrix, in NumPy format."""
        return np.array(self[0:self.max_rows, 0:self.max_cols])

    def __repr__(self):
        """Format as string for printing."""
        return repr(self.numpy())

    @property
    def shape(self):
        """Return dimensions so far."""
        return (self.max_rows, self.max_cols)


class HierarchicalDirichletProcess(object):
    """
    Implementation of a 2-level hierarchical dirichlet process (HDP) for Infinite HMM state transitions/emissions.

    States are zero-indexed, i.e. the first state is `0`.
    """
    def __init__(self, alpha, beta, gamma):
        """
        * alpha: float; self-transition hyperparameter.
        * beta: float; innovation parameter that decides how probable the oracle DP is.
        * gamma: float; new-state pseudocount parameter for the oracle DP.
        """
        # input checks:
        assert (alpha >= 0.), "[HierarchicalDirichletProcess/__init__] `alpha` must be nonnegative"
        assert (beta >= 0.), "[HierarchicalDirichletProcess/__init__] `beta` must be nonnegative"
        assert (gamma >= 0.), "[HierarchicalDirichletProcess/__init__] `gamma` must be nonnegative"
        # static attributes:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # dynamic attributes:
        self.seen_states = 1          # NB. this keeps track of number of states visited so far
        self.base_counts = Matrix()   # NB. `base_counts[s,t]` records number of times we've performed s->t transition
        self.oracle_counts = Matrix() # NB. there should only be one column in the oracle counts, i.e. it's vector-shaped
        # initialize with full probability mass on first state:
        self.base_counts[0,0] += 1
        self.oracle_counts[0,0] += 1

    def sample(self, state):
        """
        Return a new state j given that we're currently in a state i.
        Running this method updates the underlying count tables (self.{base,oracle}_counts); use
        `HDP.probas(state)` to get the probability over all visited states `j`.

        * state: int; the state we're currently in.
        """
        # get probabilities for next state over all states observed so far, plus oracle proba in final index:
        base_probas = self.base_probas(state)
        # sample one of the states (or oracle query):
        next_state = np.random.choice(range(len(base_probas)), p=base_probas)
        # update tables and return state if our choice is not oracle:
        if next_state < self.seen_states:
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
            if next_oracle_state == (oracle_probas.shape[0]-1):
                self.seen_states += 1
            # return:
            return next_oracle_state

    def base_probas(self, state):
        """
        Returns probability over all states `j` observed so far, given that we're in state `i`.
        Returns a 1d numpy array of type np.float of size `self.num_states+1`, with the final
        value representing the probability that we make a query to the oracle DP.
        
        Args:
        * state: int; represents the state that we are in at the present moment.
        """
        # if queried state is out of bounds, raise error:
        assert (state < self.seen_states), "[Hierarchical Dirichlet Process/base_probas] `state` out of bounds"
        # raw counts from state `i` to each state `j` that we've already seen before:
        n_ijs = np.array(self.base_counts[state,0:self.seen_states], dtype=np.float64)
        # convert to probabilities:
        denominator = np.reciprocal(np.sum(n_ijs) + self.beta + self.alpha)
        state_probas = n_ijs * denominator
        state_probas[state] += self.alpha
        # compute leftover remaining probability mass (probability of oracle query):
        oracle_proba = self.beta * denominator
        # join together and return:
        combined_probas = np.concatenate((state_probas, [oracle_proba]), axis=0)
        return (combined_probas / combined_probas.sum())

    def oracle_probas(self):
        """
        Return an array of probabilities based on current configuration of `self.oracle_counts`.
        Returned 1d array of type np.float is of size `self.num_states+1`, representing probabiltiies
        for returning an existing state with an additional value at the end representing the probability
        for transitioning to a new, unseen state.
        """
        n_js = np.array(self.oracle_counts[:self.seen_states,0], dtype=np.float64)
        denominator = np.reciprocal(np.sum(n_js) + self.gamma)
        new_state_proba = self.gamma * denominator
        existing_state_probas = n_js * denominator
        combined_probas = np.concatenate((existing_state_probas, [new_state_proba]), axis=0)
        return (combined_probas / combined_probas.sum())


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
        return (np.array(states), np.array(observations))
