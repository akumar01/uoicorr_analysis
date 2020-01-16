# Numerics of the nested model selection problem
import numpy as np 
import itertools
import scipy
import sys
import schwimmbad
import pickle
import pdb
import time

from mpi4py import MPI

from dchisq import DChiSq

# n : number of samples
# sigma: noise standard deviation
# gamma: eigenvalue bound
# T: alternative support dimension
# Delta: deviation magnitude
# epsilon_1, 2: Bounded difference region
def mcdarmiad_bound(epsilon, n, sigma, T, gamma, Delta):
    epsilon_1, epsilon_2 = epsilon 

    c1 = sigma**2 * (T + 2 * epsilon_1 + 2 * np.sqrt(T * epsilon_1))
    c2 = gamma**2 * ((n - T) + 2 * epsilon_2 + 2 * np.sqrt((n - T) * epsilon_2))

    prob = np.exp(-epsilon_1) + np.exp(-epsilon_2) + \
            np.exp(-2 * (Delta - (np.exp(-epsilon_1 + np.exp(-epsilon_2))) * (c1 + c2))**2/\
                        (c1 + c2))

    return prob

# n : number of samples
# sigma: noise standard deviation
# gamma: eigenvalue bound
# T: alternative support dimension
# Delta: deviation magnitude
# t: free parameter in Chernoff bounding technique
def chernoff_bound(t, n, sigma, T, gamma, Delta):
    if not np.isscalar(t):
        t = t[0]
    # Optimize the log
    log_prob = -T/2 * np.log(1 - 2 * sigma**2 * t) - (n - T)/2 * np.log(1 + 2 * gamma**2 * t) -\
    t * (sigma**2 * T - gamma*2 * (n - T)) - t * Delta
    return log_prob

# Solve the derivative equation and hard code the optimal t, and then evaluate the resulting log probability
def explicit_chernoff_bound(n, sigma, T, gamma, Delta):

    # Some parameters we defined to simplify notation

    alpha = 2 * sigma**2
    beta = 2 * gamma**2
    mu = sigma**2 * T - (n - T) * gamma**2
    gamma = 2 * (Delta - mu)

    domain = [0, 1/(2 * sigma**2)]

    t1 = n * alpha * beta + alpha * gamma + beta * gamma - np.sqrt(4 * alpha * beta * (T * alpha - n * beta + T * beta - gamma) * gamma + (n * alpha * beta + alpha * gamma + beta * gamma)**2)
    t2 = n * alpha * beta + alpha * gamma + beta * gamma + np.sqrt(4 * alpha * beta * (T * alpha - n * beta + T * beta - gamma) * gamma + (n * alpha * beta + alpha * gamma + beta * gamma)**2)

    # If both are, that is quite intriguing, and let's inspect
    if t1 in domain and t2 in domain:
        pdb.set_trace()
    elif t1 in domain:
        log_prob = chernoff_bound(t1, n, sigma, T, gamma, Delta)
    elif t2 in domain:
        log_prob = chernoff_bound(t2, n, sigma, T, gamma, Delta)
    else:
        log_prob = 0

    return min(log_prob, 0)

# Use the Gil-Palaez inversion formuale
def direct_bound(n, sigma, T, gamma, Delta):
    # Note the ordering here is different than the large deviation bounds due 
    # to the sign of the inequality

    dchi2 = DChiSq(gamma**2, sigma**2, n - T, T)
    error_prob = dchi2.nCDF(Delta)
    return error_prob


def calc_error_probabilities(n, sigma, T, gamma, Delta):

    # Optimize Chernoff bound - start at 0 and bound to the range of the MGF
    bounds = (0, 1/(2 * sigma**2))

    # optimal_chernoff_bound = scipy.optimize.minimize_scalar(chernoff_bound, args = (n, sigma, T, gamma, Delta), bounds=bounds,
    #                                        method='Bounded').fun

    optimal_chernoff_bound = explicit_chernoff_bound(n, sigma, T, gamma, Delta)
    # # Optimize the McDarmiad Bound
    # optimal_mcdarmiad_bound = scipy.optimize.minimize(mcdarmiad_bound, [1, 1], 
    #                           args = (n, sigma, T, gamma, Delta)).fun

#    actual_prob = direct_bound(n, sigma, T, gamma, Delta0)

    return optimal_chernoff_bound#, optimal_mcdarmiad_bound

# Given a problem size, number of samples, and true mdoel dimension, calculate the 
# error probabilities associated with the list of penalty magnitudes given
def nested_model_selection(task_tuple):

    sigma, gamma, p = task_tuple

    # Let n equal p
    n = p

    # Let the true dimensionality of S vary over the same 
    S = np.arange(10, p/2, 50, dtype=np.int)

    penalties = np.linspace(0, 2 * np.log(n), 25)

#    actual_prob = np.zeros((T.size, S.size, penalties.size))
    chernoff_prob = np.zeros((S.size, penalties.size))
#    mcdarmiad_prob = np.zeros((T.size, S.size, penalties.size))
    t0 = time.time()
    for j, S_ in enumerate(S):
        for k, penalty in enumerate(penalties):
            cp = 0
            # Need to sum over the full set of nested penalties
            for T_ in range(np.arange(p/2)):
                # Note that the sign is reversed because we have reversed the order of
                # the difference to take an upper tail bound
                Delta = sigma**2 * penalty * (T_ - S_)        
                cp += calc_error_probabilities(n, sigma, T_, gamma, Delta) 
            chernoff_prob[i, j] = cp

    print(time.time() - t0) 
    probs = (chernoff_prob, task_tuple)
    return probs
# Sweep over problem parameters via schwimmbad
# All arguments should be ndarray-like
def sweep_problem_params(sigma, gamma, p, savename):

    # Create the outer product of all parameters
    tasks = itertools.product(sigma, gamma, p)

    comm = MPI.COMM_WORLD

    pool = schwimmbad.MPIPool(comm)
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
 
    results = list(pool.map(nested_model_selection, tasks))

    pool.close()

    if comm.rank == 0:
        with open(savename, 'wb') as f:
            f.write(pickle.dumps(results))

if __name__ == '__main__':

    
    # Parameters to sweep over
    p = np.logspace(2, 5, 20)
    sigma = np.linspace(1, 10, 5)
    gamma = np.linspace(0.01, 1, 5)
    sweep_problem_params(sigma, gamma, p, 'numerical_results2.dat')
