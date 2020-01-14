# Numerics of the nested model selection problem
import numpy as np 
import scipy
import sys
import schwimmbad
import pickle
import pdb

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

    t = t[0]

    # Optimize the log
    log_prob = -T/2 * np.log(1 - 2 * sigma**2 * t) - (n - T)/2 * np.log(1 + 2 * gamma**2 * t) - \
    t * (sigma**2 * T - gamma*2 * (n - T)) - t * Delta
    return log_prob

# Use the Gil-Palaez inversion formuale
def direct_bound(n, sigma, T, gamma, Delta):
    # Note the ordering here is different than the large deviation bounds due 
    # to the sign of the inequality

    dchi2 = DChiSq(gamma**2, sigma**2, n - T, T)
    error_prob = dchi2.nCDF(Delta)
    return error_prob


def calc_error_probabilities(n, sigma, T, gamma, Delta, Delta0):

    # Optimize Chernoff bound - start at 0 and bound to the range of the MGF
    bounds = [(0, 1/(2 * sigma**2))]

    optimal_chernoff_bound = scipy.optimize.minimize(chernoff_bound, 0, args=(n, sigma, T, gamma, Delta), bounds=bounds).fun

    # Optimize the McDarmiad Bound
    optimal_mcdarmiad_bound = scipy.optimize.minimize(mcdarmiad_bound, [1, 1], 
                              args = (n, sigma, T, gamma, Delta)).fun

    actual_prob = direct_bound(n, sigma, T, gamma, Delta0)

    return actual_prob, optimal_chernoff_bound, optimal_mcdarmiad_bound

# Given a problem size, number of samples, and true mdoel dimension, calculate the 
# error probabilities associated with the list of penalty magnitudes given
def nested_model_selection(task_tuple):

    sigma, gamma, p = task_tuple

    # Let n equal p
    n = p

    # Possible alternative model support dimensionalities
    T = np.arange(p/2)

    # Let the true dimensionality of S vary over the same 
    S = np.arange(p/2)

    penalties = np.linspace(0, 2 * np.log(n), 25)

    actual_prob = np.zeros((T.size, S.size, penalties.size))
    chernoff_prob = np.zeros((T.size, S.size, penalties.size))
    mcdarmiad_prob = np.zeros((T.size, S.size, penalties.size))

    for i, T_ in enumerate(T):
        for j, S_ in enumerate(S):
            for k, penalty in enumerate(penalties):
                # Note that the sign is reversed because we have reversed the order of
                # the difference to take an upper tail bound
                Delta = sigma**2 * penalty * (T - S)        
                # normal ordering that we put into the direct bound
                Delta0 = sigma**2 * penalty * (S - T)
                # Need to measure deviation from the mean
                mu = sigma**2 * T_ - gamma**2 * (n - T_)

                e1, e2, e3 = calc_error_probabilities(n, sigma, T_, gamma, Delta - mu, Delta0) 
                actual_prob[i, j, k] = e1
                chernoff_prob[i, j, k] = e2
                mcdarmiad_prob[i, j, k] = e3
    probs = (actual_prob, chernoff_prob, mcdarmiad_prob) 
    return probs
# Sweep over problem parameters via schwimmbad
# All arguments should be ndarray-like
def sweep_problem_params(sigma, gamma, p, savename):

    # Create the outer product of all parameters
    tasks = itertools.product(sigma, gamma, p, S, penalties)

    comm = MPI.COMM_WORLD

    pool = schwimmbad.MPIPOOL(comm)

    results = list(pool.map(nested_model_selection, tasks))

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    pool.close()

    if comm.rank == 0:
        with open(savename, 'wb') as f:
            f.write(pickle.dumps(results))

if __name__ == '__main__':

    savename = sys.argv[1]

    # Parameters to sweep over
    p = np.logspace(2, 5, 25)
    sigma = np.linspace(1, 10, 10)
    gamma = np.linspace(0.01, 1, 10)
    sweep_problem_params(sigma, gamma, p, 'numerical_results.dat')