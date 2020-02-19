import pdb
import numpy as np 
import scipy as sp
from sklearn.linear_model import LinearRegression

# Fit a sigmoid to X and y

# f(x) = c + L/(1 + e^{-k(x - x0)})

class Sigmoid():

    def __init__(self):
        pass

    def sigmoid_fn(self, z, L, k, z0, c):

        return c + L/(1 + np.exp(-k *(z - z0)))

    def fit(self, X, y):

        # Set initial parameters using some basic herusitics
        c0 = min(y)
        L0 = max(y) - c0
        z0_0 = max(X) - (max(X) - min(X))/2
        # Use a linear fit to initialize
        k0 = np.abs(LinearRegression(normalize=True, fit_intercept=True).fit(X.reshape(-1, 1), y).coef_.ravel()[0])

        # Bound parameters within data range
        bounds = ([0, 0, min(X), 0], [1, max(X) - min(X), max(X), 1])

        # Test without jacobian first, add if ncesssary

        opt_result = sp.optimize.curve_fit(self.sigmoid_fn, X, y, p0=[L0, k0, z0_0, c0],
                              bounds=bounds)

        self.coef_ = opt_result[0]
        self.coef_cov = opt_result[1]