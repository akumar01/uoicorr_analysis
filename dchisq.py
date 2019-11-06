import numpy as np
from scipy.special import gamma
import scipy.integrate as integrate

class DChiSq(): 

    # Distribution of the weighted difference between two chi squared
    # random variables

    # alpha Chi^2(n) - beta Chi^2(m)

    # Lifted from https://projecteuclid.org/download/pdf_1/euclid.aoms/1177699531

    def __init__(self, alpha, beta, n, m):

        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.m = m

    # Psi function
    def psi_(self, a, b, x):

        I = integrate.quad(lambda t: np.exp(-t * x) * t**(a - 1) * (1 + t)**(b - a - 1), 
                        0, np.inf)[0]

        return np.power(gamma(a), -1) * I

    # Normalization constant
    def norm_const(self):

        c = (np.power(2, 0.5 * (self.n + self.m)) * gamma(self.m/2))**(-1)
        return c

    # Return the PDF evaluated at x (scalar or ndarray)
    def PDF(self, x):

        if np.isscalar(x):
            x = np.array([x])

        p = self.norm_const() * np.ones(x.size)

        # Negate all x that are less than 0
        x = np.abs(x)

        # Use efficient numpy to evaluate the basic functions
        p = np.multiply(p, np.power(x, self.m + self.n - 2) * np.exp(-x/(2 * self.alpha)))        

        # List comprehension to evaluate psi
        psi = np.array([self.psi_(self.n/2, (self.m + self.n)/2, 
                        (self.alpha + self.beta)/(2 * self.alpha * self.beta) * xx) 
                        for xx in x])
        p = np.multiply(p, psi)

        if p.size == 1:
            return p[0]
        else:
            return p

    # Return the CDF evaluated at x
    def CDF(self, x):
        pass

    # Return the inverse CDF evaluated at x:
    def invCDF(self, x):
        pass
