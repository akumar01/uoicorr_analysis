import numpy as np
from scipy.special import gamma, gammaln
import mpmath as mp
import scipy.integrate as integrate
import sympy
import pdb

class DChiSq(): 

    # Distribution of the weighted difference between two chi squared
    # random variables

    # alpha Chi^2(n) - beta Chi^2(m)

    # Lifted from https://projecteuclid.org/download/pdf_1/euclid.aoms/1177699531

    def __init__(self, alpha, beta, m, n):

        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.m = m

    # Psi function
    def psi_(self, a, b, x):
        # This does not return a number but rather an mpmath string
        return mp.hyperu(a, b, x)

    # Upper bound on psi_ using cauchy-schwarz
    def ub_psi_(self, a, b, x):

        return 1/mp.gamma(a) * \
               mp.sqrt(mp.power(2, a) * mp.power(x, -a) * mp.gamma(a) * mp.exp(x/2) * mp.expint(1 + a - b, x/2))

    # Normalization constant
    def norm_const(self):

        c = mp.power(mp.power(2, 0.5 * (self.n + self.m)), -1)
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

    # Return the log PDF (usually more manageable) 

    # The PDF is unimodal --> once we hit a threshold, stop evaluating as 
    # we then tend to get in trouble with negative values 
    # thresh: log threshold
    def logPDF(self, x, thresh = -50):

        if np.isscalar(x):
            x = np.array([x])

        xposmask = x >= 0
        xpos = x[xposmask]

        xnegmask = x < 0
        xneg = x[xnegmask]

        #### Positive part of the PDF
        if len(xpos) > 0:

            p1 = float(mp.log(self.norm_const()) -  mp.log(mp.gamma(self.m/2))) * np.ones(xpos.size)

            # Basic functions
            p1 += (self.m + self.n - 2)/2 * np.log(xpos) - xpos/(2 * self.alpha)

            # Keep track of diffs
            pdiffs = np.zeros(x.size - 1)

            for i, xx in enumerate(xpos):
#                try:
                psi_ = self.psi_(self.n/2, 
                                 (self.m + self.n)/2, 
                                 (self.alpha + self.beta)/(2 * self.alpha * self.beta) * xx)
                p1[i] = p1[i] + psi_                
#                except:
#                    p1[i] = np.nan             
            #     if i > 0:
            #         pdiffs[i - 1] = p1[i] - p1[i -1]

            #         if pdiffs[i - 1] < 0 and p1[i] < thresh:

            #             break 

            # # Set anything leftover if the threshold has been crossed to 0

            # if i < xpos.size - 1:

            #     p1[i:] = np.nan
        else:
            p1 = []

        #### Negative part of the PDf
        if len(xneg) > 0:
            xneg = np.abs(xneg)
            p2 = float(mp.log(self.norm_const() / mp.gamma(self.n/2))) * np.ones(xneg.size)

            # Basic functions
            p2 += (self.m + self.n - 2) * np.log(xneg) - xneg/(2 * self.beta)

            for i, xx in enumerate(xpos):
                try:
                    p2[i] += float(mp.log(self.psi_(self.m/2, 
                                                    (self.m + self.n)/2, 
                                                    (self.alpha + self.beta)/(2 * self.alpha * self.beta) * xx)))
                except:
                    p2[i] = np.nan             


        else:
            p2 = []

        p = np.zeros(x.size)
        p[xposmask] = p1
        p[xnegmask] = p2

        if p.size == 1:
            return p[0]
        else:
            return p        

    def char_fn(self, t, alpha = None, beta = None, n = None, m = None):

        # Product of chi squared characteristic functions, rescaled by factors to adjust
        # for the fact that we have the weighted difference. Also note the complex conjugation

        return mp.mpc((1 - 2 * 1j * self.alpha * t)**(-self.m/2) * (1 + 2 * self.beta * 1j * t)**(-self.n/2))


    # Calculate the mean and variance in a region surrounding the 
    def mean(self):

        return self.alpha * self.m - self.beta * self.n

    def variance(self):

        return 2 * (self.m * self.alpha**2 * self.n * self.beta**2)

    # Use sympy to calculate the nth derivative of the char. fn. symbolically
    def diff(self, n):

        a, b, m, n, t = sympy.symbols('a b m n t')
        return diff((1 - 2 * I * a * t)**(-m/2) * (1 + 2 * I * b * t)**(-n/2), t, n)

    # Calculate the symbolic derivatives once up front and store them in a list
    # Calculates all derivatives of order <= n
    def gen_diffs(self, n):

        self.diffs = []
        for i in range(1, n + 1):

            self.diffs.append(self.diff(i))

    def series_term(self, order):



    # Use the method detailed in https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2004.1401
    # to handle highly oscillatory instantiations of the characteristic function inversion integral
    def asymptotic_expansion(self):

        # Steps
        # (1) Decide what the cutoff for integration will be 
        # (2) we have to rescale our integral to fit on [0, 1]
        # (3) Generate derivatives for the asymptotic expansion
        # (4) Evaluate the expansion

        # For the cutoff, we can evaluate the norm of the characteristic function

        # Evaluate the modulus of the characteristic function
        # I would be shocked if [0, 5] is not a large enough range
        domain = np.linspace(0, 5, 5000)
        char_fn = list(map(lambda t: mp.fabs(self.char_fn(t)), np.linspace(0, 1, 1000)))

        # thresh 1e-50 
        thresh_check = (domain[i] for i in range(1000) if char_fn[i] < mpf(1e-50))
        cutoff = next(thresh_check)

        # Generate the derivatives for the asymptotic expansion
        # Try fifth order
        order = 5

        if not hasattr(self, diffs):
            self.gen_diffs(order)

        # Evaluate the expansion
        asym_series = mp.matrix(5, 1)
        for i in range(order):
            asym_series[i] = self.series_term(i)

        # Sum up and multiply by cutoff (rescaling)
        return -cutoff * mp.fsum(asym_series)

    # Calculate the PDF via numerical inversion of the characteristic function
    def nPDF(self, x):

        p = np.zeros(x.size)
        for i, xx in enumerate(x):

            gil_pelaez = lambda t: mp.re(self.char_fn(t) * mp.exp(-1j * t * xx))

            I = mp.quad(gil_pelaez, [0, np.inf])
            p[i] = 1/np.pi * float(I)

        return p

    # Calculate the CDF via numerical inversion of the characteristic function
    def nCDF(self, x):

        p = np.zeros(x.size)
        for i, xx in enumerate(x):

            gil_pelaez = lambda t: mp.im(self.char_fn(t) * mp.exp(-1j * t * xx))/t

            I = mp.quad(gil_pelaez, [0, np.inf])
            p[i] = 1/2 - 1/np.pi * float(I)

        return p

    def MCCDF_(self, x, n_samples = 1000000):

        # Use Monte Carlo samples to approximate the CDF

        threshold_count = 0

        for _ in range(n_samples):

            x1 = np.random.chisquare(self.m)
            x2 = np.random.chisquare(self.n)

            sample = self.alpha * x1 - self.beta * x2 

            if sample <= x:
                threshold_count += 1

        return float(threshold_count)/float(n_samples)


    # Return the CDF evaluated at x (scalar or ndarray)
    def CDF(self, x, method='MC'):

        if np.isscalar(x):
            x = np.array([x])

        c = np.zeros(x.size)
        for i, xx in enumerate(x):
            c[i] = self.MCCDF_(xx)

        return c

    # Return the inverse CDF evaluated at x:
    def invCDF(self, x):
        pass
