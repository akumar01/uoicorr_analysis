import pdb
import numpy as np 
import scipy as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Fit a sigmoid to X and y

# f(x) = c + L/(1 + e^{-k(x - x0)})

class Sigmoid():

    def __init__(self, robust=False, orientation='right'):
        self.robust = robust        
        self.orientation = orientation

    def sigmoid_fn(self, z, L, k, z0, c):
        if self.orientation == 'right':        

            return c + L/(1 + np.exp(-k *(z - z0)))

        elif self.orientation == 'left':

            return c + L/(1 + np.exp(k *(z - z0)))            

    def fit(self, X, y):

        # Set initial parameters using some basic herusitics
        c0 = min(y)
        L0 = max(y) - c0
        z0_0 = max(X) - (max(X) - min(X))/2
        # Use a linear fit to initialize
        k0 = np.abs(LinearRegression(normalize=True, fit_intercept=True).fit(X.reshape(-1, 1), y).coef_.ravel()[0])

        # Bound parameters within data range
        bounds = ([0, 0, min(X), 0], [1, max(X) - min(X), max(X), 1])

        if self.robust:

            sigmoid_loss = lambda params, xx, yy: self.sigmoid_fn(xx, *params) - yy

            # Use Huber loss
            opt_result = sp.optimize.least_squares(sigmoid_loss, x0=[L0, k0, z0_0, c0], args=(X, y),
                                                   bounds=bounds, loss='huber')

            self.coef_ = opt_result.x

        else:

            # Ordinary least squares
            opt_result = sp.optimize.curve_fit(self.sigmoid_fn, X, y, p0=[L0, k0, z0_0, c0],
                                 bounds=bounds)

            self.coef_ = opt_result[0]
            self.coef_cov = opt_result[1]


    def predict(self, X):

        return self.sigmoid_fn(X, *self.coef_)

class Exponen():

    def __init__(self, robust=False):
        self.robust = robust

    def exp_fn(self, z, L, k, z0, c):

        return c + L * np.exp(-k * (z - z0))

    def fit(self, X, y):

        # Set initial parameters using some basic heuristics
        c0 = min(y)
        L0 = 0.25
        z0_0 = np.min(X) + (np.max(X) - np.min(X))/2
        k0 = np.abs(LinearRegression(normalize=True, fit_intercept=True).fit(X.reshape(-1, 1), y).coef_.ravel()[0])

        # Bound paramters within data range
        bounds = ([0, 0, min(X), 0], [1, max(X) - min(X), max(X), 1])

        if self.robust:

            expon_loss = lambda params, xx, yy: self.exp_fn(xx, *params) - yy

            # Use Huber loss
            try:
                opt_result = sp.optimize.least_squares(expon_loss, x0=[L0, k0, z0_0, c0], args=(X, y),
                                                       bounds=bounds, loss='huber')
            except:
                pdb.set_trace()
            
            self.coef_ = opt_result.x

        else:

            # Ordinary least squares
            opt_result = sp.optimize.curve_fit(self.exp_fn, X, y, p0=[L0, k0, z0_0, c0],
                                 bounds=bounds)

            self.coef_ = opt_result[0]
            self.coef_cov = opt_result[1]

    def predict(self, X):

        return self.exp_fn(X, *self.coef_)


class AverageModel():

    def __init__(self):
        pass

    def fit(self, X, y):

        # Set the offset to the average value
        self.coef_ = [np.nan, np.nan, np.nan, np.mean(y)]

    def predict(self, X):

        return self.coef_[-1] * np.ones(X.size)



# Class to try and fit all 3 models to the data and choose the one that does the best w.r.t to R^2
class Fit_and_Select():

    def __init__(self, sig_orientation='left'):
        self.sig_orientation = sig_orientation

    def fit(self, X, y):

        # Try all three models
        avg_model = AverageModel()
        avg_model.fit(X, y)
        y_pred0 = avg_model.predict(X)

        expon_model = Exponen(robust=True)
        expon_model.fit(X, y)
        y_pred1 = expon_model.predict(X)
        sigmoid_model = Sigmoid(robust=True, 
                        orientation=self.sig_orientation)
        sigmoid_model.fit(X, y)
        y_pred2 = sigmoid_model.predict(X)

        # Score models
        r2_avg = r2_score(y, y_pred0)

        r2_expon = r2_score(y, y_pred1)

        r2_sigmoid = r2_score(y, y_pred2)

        if max([r2_avg, r2_expon, r2_sigmoid]) == r2_avg:
            self.coef_ = avg_model.coef_
            self.model_ = 'avg'
            self.score_ = r2_avg
            self.model_obj = avg_model
        elif max([r2_avg, r2_expon, r2_sigmoid]) == r2_expon:
            self.coef_ = expon_model.coef_
            self.model_ = 'exp'
            self.score_ = r2_expon
            self.model_obj = expon_model
        elif max([r2_avg, r2_expon, r2_sigmoid]) == r2_sigmoid:
            self.coef_ = sigmoid_model.coef_
            self.model_ = 'sig'
            self.score_ = r2_sigmoid
            self.model_obj = sigmoid_model

    def predict(self, X):
        if not hasattr(self.model_obj, 'orientation'):
            self.model_obj.orientation = 'left'
        return self.model_obj.predict(X)


