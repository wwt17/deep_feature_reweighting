import numpy as np
import scipy as sp
import torch


def solve(A, b):
    c, low = sp.linalg.cho_factor(A)
    x = sp.linalg.cho_solve((c, low), b)
    return x


class BayesianLinearRegression:
    """Bayesian Linear Regression model. Follows Bishop 3.3.
    """
    def __init__(self, d, mean=None, precision=1., noise_precision=10.):
        """
        Args:
            d: int, dimension of the space
            mean: m_0, prior mean
            precision: S_0^{-1}, prior covariance; If given as a scalar,
                interpret the scalar as the alpha and the precision is alpha I
            noise_precision: beta, the noise precision
        """
        self.d = d
        if mean is None:
            mean = np.zeros((d,), dtype=float)
        else:
            mean = np.array(mean)
        self.mean = mean
        precision = np.array(precision)
        if precision.ndim == 0:
            precision = precision * np.eye(d, dtype=float)
        self.precision = precision
        self.noise_precision = noise_precision

    def fit(self, Phi, y):
        """Update the posteriors by observing data.
        Args:
            Phi: design matrix, float array of shape (n, d)
            y: target labels, int array of shape (n,)
        """
        pos_precision = self.precision + self.noise_precision * (Phi.T @ Phi)
        pos_mean = solve(pos_precision, self.precision @ self.mean + self.noise_precision * (Phi.T @ y))
        self.mean, self.precision = pos_mean, pos_precision

    def predictive_distribution(self, phi):
        """Predictive distribution on test points x.
        Args:
            phi: phi(x), features of test points.
        Returns:
            pred_mean, pred_variance: the predictive distribution is
                Normal(pred_mean, pred_variance)
        """
        #c, low = sp.linalg.cho_factor(self.precision)
        covariance = sp.linalg.inv(self.precision)
        def _get_epistemic_variance(phi_x):
            #return phi_x @ solve((c, low), phi_x)
            return phi_x @ covariance @ phi_x
        pred_variance = 1. / self.noise_precision + np.apply_along_axis(_get_epistemic_variance, -1, phi)
        pred_mean = np.dot(phi, self.mean)
        pred_mean, pred_variance = torch.tensor(pred_mean), torch.tensor(pred_variance)
        return torch.distributions.normal.Normal(pred_mean, torch.sqrt(pred_variance))