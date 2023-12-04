import numpy as np
import scipy as sp
import torch


def solve(A, b):
    c, low = sp.linalg.cho_factor(A)
    x = sp.linalg.cho_solve((c, low), b)
    return x


def append_intercept_scaling(phi, intercept_scaling):
    return np.concatenate(
        [phi, np.full(phi.shape[:-1] + (1,), intercept_scaling)],
        axis=-1)


class BayesianLinearRegression:
    """Bayesian Linear Regression model. Follows Bishop 3.3.
    """
    def __init__(
            self, d, mean=None, precision=1., noise_precision=None,
            intercept_scaling=0.
    ):
        """
        Args:
            d: int, dimension of the space
            mean: m_0, prior mean
            precision: S_0^{-1}, prior covariance; If given as a scalar,
                interpret the scalar as the alpha and the precision is alpha I
            noise_precision: beta, the noise precision; If None, must give
                heteroskedastic noise when calling the fit method.
            intercept_scaling: features phi becomes [phi, intercept_scaling],
                i.e., a "synthetic" feature with constant value equal to
                intercept_scaling is appended to represent the bias term. Note
                the synthetic feature weight is also represented in the prior
                and the posterior.
        """
        self.d, self._d = d, d
        self.with_bias = bool(intercept_scaling)
        if self.with_bias:
            self.intercept_scaling = intercept_scaling
            self._d += 1
        if mean is None:
            mean = np.zeros((self._d,), dtype=float)
        else:
            mean = np.array(mean)
        self.mean = mean
        precision = np.array(precision)
        if precision.ndim == 0:
            precision = precision * np.eye(self._d, dtype=float)
        self.precision = precision
        self.noise_precision = noise_precision

    def fit(self, Phi, y, noise_precision=None):
        """Update the posteriors by observing data.
        Args:
            Phi: design matrix, float array of shape (n, d)
            y: targets, float array of shape (n,)
            noise_precision (optional): noise. float array of shape (n,) for
                heteroskedastic noise; float scalar for homoskedastic noise.
                None for self.noise_precision.
        """
        if self.with_bias:
            Phi = append_intercept_scaling(Phi, self.intercept_scaling)
        if noise_precision is None:
            noise_precision = self.noise_precision
        pos_precision = self.precision + Phi.T * noise_precision @ Phi
        pos_mean = solve(
            pos_precision,
            self.precision @ self.mean + Phi.T @ (noise_precision * y))
        self.mean, self.precision = pos_mean, pos_precision

    def predictive_distribution(self, phi):
        """Predictive distribution on test points x.
        Args:
            phi: phi(x), features of test points.
        Returns:
            predictive_distribution: Normal(pred_mean, pred_variance)
        """
        if self.with_bias:
            phi = append_intercept_scaling(phi, self.intercept_scaling)
        #c, low = sp.linalg.cho_factor(self.precision)
        covariance = sp.linalg.inv(self.precision)
        def _get_epistemic_variance(phi_x):
            #return phi_x @ solve((c, low), phi_x)
            return phi_x @ covariance @ phi_x
        pred_variance = np.apply_along_axis(_get_epistemic_variance, -1, phi)
        if self.noise_precision is not None:
            pred_variance += 1. / self.noise_precision
        pred_mean = np.dot(phi, self.mean)
        pred_mean, pred_variance = torch.tensor(pred_mean), torch.tensor(pred_variance)
        return torch.distributions.normal.Normal(pred_mean, torch.sqrt(pred_variance))


class LabelRegressionModel:
    """Direct linear regression labels. Same as BayesianLinearRegression, but
    encapsulates the transformation between {0, 1} and {-1, +1}.
    """
    def __init__(
            self, d, mean=None, precision=1., noise_precision=None,
            intercept_scaling=0.
    ):
        self.regression_model = BayesianLinearRegression(
            d, mean=mean, precision=precision, noise_precision=noise_precision,
            intercept_scaling=intercept_scaling,
        )

    def fit(self, x, y, noise_precision=None):
        return self.regression_model.fit(
            x, y * 2 - 1, noise_precision=noise_precision)
    
    def predict_proba(self, x):
        pred_dist = self.regression_model.predictive_distribution(x)
        pred_prob0 = pred_dist.cdf(torch.zeros_like(pred_dist.mean))
        pred_prob1 = 1 - pred_prob0
        pred_probs = np.column_stack((pred_prob0, pred_prob1))
        return pred_probs