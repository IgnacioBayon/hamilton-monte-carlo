import numpy as np


class Density:
    def U(
        self,
        x: np.ndarray,
    ) -> float:
        pass

    def grad_U(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        pass


class CustomGaussian(Density):
    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        grad_weights: np.ndarray = None,
        grad_means: np.ndarray = None,
        grad_covariances: np.ndarray = None,
    ):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.grad_weights = grad_weights
        self.grad_means = grad_means
        self.grad_covariances = grad_covariances

    def U(
        self,
        x: np.ndarray,
    ) -> float:
        """Unnormalised log density log f(x) for a N-component 2D Gaussian mixture

        Args:
            x (np.ndarray): Input 2D point

        Returns:
            float: Unnormalised log density log f(x)
        """
        x = np.asarray(x)
        d = 2

        log_weights = np.log(self.weights)
        log_two_pi = np.log(2 * np.pi)
        log_pdfs = []

        K = len(self.weights)
        for k in range(K):
            diff = x - self.means[k]
            inv_cov = np.linalg.inv(self.covariances[k])
            log_det = np.log(np.linalg.det(self.covariances[k]))
            quad = diff.T @ inv_cov @ diff
            log_pdf = -0.5 * (d * log_two_pi + log_det + quad)
            log_pdfs.append(log_pdf)

        log_pdfs = np.array(log_pdfs)
        max_log = np.max(log_weights + log_pdfs)
        log_sum_exp = max_log + np.log(np.sum(np.exp(log_weights + log_pdfs - max_log)))

        return -log_sum_exp

    def grad_U(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Gradient of the unnormalised log density ∇ log f(x) for a N-component 2D Gaussian mixture
        Args:
            x (np.ndarray): Input 2D point
        Returns:
            np.ndarray: Gradient of the unnormalised log density ∇ log f(x)
        """
        x = np.asarray(x)
        d = 2

        log_weights = np.log(self.grad_weights)
        log_two_pi = np.log(2 * np.pi)

        log_pdfs = np.empty(len(self.grad_weights))
        inv_covs = []

        K = len(self.weights)

        for k in range(K):
            diff = x - self.grad_means[k]
            inv_cov = np.linalg.inv(self.grad_covariances[k])
            inv_covs.append(inv_cov)
            log_det = np.log(np.linalg.det(self.grad_covariances[k]))
            quad = diff.T @ inv_cov @ diff
            log_pdf = -0.5 * (d * log_two_pi + log_det + quad)
            log_pdfs[k] = log_pdf

        log_combined = log_weights + log_pdfs
        max_log = np.max(log_combined)
        weights_norm = np.exp(log_combined - max_log)
        norm = np.sum(weights_norm)
        r = weights_norm / norm
        grad = np.zeros_like(x, dtype=float)

        for k in range(K):
            grad += r[k] * (inv_covs[k] @ (self.grad_means[k] - x))

        return -grad


class NealsFunnel(Density):
    """
    Implementation of Neal's Funnel distribution for HMC testing.

    The distribution is defined as:
    y ~ Normal(0, sigma_y^2)
    x_i ~ Normal(0, exp(y)) for i in 1..D-1

    Total dimension = D.
    q[0] corresponds to 'y' (the log-variance controller).
    q[1:] corresponds to 'x' (the vector dependent on y).
    """

    def __init__(self, dim=10, sigma_y=3.0):
        self.dim = dim
        self.sigma_y = sigma_y

    def __call__(self, q):
        """
        Probability Density Function p(q)
        """
        y = q[0]
        x = q[1:]

        p_y = (1 / (np.sqrt(2 * np.pi) * self.sigma_y)) * np.exp(
            -0.5 * (y**2) / (self.sigma_y**2)
        )

        p_x_given_y = np.prod(
            (1 / np.sqrt(2 * np.pi * np.exp(y))) * np.exp(-0.5 * (x**2) / np.exp(y))
        )

        return p_y * p_x_given_y

    def U(self, q):
        """
        Potential Energy Function U(q) = -log(p(q))
        """
        y = q[0]
        x = q[1:]

        u_y = (y**2) / (2 * self.sigma_y**2)

        u_x_det = (self.dim - 1) * y / 2.0
        u_x_quad = 0.5 * np.sum(x**2) * np.exp(-y)

        return u_y + u_x_det + u_x_quad

    def grad_U(self, q):
        """
        Gradient of Potential Energy with respect to q.
        Returns numpy array of shape (dim,)
        """
        y = q[0]
        x = q[1:]
        grad = np.zeros_like(q)

        exp_neg_y = np.exp(-y)

        grad[1:] = x * exp_neg_y

        dy_prior = y / (self.sigma_y**2)

        dy_det = (self.dim - 1) / 2.0

        dy_quad = -0.5 * np.sum(x**2) * exp_neg_y

        grad[0] = dy_prior + dy_det + dy_quad

        return grad
