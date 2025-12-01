import numpy as np

from typing import Callable


class CustomGaussian:
    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ):
        self.weights = weights
        self.means = means
        self.covariances = covariances

    def log_prob_mixture(
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

        return log_sum_exp


def grad_log_prob_mixture(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Gradient of the unnormalised log density ∇ log f(x) for a N-component 2D Gaussian mixture

    Args:
        x (np.ndarray): Input 2D point
    Returns:
        np.ndarray: Gradient of the unnormalised log density ∇ log f(x)
    """
    x = np.asarray(x)
    d = 2

    log_weights = np.log(weights)
    log_two_pi = np.log(2 * np.pi)

    log_pdfs = np.empty(len(weights))
    inv_covs = []

    K = len(weights)

    for k in range(K):
        diff = x - means[k]
        inv_cov = np.linalg.inv(covariances[k])
        inv_covs.append(inv_cov)
        log_det = np.log(np.linalg.det(covariances[k]))
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
        grad += r[k] * (inv_covs[k] @ (means[k] - x))

    return grad


class HMCSampler:
    def __init__(self, U: Callable, grad_U: Callable):
        self.U = U
        self.grad_U = grad_U

    def leapfrog(
        self,
        x: np.ndarray,
        p: np.ndarray,
        leapfrog_steps: int,
        step_size: int,
        return_path: bool = True,
    ):
        """Use the leapfrog algorithm to compute the next sample (x*, p*)

        Args:
            x: Current sample
            p: Current momentums
            leapfrog_steps: Number of leapfrog steps
            step_size: Size of each step
            return_path: Whether to return the full path of samples. Defaults to True.
        Returns:
            x_new, p_new: New sample and momentums after leapfrog steps
            np.ndarray: Path of samples if return_path is True
        """

        x_new = np.asarray(x, dtype=float).copy()
        p_new = np.asarray(p, dtype=float).copy()

        if return_path:
            path = [x_new.copy()]

        p_new = p_new - (step_size / 2) * (-self.grad_U(x_new))
        for t in range(leapfrog_steps):
            x_new = x_new + step_size * p_new

            if t != leapfrog_steps - 1:
                p_new = p_new - step_size * (-self.grad_U(x_new))

            if return_path:
                path.append(x_new.copy())

        p_new = p_new - (step_size / 2) * (-self.grad_U(x_new))
        p_new = -p_new

        if return_path:
            return x_new, p_new, np.array(path)

        return x_new, p_new

    def sample(
        self,
        n_samples: int,
        leapfrog_steps: int,
        initial_x: np.array,
        step_size: float = 0.1,
        dim: int = 2,
    ) -> np.ndarray:
        """Basic hmc sampler for the class

        Args:
            n_samples (int): Number of samples to generate
            leapfrog_steps (int): Number of leapfrog steps to use
            initial_x (np.array): Initial point
            step_size (float, optional): Size of the leapfrog steps. Defaults to 0.1.
            dim (int, optional): Dimension. Defaults to 2.

        Returns:
            np.ndarray: Sampled points of the distribution defined by the class' arguments.
        """
        samples = np.zeros((n_samples, dim))
        x = np.asarray(initial_x, dtype=float).copy()

        accepted = 0
        for i in range(n_samples):
            p = np.random.normal(size=dim)

            current_U = -self.U(x)
            current_K = 0.5 * np.dot(p, p)

            x_prop, p_prop = self.leapfrog(
                x,
                p,
                leapfrog_steps,
                step_size,
                return_path=False,
            )

            proposed_U = -self.U(x_prop)
            proposed_K = 0.5 * np.dot(p_prop, p_prop)

            log_accept_ratio = current_U - proposed_U + current_K - proposed_K
            if np.log(np.random.rand()) < log_accept_ratio:
                x = x_prop
                accepted += 1
            samples[i] = x

        return samples, accepted / n_samples
