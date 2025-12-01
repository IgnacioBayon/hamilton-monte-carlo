import numpy as np

from src.density import Density


class HMCSampler:
    def __init__(self, density: Density):
        self.U = density.U
        self.grad_U = density.grad_U

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

        p_new = p_new - (step_size / 2) * self.grad_U(x_new)
        for t in range(leapfrog_steps):
            x_new = x_new + step_size * p_new

            if t != leapfrog_steps - 1:
                p_new = p_new - step_size * self.grad_U(x_new)

            if return_path:
                path.append(x_new.copy())

        p_new = p_new - (step_size / 2) * self.grad_U(x_new)
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

            current_U = self.U(x)
            current_K = 0.5 * np.dot(p, p)

            x_prop, p_prop = self.leapfrog(
                x,
                p,
                leapfrog_steps,
                step_size,
                return_path=False,
            )

            proposed_U = self.U(x_prop)
            proposed_K = 0.5 * np.dot(p_prop, p_prop)

            log_accept_ratio = current_U - proposed_U + current_K - proposed_K
            if np.log(np.random.rand()) < log_accept_ratio:
                x = x_prop
                accepted += 1
            samples[i] = x

        return samples, accepted / n_samples
