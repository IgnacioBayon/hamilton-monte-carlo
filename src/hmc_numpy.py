import numpy as np

from typing import Callable


def U(x: np.ndarray, f: Callable) -> float:
    """Compute the potential energy for a target

    Args:
        x: Target
        f: Function we want to sample from

    Returns:
        float: Potential energy
    """
    return -np.log(f(x))


def grad_U(x: np.ndarray, f: Callable, grad_f: Callable) -> float:
    """Compute the gradient of the potential energy

    Args:
        x: Target
        f: Function we want to get the probability distribution from
        grad_f: Gradient of the function we want to sample from

    Returns:
        float: Gradient of the potential energy
    """
    return grad_f(x) / f(x)


def kinetic_energy(M: np.ndarray, p: np.ndarray) -> float:
    """Compute the kinetic energy of a target given its momentum

    Args:
        M: Mass matrix
        p: Momentums
    """
    return 0.5 * p.dot(np.linalg.inv(M).dot(p))  # type: ignore


def H(x: np.ndarray, p: np.ndarray, M: np.ndarray, f: Callable) -> float:
    """Compute the hamiltonian of a target

    Args:
        x: Target
        p: Momentums
        M: Mass matrix
        f: Function we want to sample from
    """
    return U(x, f) + kinetic_energy(M, p)


def compute_alpha(
    x_L: np.ndarray,
    p_L: np.ndarray,
    x_0: np.ndarray,
    p_0: np.ndarray,
    M: np.ndarray,
    L: int,
    f: Callable,
) -> float:
    """Compute the acceptance ratio for the next sample

    Args:
        x: Current sample
        p: Momentums
        M: Mass matrix
        L: Number of leapfrog steps
        f: Function we want to sample from

    Returns:
        float: acceptance ratio of next sample
    """

    hamiltonian_L = H(x_L, p_L, M, f)
    hamiltonian_0 = H(x_0, p_0, M, f)

    numerator = np.exp(-hamiltonian_L)
    denominator = np.exp(-hamiltonian_0)

    alpha = min(1, numerator / denominator)

    return alpha


def leapfrog(
    x: np.ndarray,
    p: np.ndarray,
    M: np.ndarray,
    L: int,
    step_size: int,
    f: Callable,
    grad_f: Callable,
):
    """Use the leapfrog algorithm to compute the next sample (x*, p*)

    Args:
        x: Current sample
        p: Current momentums
        M: Mass matrix
        L: Number of leapfrog steps
        step_size: Size of each step
        f: Function we want to sample from
        grad_f: Gradient of the function we want to sample from
    """

    x_new = x.copy()
    p_new = x.copy()

    for t in range(L):
        p_halfstep = p - (step_size / 2) * grad_U(x, f, grad_f)
        x_new = x + step_size * np.linalg.inv(M).dot(p_halfstep)
        p_new = p_halfstep - (step_size / 2) * grad_U(x, f, grad_f)

    return x_new, p_new


def hamiltonian_sample(
    x: np.ndarray,
    f: Callable,
    grad_f: Callable,
    num_samples: int,
    L: int = 100,
    step_size: int = 1,
    mass=1,
) -> np.ndarray:
    """Sample points using hamiltonian-montecarlo

    Args:
        x: Initial point
        f: Function you want to sample from
        grad_f: Gradient of the function you want to sample from
        num_samples: Number of samples to generate
        L: Number of leapfrog steps
        step_size: Size of each timestep

    Returns:
        np.ndarray: Array of the sampled points
    """

    M = mass * np.eye(2)  # initial value of mass matrix
    # p = np.random.multivariate_normal(
    #     mean=np.zeros(2), cov=M
    # )  # initial value (sample random momentum)
    p = np.zeros(2)

    samples = np.empty((0, 2))

    for i in range(num_samples):
        x_new, p_new = leapfrog(x, p, M, L, step_size, f, grad_f)

        acceptance_rate = compute_alpha(x_new, p_new, x, p, M, L, f)

        if np.random.random() < acceptance_rate:
            samples = np.vstack([samples, x_new])

            x = x_new
            p = p_new
        else:
            samples = np.vstack([samples, x])

    return samples
