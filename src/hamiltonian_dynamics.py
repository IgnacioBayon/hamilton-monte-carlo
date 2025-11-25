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
    return -1 * (grad_f(x) / f(x))


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
    x: np.ndarray, p: np.ndarray, M: np.ndarray, L: int, f: Callable
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

    hamiltonian_L = H(x[L], p[L], M, f)
    hamiltonian_0 = H(x[0], p[0], M, f)

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
        p_halfstep = p[t] - (step_size / 2) * grad_U(
            x,
        )
        x_new = x[t] + step_size * np.linalg.inv(M) * p_halfstep
        p_new = p_halfstep - (step_size / 2) * grad_U(x)

    return x_new, p_new
