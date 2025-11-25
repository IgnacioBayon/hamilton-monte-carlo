import numpy as np


def U(x: np.ndarray):
    """Compute the potential energy for a target

    Args:
        x: Target
    """
    return -np.log(x)


def grad_U(x: np.ndarray):
    """Compute the gradient of the potential energy

    Args:
        x: Target
    """
    return -1 / x


def kinetic_energy(M: np.ndarray, p: np.ndarray):
    """Compute the kinetic energy of a target given its momentum

    Args:
        M: Mass matrix
        p: Momentums
    """
    return 0.5 * p.dot(np.linalg.inv(M).dot(p))


def H(x: np.ndarray, p: np.ndarray, M: np.ndarray):
    """Compute the hamiltonian of a target

    Args:
        x: Target
        p: Momentums
        M: Mass matrix
    """
    return U(x) + kinetic_energy(M, p)


def compute_alpha(x: np.ndarray, p: np.ndarray, M: np.ndarray, L: int) -> float:
    """Compute the acceptance ratio for the next sample

    Args:
        x: Current sample
        p: Momentums
        M: Mass matrix
        L: Number of leapfrog steps

    Returns:
        float: acceptance ratio of next sample
    """

    hamiltonian_L = H(x[L], p[L], M)
    hamiltonian_0 = H(x[0], p[0], M)

    numerator = np.exp(-hamiltonian_L)
    denominator = np.exp(-hamiltonian_0)

    alpha = min(1, numerator / denominator)

    return alpha


def leapfrog(x: np.ndarray, p: np.ndarray, M: np.ndarray, L: int, step_size: int):
    """Use the leapfrog algorithm to compute the next sample (x*, p*)

    Args:
        x: Current sample
        p: Current momentums
        M: Mass matrix
        L: Number of leapfrog steps
        step_size: Size of each step
    """

    x_new = x.copy()
    p_new = x.copy()

    for t in range(L):
        p_halfstep = p[t] - (step_size / 2) * grad_U(x)
        x_new = x[t] + step_size * np.linalg.inv(M) * p_halfstep
        p_new = p_halfstep - (step_size / 2) * grad_U(x)

    return x_new, p_new
