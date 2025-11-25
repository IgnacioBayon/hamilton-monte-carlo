import torch


# -----------------------------
# 1. Potential function U
# -----------------------------
def U(theta: torch.Tensor, f: callable) -> torch.Tensor:
    """
    Rosenbrock potential (negative log-probability up to constant).
    Args:
        theta (torch.tensor): target position
        f (callable): target density function
    Returns:
        torch.tensor of the potential energy
    """
    return -torch.log(f(theta))


# -----------------------------
# 2. Leapfrog integrator
# -----------------------------
def leapfrog(
    theta: torch.Tensor, p: torch.Tensor, f: callable, step_size: float, n_steps: int
):
    """
    One leapfrog trajectory.
    Args:
        theta (torch.tensor): initial position
        p (torch.tensor): initial momentum
        f (callable): target density function
        step_size (float): leapfrog step size
        n_steps (int): number of leapfrog steps
    Returns:
        torch.tensor, torch.tensor: new position and momentum after leapfrog
    """

    # First half momentum update
    theta = theta.clone().detach().requires_grad_(True)
    U(theta, f).backward()
    p = p - 0.5 * step_size * theta.grad
    theta = theta.detach()  # Clear graph

    # Full steps
    for _ in range(n_steps):
        # Position update
        theta = (theta + step_size * p).clone().detach().requires_grad_(True)

        # Momentum update (skip gradient on last iteration)
        U(theta, f).backward()
        p = p - step_size * theta.grad
        theta = theta.detach()

    # Final position update
    theta = (theta + step_size * p).clone().detach().requires_grad_(True)

    # Final half momentum update
    theta.requires_grad_(True)
    U(theta, f).backward()
    p = p - 0.5 * step_size * theta.grad
    theta = theta.detach()

    return theta, p


# -----------------------------
# 3. HMC sampler
# -----------------------------
def hmc(
    initial_theta: torch.Tensor,
    f: callable,
    n_samples: int,
    step_size: float = 0.01,
    n_steps: int = 20,
    mass: float = 1.0,
):
    samples = []
    theta = torch.tensor(initial_theta, dtype=torch.float32)

    for _ in range(n_samples):
        # Sample fresh momentum
        p0 = torch.randn_like(theta) * mass

        # Simulate trajectory
        theta_prop, p_prop = leapfrog(theta, p0.clone(), f, step_size, n_steps)

        # Compute Hamiltonians for acceptance test
        H_initial = U(theta, f) + 0.5 * torch.dot(p0, p0)
        H_prop = U(theta_prop, f) + 0.5 * torch.dot(p_prop, p_prop)

        # Accept/reject
        accept_prob = torch.exp(H_initial - H_prop)
        # We do not need to compute min(1, accept_prob) because if accept_prob > 1,
        # the condition torch.rand(1) < accept_prob will always be true (same as
        # with accept_prob = 1)
        if torch.rand(1) < accept_prob:
            theta = theta_prop

        samples.append(theta.clone())

    return torch.stack(samples)
