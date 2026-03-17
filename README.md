# Hamiltonian Monte Carlo 🚀

A compact implementation of **Hamiltonian Monte Carlo (HMC)** for sampling from complex probability distributions, with simple experiments and visualizations.

---

## ✨ Project Info

This project explores **Hamiltonian Monte Carlo**, a Markov Chain Monte Carlo method that uses gradient information to propose efficient moves.

Instead of relying on random-walk proposals, HMC introduces an auxiliary **momentum** variable and simulates a particle moving through the probability landscape. The method defines a **Hamiltonian**:

$$
H(q, p) = U(q) + K(p)
$$

where:

* $U(q) = -\log \pi(q)$ represents the target distribution (potential energy)
* $K(p)$ is a kinetic energy term, typically Gaussian

By approximately simulating Hamiltonian dynamics (using the leapfrog integrator), HMC generates proposals that follow the geometry of the distribution, allowing larger and more efficient moves while maintaining a high acceptance rate.

---

## 📁 Project structure

```
src/            → core samplers and target distributions  
notebooks/      → experiments and visualizations  
report-latex/   → slides explaining the method
```

---

## 📓 Notebooks

* **gaussian_mixture.ipynb**
  Visualizes sampling on a simple 2D distribution.

* **neals_funnel.ipynb**
  Tests HMC on a difficult, highly non-linear distribution.

---

## ⚙️ Setup (with uv)

This project uses **uv** for fast dependency management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

---

## 🧠 Why HMC?

* Avoids inefficient random-walk behavior by proposing informed moves
* Uses gradient information to follow the shape of the distribution
* Produces less correlated samples, improving statistical efficiency
* Scales better to higher-dimensional problems than basic MCMC methods

---

## 📌 Summary

* Lightweight HMC implementation
* Includes challenging test distributions
* Best explored through the notebooks

---
