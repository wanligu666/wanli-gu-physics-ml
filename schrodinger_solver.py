"""RK4 shooting method for a particle between two hard walls.

Based on my second-year computational-physics assignment. Positions are in
metres and energies in joules. Run this file to compare grid and secant errors.
"""

import numpy as np

HBAR = 1.054571817e-34  # J s
ELECTRON_MASS = 9.1093837015e-31  # kg


def schrodinger_rhs(
    x, state, energy, potential, mass=ELECTRON_MASS, hbar=HBAR,
):
    """Use state = [psi, dpsi/dx] to write the equation as two first-order ODEs."""
    psi, phi = state
    return np.array([phi, (2 * mass / hbar**2) * (potential(x) - energy) * psi])


def integrate_wavefunction(
    energy, x_grid, potential, mass=ELECTRON_MASS, hbar=HBAR,
    initial_derivative=1.0,
):
    """Propagate a trial wavefunction from the left wall using RK4.

    psi starts at zero. A non-zero initial slope sets its arbitrary scale;
    normalisation can be done after finding the energy.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or len(x_grid) < 2:
        raise ValueError("x_grid must be a one-dimensional array with at least two points")
    if not np.all(np.isfinite(x_grid)) or np.any(np.diff(x_grid) <= 0):
        raise ValueError("x_grid must contain finite, strictly increasing points")
    if not np.isfinite(initial_derivative) or initial_derivative == 0:
        raise ValueError("The initial slope must be finite and non-zero")

    values = np.empty_like(x_grid)
    state = np.array([0.0, initial_derivative], dtype=float)
    values[0] = state[0]

    for index in range(len(x_grid) - 1):
        x = x_grid[index]
        step = x_grid[index + 1] - x
        k1 = step * schrodinger_rhs(x, state, energy, potential, mass, hbar)
        k2 = step * schrodinger_rhs(
            x + 0.5 * step, state + 0.5 * k1, energy, potential, mass, hbar
        )
        k3 = step * schrodinger_rhs(
            x + 0.5 * step, state + 0.5 * k2, energy, potential, mass, hbar
        )
        k4 = step * schrodinger_rhs(
            x + step, state + k3, energy, potential, mass, hbar
        )
        state = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        values[index + 1] = state[0]

    return values


def shooting_residual(
    energy, x_grid, potential, mass=ELECTRON_MASS, hbar=HBAR,
):
    """Return psi at the right wall; an eigenenergy makes this zero."""
    return float(integrate_wavefunction(energy, x_grid, potential, mass, hbar)[-1])


def secant_root(
    function, first_guess, second_guess, tolerance, max_iterations=200,
):
    """Stop when successive estimates differ by at most tolerance.

    For energy roots, tolerance is an absolute change in joules. It is not
    a bound on the true energy error, and it can hide grid convergence.
    """
    x0, x1 = float(first_guess), float(second_guess)
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Use a finite, positive tolerance")

    f0, f1 = float(function(x0)), float(function(x1))
    for _ in range(max_iterations):
        if f1 == 0:
            return x1
        denominator = f1 - f0
        if denominator == 0:
            raise ZeroDivisionError("invalid slope: secant values are equal")
        x2 = x1 - f1 * (x1 - x0) / denominator
        if not np.isfinite(x2):
            raise RuntimeError("Secant iteration diverged")
        if abs(x2 - x1) <= tolerance:
            return x2
        f2 = float(function(x2))
        x0, x1, f0, f1 = x1, x2, f1, f2

    raise RuntimeError("Secant iteration did not converge")


def solve_eigenenergy(
    x_grid, potential, initial_guesses, tolerance,
    mass=ELECTRON_MASS, hbar=HBAR, max_iterations=200,
):
    """Adjust the trial energy until the right-wall boundary condition is met."""
    def residual(energy):
        return shooting_residual(energy, x_grid, potential, mass, hbar)

    return secant_root(
        residual, initial_guesses[0], initial_guesses[1], tolerance, max_iterations
    )


def infinite_square_well_energy(
    quantum_number, width, mass=ELECTRON_MASS, hbar=HBAR,
):
    """Return the analytical eigenenergy for an infinite square well."""
    if quantum_number < 1 or width <= 0:
        raise ValueError("Use n >= 1 and a positive well width")
    return quantum_number**2 * np.pi**2 * hbar**2 / (2.0 * mass * width**2)


if __name__ == "__main__":
    width = 6e-9
    exact_energy = infinite_square_well_energy(1, width)

    def zero_potential(x):
        return 0.0

    # Tighten the energy iteration separately to expose the spatial error.
    print("Relative energy error for a 6 nm infinite square well")
    print("N       tol=1e-25 J     tol=1e-35 J")
    for points in [300, 600, 1200, 1800, 2400]:
        grid = np.linspace(-width / 2, width / 2, points)
        errors = []
        for tolerance in [1e-25, 1e-35]:
            energy = solve_eigenenergy(
                grid, zero_potential, (1e-21, 2e-21), tolerance,
            )
            errors.append(abs(energy - exact_energy) / exact_energy)
        print(f"{points:<6}  {errors[0]:.6e}   {errors[1]:.6e}")
