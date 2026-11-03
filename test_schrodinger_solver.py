import numpy as np
import pytest

from schrodinger_solver import (
    infinite_square_well_energy,
    integrate_wavefunction,
    secant_root,
    solve_eigenenergy,
)


def test_rk4_free_particle():
    x_grid = np.linspace(0.0, 1.0, 51)
    wavefunction = integrate_wavefunction(0.0, x_grid, lambda x: 0.0)
    np.testing.assert_allclose(wavefunction, x_grid, atol=1e-14)


def test_secant_root():
    root = secant_root(lambda x: x**2 - 2.0, 1.0, 2.0, 1e-12)
    assert root == pytest.approx(np.sqrt(2.0), abs=1e-12)


def test_ground_state():
    width = 6e-9
    x_grid = np.linspace(-width / 2, width / 2, 2400)
    numerical = solve_eigenenergy(
        x_grid, lambda x: 0.0, (1e-21, 2e-21), 1e-35,
    )
    analytical = infinite_square_well_energy(1, width)
    assert abs(numerical - analytical) / analytical < 1e-12


def test_grid_convergence():
    # From my coursework stress test; coarser grids and tighter energy tolerance.
    exact = infinite_square_well_energy(1, 6e-9)
    grid_sizes = [300, 600, 1200]
    errors = []
    for points in grid_sizes:
        grid = np.linspace(-3e-9, 3e-9, points)
        energy = solve_eigenenergy(grid, lambda x: 0.0, (1e-21, 2e-21), 1e-35)
        errors.append(abs(energy - exact) / exact)
    for i in range(2):
        step_ratio = (grid_sizes[i + 1] - 1) / (grid_sizes[i] - 1)
        order = np.log(errors[i] / errors[i + 1]) / np.log(step_ratio)
        assert order == pytest.approx(4.0, abs=0.15)


def test_energy_tolerance():
    grid = np.linspace(-3e-9, 3e-9, 1200)
    exact = infinite_square_well_energy(1, 6e-9)
    loose = solve_eigenenergy(grid, lambda x: 0.0, (1e-21, 2e-21), 1e-25)
    tight = solve_eigenenergy(grid, lambda x: 0.0, (1e-21, 2e-21), 1e-35)

    assert abs(tight - exact) < abs(loose - exact) / 100


def test_bad_grid():
    with pytest.raises(ValueError):
        integrate_wavefunction(
            1.0, np.array([0.0, 0.5, 0.5]), lambda x: 0.0,
        )


def test_flat_residual():
    with pytest.raises(ZeroDivisionError):
        secant_root(lambda x: 1.0, 0.0, 1.0, 1e-6)
