import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.io import savemat


def _spectral_rhs(
    time: float,
    solution: NDArray[np.float64],
    *,
    wave_numbers: NDArray[np.float64],
    viscosity: float,
) -> NDArray[np.float64]:
    """Evaluate the periodic viscous Burgers method-of-lines system.

    The nonlinear conservative flux and diffusion are differentiated spectrally without dealiasing, matching the
    stored numerical benchmark.

    Args:
        time: Current integration time, unused because the equation is autonomous.
        solution: Scalar solution on unique equispaced periodic nodes.
        wave_numbers: Fourier angular wave numbers aligned with `solution`.
        viscosity: Positive kinematic-viscosity coefficient.

    Returns:
        Time derivative at every unique periodic node.
    """
    del time
    solution_spectrum = np.fft.fft(solution)
    diffusion = np.fft.ifft(-(wave_numbers**2) * solution_spectrum).real
    flux_spectrum = np.fft.fft(0.5 * solution**2)
    convection = np.fft.ifft(1j * wave_numbers * flux_spectrum).real
    return viscosity * diffusion - convection


def generate_burgers_solution(
    *,
    unique_space_points: int = 511,
    time_steps: int = 200,
    viscosity: float = 0.01 / np.pi,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
    max_step: float = 1.0e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate the periodic viscous Burgers benchmark from scratch.

    The solver adapts `gen_burgers.m` to Python. It integrates `u_t + u*u_x - nu*u_xx = 0` on `[-1, 1]` from
    `u(0, x) = -sin(pi*x)` using a Fourier pseudo-spectral spatial discretization and SciPy's eighth-order explicit
    `DOP853` integrator. The repeated right endpoint is appended only after integration.

    Args:
        unique_space_points: Number of unique equispaced periodic nodes. The saved grid has one additional repeated
            endpoint.
        time_steps: Number of uniform temporal intervals. The saved grid contains `time_steps + 1` values.
        viscosity: Positive kinematic-viscosity coefficient.
        relative_tolerance: Positive relative integration tolerance.
        absolute_tolerance: Positive absolute integration tolerance.
        max_step: Positive maximum adaptive integration step.

    Returns:
        Solution with layout `[time, space]`, time coordinates, and spatial coordinates including both endpoints.

    Raises:
        RuntimeError: If SciPy's adaptive integration fails.
        ValueError: If grid sizes, viscosity, tolerances, or maximum step are invalid.
    """
    if unique_space_points < 3 or time_steps < 1:
        raise ValueError("Burgers generation requires at least three spatial points and one temporal step.")
    positive_values = {
        "viscosity": viscosity,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "max_step": max_step,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in positive_values.values()):
        raise ValueError("Burgers viscosity, tolerances, and maximum step must be finite and positive.")

    times = np.linspace(0.0, 1.0, time_steps + 1, dtype=np.float64)
    unique_positions = np.linspace(-1.0, 1.0, unique_space_points, endpoint=False, dtype=np.float64)
    spacing = 2.0 / unique_space_points
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(unique_space_points, d=spacing)
    initial_solution = -np.sin(np.pi * unique_positions)
    right_hand_side = partial(_spectral_rhs, wave_numbers=wave_numbers, viscosity=viscosity)
    integration = solve_ivp(
        right_hand_side,
        (times[0], times[-1]),
        initial_solution,
        method="DOP853",
        t_eval=times,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=max_step,
    )
    if not integration.success:
        raise RuntimeError(f"Burgers integration failed: {integration.message}")
    unique_solution = np.asarray(integration.y.T, dtype=np.float64)
    solution = np.concatenate((unique_solution, unique_solution[:, :1]), axis=1)
    positions = np.linspace(-1.0, 1.0, unique_space_points + 1, dtype=np.float64)
    return solution, times, positions


def save_burgers_dataset(destination: str | Path = "data/burgers/burgers.mat") -> Path:
    """Generate and atomically save the default Burgers benchmark as a MATLAB artifact.

    Args:
        destination: Output `.mat` path containing `usol`, `t`, `x`, and `nu` arrays.

    Returns:
        Absolute path to the generated dataset.
    """
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    solution, times, positions = generate_burgers_solution()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="burgers-", suffix=".mat", dir=path.parent, delete=False) as file:
            temporary_path = Path(file.name)
        savemat(
            temporary_path,
            {
                "usol": solution,
                "t": times[None, :],
                "x": positions[None, :],
                "nu": np.asarray([[0.01 / np.pi]], dtype=np.float64),
            },
            do_compression=True,
        )
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def ensure_burgers_dataset(destination: str | Path = "data/burgers/burgers.mat") -> Path:
    """Return an existing Burgers dataset or generate it when absent.

    Args:
        destination: Expected local `burgers.mat` path.

    Returns:
        Absolute path to the existing or newly generated dataset.
    """
    path = Path(destination).expanduser().resolve()
    return path if path.is_file() else save_burgers_dataset(path)
