from collections.abc import Callable

import numpy as np

FluxMap = Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray]]


class FluxValues:
    """
    Centralized registry for machine flux data.
    Handles both constant flux vectors and (eventually) dynamic
    operating-point-dependent maps. Each machine is registered by name.
    """

    def __init__(self) -> None:
        self._registry: dict[str, FluxMap | tuple[np.ndarray, np.ndarray]] = {}
        ieee_flux_volt = np.array([0.0115, 0.0018, 0.0, 0.0])
        ieee_flux_torq = np.array(
            [1.12810358e-02, -6.28421072e-04, 1.55053034e-04, -4.81016476e-05]
        )
        self.register_constant("IEEEMachine2", ieee_flux_volt, ieee_flux_torq)

    def register_constant(
        self, machine_name: str, flux_volt: np.ndarray, flux_torq: np.ndarray
    ) -> None:
        """Registers a machine with constant flux vectors."""
        self._registry[machine_name] = (
            np.asarray(flux_volt, dtype=float),
            np.asarray(flux_torq, dtype=float),
        )

    def register_map(self, machine_name: str, file_path: str) -> None:
        """
        Registers a dynamic, operating-point-dependent flux map for a machine.

        Placeholder for future support. Currently raises immediately so callers
        cannot accidentally register a silently-broken map and discover the
        breakage later at get_flux time.
        """
        raise NotImplementedError(
            "Dynamic flux maps are not yet implemented. "
            "Use register_constant for constant-flux machines."
        )

    def get_flux(
        self, machine_name: str, omega: float, vec_curr_dq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (flux_volt_vec, flux_torq_vec) for a given machine.

        Constant entries are copied before return so callers may mutate the
        result without corrupting the registry; dynamic map callables are
        expected to return fresh arrays each call.
        """
        if machine_name not in self._registry:
            raise ValueError(f"No flux data registered for machine: '{machine_name}'")

        data = self._registry[machine_name]

        if isinstance(data, tuple):
            return data[0].copy(), data[1].copy()

        if callable(data):
            return data(omega, vec_curr_dq)

        raise TypeError(f"Invalid flux data format in registry for '{machine_name}'")
