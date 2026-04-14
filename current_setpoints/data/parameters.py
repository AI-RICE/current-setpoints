from typing import Callable, Dict, Tuple, Union

import numpy as np


class FluxValues:
    """
    Centralized registry for machine flux data.
    Seamlessly handles both constant flux vectors and dynamic operating-point maps.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Union[Callable, Tuple[np.ndarray, np.ndarray]]] = {}
        ieee_flux_volt = np.array([0.0115, 0.0018, 0.0, 0.0])
        ieee_flux_torq = np.array(
            [1.12810358e-02, -6.28421072e-04, 1.55053034e-04, -4.81016476e-05]
        )
        self.register_constant("IEEEMachine2", ieee_flux_volt, ieee_flux_torq)

    def register_constant(
        self, machine_name: str, flux_volt: np.ndarray, flux_torq: np.ndarray
    ) -> None:
        """Registers a machine with constant flux vectors."""
        self._registry[machine_name] = (flux_volt, flux_torq)

    def register_map(self, machine_name: str, file_path: str) -> None:
        """Loads and registers a dynamic flux map from a disk file."""

        def dummy_map(
            omega: float, vec_curr_dq: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            raise NotImplementedError("Dynamic map evaluation is not yet implemented.")

        self._registry[machine_name] = dummy_map

    def get_flux(
        self, machine_name: str, omega: float, vec_curr_dq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (flux_volt_vec, flux_torq_vec) for a given machine.
        Works transparently whether the underlying data is constant or mapped.
        """
        if machine_name not in self._registry:
            raise ValueError(f"No flux data registered for machine: '{machine_name}'")

        data = self._registry[machine_name]

        if isinstance(data, tuple):
            return data[0].copy(), data[1].copy()

        if callable(data):
            return data(omega, vec_curr_dq)

        raise TypeError(f"Invalid flux data format in registry for '{machine_name}'")
