from abc import ABC, abstractmethod

import numpy as np


class Flux(ABC):
    """
    Abstract interface for flux linkage providers. Returns one vector for the
    back-EMF (voltage) equation and one for the torque equation, both in the
    DQ frame, ordered ``[d_1, q_1, d_3, q_3, ...]``.
    """

    @abstractmethod
    def get_flux(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns ``(flux_volt, flux_torq)`` for the given operating point.

        Args:
            omega: Electrical speed [rad/s].
            curr_dq: DQ current vector, length ``n_phases - 1``.
        """
        pass


class ConstantFlux(Flux):
    """Flux provider returning fixed vectors, independent of speed and current."""

    def __init__(self, ieee_flux_volt: np.ndarray, ieee_flux_torq: np.ndarray) -> None:
        self.ieee_flux_volt: np.ndarray = ieee_flux_volt
        self.ieee_flux_torq: np.ndarray = ieee_flux_torq

    def get_flux(self, omega: float, curr_dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.ieee_flux_volt, self.ieee_flux_torq


class Flux_IEEEMachine2(ConstantFlux):
    """Identified flux vectors for the 5-phase IEEE machine 2."""

    def __init__(self) -> None:
        ieee_flux_volt = np.array([0.0115, 0.0018, 0.0, 0.0])
        ieee_flux_torq = np.array([1.18255974e-02, -1.36757644e-03, 8.94095382e-05, -4.58552615e-05])

        super().__init__(ieee_flux_volt, ieee_flux_torq)
