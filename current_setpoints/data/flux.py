from abc import ABC, abstractmethod

import numpy as np


class Flux(ABC):
    @abstractmethod
    def get_flux(self, omega, curr_dq) -> tuple[np.ndarray, np.ndarray]:
        pass


class ConstantFlux(Flux):
    def __init__(self, ieee_flux_volt, ieee_flux_torq):
        self.ieee_flux_volt = ieee_flux_volt
        self.ieee_flux_torq = ieee_flux_torq

    def get_flux(self, omega, curr_dq) -> tuple[np.ndarray, np.ndarray]:
        return self.ieee_flux_volt, self.ieee_flux_torq


class Flux_IEEEMachine2(ConstantFlux):
    def __init__(self):
        ieee_flux_volt = np.array([0.0115, 0.0018, 0.0, 0.0])
        ieee_flux_torq = np.array([1.12810358e-02, -6.28421072e-04, 1.55053034e-04, -4.81016476e-05])

        super().__init__(ieee_flux_volt, ieee_flux_torq)
