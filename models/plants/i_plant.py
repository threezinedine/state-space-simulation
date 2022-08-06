from abc import ABC, abstractmethod
import numpy as np


class IPlant(ABC):
    @abstractmethod
    def run(self, input_data:np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_current_state(self) -> np.ndarray:
        raise NotImplementedError()
