from abc import ABC, abstractmethod
import numpy as np


class IPlant(ABC):
    @abstractmethod
    def run(self, input_data:np.ndarray) -> np.ndarray:
        raise NotImplementedError()
