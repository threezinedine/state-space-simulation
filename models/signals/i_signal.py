from abc import ABC, abstractmethod
import numpy as np


class ISignal(ABC):
    @abstractmethod
    def get_arr(self) -> np.ndarray:
        raise NotImplementedError()
