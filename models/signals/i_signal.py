from abc import ABC, abstractmethod
import numpy as np


class ISignal(ABC):
    @abstractmethod
    def get_input(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def is_finished(self) -> bool:
        raise NotImplementedError()
