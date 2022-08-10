from abc import ABC, abstractmethod
import numpy as np


class ITimer(ABC):
    @abstractmethod
    def get_length(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_timer_input(self) -> np.ndarray:
        raise NotImplementedError()
