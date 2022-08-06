from .i_signal import ISignal
import numpy as np


class Impulse(ISignal):
    def __init__(self, timer):
        self._timer = timer

    def get_arr(self) -> np.ndarray:
        return np.array([[0.]])
