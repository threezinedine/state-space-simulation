from .signal import Signal
import numpy as np
from ..timer import ITimer


class Impulse(Signal):
    def __init__(self, timer:ITimer):
        Signal.__init__(self, timer)

    def get_input(self) -> np.ndarray:
        self._index += 1
        return 0.
