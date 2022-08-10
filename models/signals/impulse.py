from .signal import Signal
import numpy as np
from ..timer import ITimer


DTYPE = np.float32


class Impulse(Signal):
    def __init__(self, timer:ITimer):
        Signal.__init__(self, timer)

    def get_input(self, dtype=DTYPE) -> np.ndarray:
        self._index += 1

        timer_input = self._timer.get_time_input().copy()
        output = timer_input.copy()

        for index, input_data in enumerate(timer_input):
            if input_data == 0:
                output[index] = 1.
            else:
                output[index] = 0.

        return output
