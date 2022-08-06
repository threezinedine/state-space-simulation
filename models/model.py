import numpy as np
from .signals import ISignal, IFeedBack
from .plants import IPlant, Plant


DTYPE = np.float32


class Model:
    def __init__(self, plant:IPlant, initial_state:np.ndarray=None, dtype=DTYPE):
        self._dtype = dtype
        self._plant = plant

    def _get_dtype(self, dtype) -> np.dtype:
        if dtype is not None:
            return dtype
        else:
            return self._dtype

    def run(self, input_signal:ISignal, dtype=None):
        dtype = self._get_dtype(dtype)

        result = []

        while not input_signal.is_finished():
            input_data = input_signal.get_input()
            result.append(self._plant.run(input_data))
        return np.array(result, dtype=dtype).reshape(-1, self._plant.output_dim)
