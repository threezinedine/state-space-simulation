import numpy as np
from .signals import ISignal
from .plants import IPlant, Plant


DTYPE = np.float32


class Model:
    def __init__(self, plant:IPlant, initial_state:np.ndarray=None, dtype=DTYPE):
        self._dtype = dtype
        self._plant = plant
        self._state = self._generate_zero_intitial_states(initial_state)

    def _generate_zero_intitial_states(self, initial_state:np.ndarray, dtype=DTYPE) -> np.ndarray:
        if initial_state is not None:
            return initial_state
        else:
            return np.zeros(shape=(1, self._plant.num_states), dtype=dtype)

    def _get_dtype(self, dtype) -> np.dtype:
        if dtype is not None:
            return dtype
        else:
            return self._dtype

    def _run_input_arr(self, input_arr:np.ndarray, dtype=None):
        dtype = self._get_dtype(dtype)
        result = []
        for input_data in input_arr:
            result.append(self._plant.run(input_data))
        return np.array(result, dtype=dtype).reshape(-1, self._plant.output_dim)

    def run(self, input_signal:ISignal, dtype=None):
        input_arr = input_signal.get_arr()
        dtype = self._get_dtype(dtype)
        output = self._run_input_arr(input_arr, dtype=dtype)
        return output
