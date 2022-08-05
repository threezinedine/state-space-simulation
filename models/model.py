import numpy as np
from .signals import ISignal


DTYPE = np.float32


class Model:
    num_states = 3
    output_dim = 3

    def __init__(self, initial_state:np.ndarray):
        self._initial_state = initial_state

    def run_one_input(self, input_data:np.ndarray, matrix_A:np.ndarray, 
            matrix_B:np.ndarray, matrix_C:np.ndarray, dtype=DTYPE):
        return np.zeros(shape=(1, self.output_dim), dtype=DTYPE)

    def run(self, input_arr:np.ndarray, dtype=DTYPE):
        output = []
        for input_data in input_arr:
            output.append(self._run_one_input(input_data))

        return np.array(output, dtype=DTYPE)

    def run(self, input_signal:ISignal, dtype=DTYPE):
        input_arr = input_signal.get_arr()
        output = self._run(input_arr)

        return DTYPE(output)
