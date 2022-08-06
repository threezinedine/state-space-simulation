from .i_plant import IPlant
import numpy as np


DTYPE = np.float32


class Plant(IPlant):
    num_states = 3
    output_dim = 3

    def run(self, input_data:np.ndarray, dtype=DTYPE) -> np.ndarray:
        return np.zeros(shape=(1, self.num_states), dtype=DTYPE)
