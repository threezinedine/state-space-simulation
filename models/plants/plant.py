from .i_plant import IPlant
from ..signals import IFeedBack
import numpy as np


DTYPE = np.float32


class Plant(IPlant):
    num_states = 3
    output_dim = 3

    def __init__(self, initial_state:np.ndarray=None, dtype=DTYPE):
        if initial_state is not None:
            self._state = state
        else:
            self._state = np.zeros(shape=(1, self.num_states), dtype=dtype)

    def run(self, input_data:np.ndarray, dtype=DTYPE) -> np.ndarray:
        return np.zeros(shape=(1, self.num_states), dtype=DTYPE)

    def get_current_state(self) -> np.ndarray:
        return self._state

    def get_feedback(self) -> np.ndarray:
        return self._state
