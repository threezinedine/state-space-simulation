import unittest
from unittest.mock import Mock
import numpy as np
from parameterized import parameterized
from models.plants import Plant


DTYPE = np.float32


class TestPlant(unittest.TestCase):
    @parameterized.expand([
            [np.array([2, 2, 1.3], dtype=DTYPE), np.array([1, 1, .5], dtype=DTYPE)] 
        ])
    def test_run_function(self, input_data, initial_state):
        plant = Plant(initial_state=initial_state, dtype=DTYPE)

        output = plant.run(input_data)

        self.assertNotEqual(output.tolist(), [0., 0., 0.])
