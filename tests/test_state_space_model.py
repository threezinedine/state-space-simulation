import pytest 
import unittest
from unittest.mock import Mock
import numpy as np
from models import Model
from models.signals import ISignal
from models.plants import IPlant, Plant
from parameterized import parameterized


DTYPE = np.float32


class TestStateSpaceModel(unittest.TestCase):
    @parameterized.expand([
            [1000, DTYPE, 3, 3],
            [1000, np.float32, 4, 3],
            [2200, np.int32, 4, 4],
            [1200, np.int32, 5, 4],
        ])
    def test_model_object_run_method_with_zeros_initial_state_zero_plant_return(self, iterate_times, 
            dtype, num_states, output_dim):
        signal = Mock(spec=ISignal)
        signal.get_arr.return_value = np.zeros(shape=(iterate_times, num_states), dtype=dtype)

        plant = Mock(spec=IPlant)
        plant.num_states = num_states
        plant.output_dim = output_dim 
        plant.run.return_value = np.zeros(shape=(1, plant.output_dim), dtype=dtype)
        
        expected = np.zeros(shape=(iterate_times, output_dim), dtype=dtype)

        model = Model(plant)

        output = model.run(signal, dtype=dtype)

        assert isinstance(output, np.ndarray) 
        assert output.dtype == dtype
        self.assertTupleEqual(output.shape, expected.shape)
        assert output.all() == expected.all()

    @parameterized.expand([
            [1000, DTYPE, 3, 3],
            [1000, np.float64, 4, 3],
            [2000, np.float16, 4, 4]
        ])
    def test_model_run_method_non_zero_initial_states(self, iterate_times, 
            dtype, num_states, output_dim):
        signal = Mock(spec=ISignal)
        signal.get_arr.return_value = np.random.rand(iterate_times, num_states)

        plant = Mock(spec=IPlant)
        plant.num_states = num_states
        plant.output_dim = output_dim 
        plant.run.return_value = np.random.rand(1, plant.output_dim)
        
        expected = np.zeros(shape=(iterate_times, output_dim), dtype=dtype)

        model = Model(plant)

        output = model.run(signal, dtype=dtype)

        assert isinstance(output, np.ndarray) 
        assert output.dtype == dtype
        self.assertTupleEqual(output.shape, expected.shape)
        assert output.any() != expected.any()
