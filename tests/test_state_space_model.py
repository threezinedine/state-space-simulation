import pytest 
import unittest
from unittest.mock import Mock
import numpy as np
from models import Model
from models.signals import ISignal
from models.plants import IPlant, Plant


DTYPE = np.float32


class TestStateSpaceModel(unittest.TestCase):
    def test_zero_states_zero_inputs_1000_times_zeros_output(self):
        iterate_times = 1000

        signal = Mock(spec=ISignal)
        signal.get_arr.return_value = np.zeros(shape=(iterate_times, Plant.num_states), dtype=DTYPE)

        plant = Mock(spec=IPlant)
        plant.num_states = 3
        plant.output_dim = 3
        plant.run.return_value = np.zeros(shape=(1, Plant.output_dim), dtype=DTYPE)

        model = Model(plant)

        expected = np.zeros(shape=(iterate_times, Plant.output_dim), dtype=DTYPE)

        output = model.run(signal)

        assert isinstance(output, np.ndarray) 
        self.assertTupleEqual(output.shape, expected.shape)
        assert output.all() == expected.all()

    def test_non_zero_states_zero_inputs_1000_times_non_zeros_output(self):
        iterate_times = 1000

        signal = Mock(spec=ISignal)
        signal.get_arr.return_value = np.random.rand(iterate_times, Plant.num_states)

        plant = Mock(spec=IPlant)
        plant.num_states = 3
        plant.output_dim = 3
        plant.run.return_value = np.random.rand(1, Plant.output_dim)

        model = Model(plant)

        non_expected = np.zeros(shape=(iterate_times, Plant.output_dim), dtype=DTYPE)

        output = model.run(signal)

        assert isinstance(output, np.ndarray) 
        self.assertTupleEqual(output.shape, non_expected.shape)
        assert output.any() != non_expected.any()
