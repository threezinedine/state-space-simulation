import pytest 
import unittest
from unittest.mock import Mock
import numpy as np
from models import Model
from models.signals import ISignal


DTYPE = np.float32



class TestStateSpaceModel(unittest.TestCase):
    def test_open_loop_with_zeros_inputs_and_zeros_initial_states_full_state_input_1000_iterate_times(self):
        iterate_times = 1000
        initial_state = np.zeros(shape=(1, Model.num_states), dtype=DTYPE)
        open_loop_model = Model(initial_state=initial_state)

        input_dim = Model.num_states
        output_dim = Model.output_dim

        input_signal = Mock(spec=ISignal)
        input_signal.get_arr.return_value = np.zeros(shape=(iterate_times, input_dim), dtype=DTYPE)
        
        expected_output = np.zeros(shape=(iterate_times, output_dim), dtype=DTYPE)

        output = open_loop_model.run(input_signal, dtype=DTYPE)

        assert isinstance(output, np.ndarray)
        self.assertAlmostEqual(expected_output.all(), output.all())

    def test_open_loop_with_zeros_inputs_and_non_zero_intial_states_full_state_input_1000_iterate_times(self):
        iterate_times = 1000
        initial_state = np.random.rand(1, Model.num_states)
        open_loop_model = Model(initial_state=initial_state)

        input_dim = Model.num_states
        output_dim = Model.output_dim

        input_signal = Mock(spec=ISignal)
        input_signal.get_arr.return_value = np.zeros(shape=(iterate_times, input_dim), dtype=DTYPE)

        non_expected_output = np.zeros(shape=(iterate_times, output_dim), dtype=DTYPE)

        output = open_loop_model.run(input_signal, dtype=DTYPE)

        assert isinstance(output, np.ndarray)
        assert output.any() != non_expected_output.any()
