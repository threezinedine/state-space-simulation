import unittest
import pytest
import numpy as np
from parameterized import parameterized
from models import Model
from models.signals import Impulse, FullStateFeedBackController
from models.timer import Timer
from models.plants import Plant


DTYPE = np.float32


class TestE2EStateSpaceModel(unittest.TestCase):
    @parameterized.expand([
            [DTYPE, 0, 1, .1, 10],
            [np.int32, 0, 1, .2, 5],
            [np.float64, 0, 2, .1, 20],
            [np.int64, -1, 1, .2, 10]
        ])
    def test_impulse_signal_input_DTYPE_type(self, dtype, start, stop, interval, expected_output):
        timer = Timer(start=start, stop=stop, iterval=interval)
        signal = Impulse(timer)
        plant = Plant()
        output_dim = Plant.output_dim
        model = Model(plant, dtype=dtype)

        output = model.run(signal)

        assert isinstance(output, np.ndarray)
        assert output.dtype == dtype
        self.assertTupleEqual(output.shape, (expected_output , output_dim))


    def test_full_state_feed_back_signal(self):
        timer = Timer()
        plant = Plant()
        signal = FullStateFeedBackController(timer, feedback_model=plant)
        model = Model(plant)

        output = model.run(signal)

        assert isinstance(output, np.ndarray)
