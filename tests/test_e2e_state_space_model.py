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
            [DTYPE],
            [np.int32],
            [np.float64],
            [np.int64]
        ])
    def test_impulse_signal_input_DTYPE_type(self, dtype):
        timer = Timer()
        signal = Impulse(timer)
        plant = Plant()
        model = Model(plant, dtype=dtype)

        output = model.run(signal)

        assert isinstance(output, np.ndarray)
        assert output.dtype == dtype

    def test_full_state_feed_back_signal(self):
        timer = Timer()
        plant = Plant()
        model = Model(plant)
        signal = FullStateFeedBackController(timer, feedback_model=model)

        output = model.run(signal)

        assert isinstance(output, np.ndarray)
