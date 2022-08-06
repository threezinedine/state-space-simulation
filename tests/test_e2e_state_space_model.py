import unittest
import pytest
import numpy as np
from models import Model
from models.signals import Impulse
from models.timer import Timer
from models.plants import Plant


DTYPE = np.float32


class TestE2EStateSpaceModel(unittest.TestCase):
    def test_impulse_signal_input_DTYPE_type(self):
        timer = Timer()
        signal = Impulse(timer)
        plant = Plant()
        model = Model(plant, dtype=DTYPE)

        output = model.run(signal)

        assert isinstance(output, np.ndarray)
        assert output.dtype == DTYPE

    def test_impulse_signal_input_int32_type(self):
        timer = Timer()
        signal = Impulse(timer)
        plant = Plant()
        model = Model(plant, dtype=np.int32)

        output = model.run(signal)

        assert isinstance(output, np.ndarray)
        assert output.dtype == np.int32
