import unittest
from unittest.mock import Mock
import pytest
from parameterized import parameterized
from models.timer import ITimer
from models.signals import Impulse
import numpy as np


DTYPE = np.float32


class TestImpulse(unittest.TestCase):
    @parameterized.expand([
            [np.array([-.1, 0., .1, .2, .3], dtype=DTYPE), np.array([0., 1., 0., 0., 0.], dtype=DTYPE)],
            [np.array([-.2, -.1, 0., .1, .2], dtype=DTYPE), np.array([0., 0., 1., 0., 0.], dtype=DTYPE)],
            [np.array([.1, .15, .2], dtype=DTYPE), np.array([0, 0, 0], dtype=DTYPE)]
        ])
    def test_impulse_get_input_function_with_different_time(self, input_time, expected):
        timer = Mock(spec=ITimer)  
        timer.get_time_input.return_value = input_time
        impulse = Impulse(timer)

        output = impulse.get_input(dtype=DTYPE)

        self.assertTupleEqual(expected.shape, output.shape)
        assert isinstance(output, np.ndarray)
        self.assertListEqual(output.tolist(), expected.tolist())
