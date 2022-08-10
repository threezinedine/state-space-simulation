import unittest
from unittest.mock import Mock
import pytest
from parameterized import parameterized
import numpy as np
from models.timer import Timer


DTYPE = np.float32


class TestTimer(unittest.TestCase):
    @parameterized.expand([
            [0, .1, 1, np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9], dtype=DTYPE)],
            [-1, .2, 1, np.array([-1., -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8], dtype=DTYPE)]
        ])
    def test_timer_get_time_input_function(self, start, interval, stop, expected):
        timer = Timer(start=start, stop=stop, iterval=interval, dtype=DTYPE)

        output = timer.get_time_input()

        print(output, expected)
        self.assertListEqual(output.tolist(), expected.tolist())

    @parameterized.expand([
            [0, .1, 1, 10],
            [0, .2, 1, 5],
            [-1, .1, 1, 20],
        ])
    def test_timer_get_length_function(self, start, interval, stop, expected_length):
        timer = Timer(start=start, stop=stop, iterval=interval)

        output = timer.get_length()

        assert output == expected_length
