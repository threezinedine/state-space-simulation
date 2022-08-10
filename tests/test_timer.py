import unittest
from unittest.mock import Mock
import pytest
from parameterized import parameterized
import numpy as np
from models.timer import Timer


class TestTimer(unittest.TestCase):
    @parameterized.expand([
            [0, .1, 10]
        ])
    def test_timer_get_length_function(self, start, interval, iterate_times):
        pass
